#!/usr/bin/env python3
"""Sweep Gurobi Threads on RoutePool snapshots from capture_pool_snapshots.py.

For each snapshot × thread count × mode × repeat, runs the production LP-then-MIP
SC/SP pipeline (``build_and_solve``) with ``Threads`` injected, and appends
JSONL rows for thread-count vs wall-time analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Isolate Gurobi threading from BLAS/OpenMP oversubscription (same as Docker).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import colorlog

from drispi.pipeline.runner import load_instance_from_vrp_path
from drispi.route_pool.pool import RoutePool
from drispi.sp.model import build_and_solve

# Match production default when the sweep CLI does not expose --mip-gap.
_DEFAULT_MIP_GAP = 0.0005
_MANIFEST_NAME = "snapshot_manifest.json"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("sweep_gurobi_threads")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = colorlog.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s%(reset)s | %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
            },
        )
    )
    logger.addHandler(handler)
    return logger


def _format_sweep_tag(done: int, total: int) -> str:
    return f"Sweep {done}/{total}"


def _resolve_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["sp", "sc"]
    if mode in ("sp", "sc"):
        return [mode]
    raise ValueError(f"Unknown mode: {mode}")


def _load_manifest(snapshots_dir: Path) -> dict[str, Any]:
    path = snapshots_dir / _MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {_MANIFEST_NAME} in {snapshots_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pool(path: Path) -> RoutePool:
    with path.open("rb") as fh:
        pool = pickle.load(fh)
    if not isinstance(pool, RoutePool):
        raise TypeError(f"Expected RoutePool, got {type(pool).__name__}")
    return pool


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _print_summary(rows: list[dict[str, Any]], thread_counts: list[int]) -> None:
    """Print mean total wall-time per (instance, mark, mode) × threads."""
    groups: dict[tuple[str, int, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["instance"]), int(row["mark_min"]), str(row["mode"]))
        groups[key][int(row["threads"])].append(float(row["total_wall_s"]))

    if not groups:
        print("No successful solves to summarize.")
        return

    header = f"{'instance':<16} {'mark':>5} {'mode':>4}"
    for t in thread_counts:
        header += f"  t={t:>2}"
    print()
    print("Mean total wall-time (s) by thread count")
    print(header)
    print("-" * len(header))
    for (instance, mark, mode) in sorted(groups):
        line = f"{instance:<16} {mark:>4}m {mode:>4}"
        for t in thread_counts:
            vals = groups[(instance, mark, mode)].get(t, [])
            if vals:
                line += f"  {_mean(vals):5.1f}"
            else:
                line += f"  {'—':>5}"
        print(line)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Gurobi Threads on RoutePool snapshots (LP-then-MIP SC/SP)."
        ),
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        required=True,
        help="Directory with .pkl snapshots and snapshot_manifest.json",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        metavar="N",
        help="Thread counts to sweep (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeats per (snapshot, threads, mode) (default: 3)",
    )
    parser.add_argument(
        "--mode",
        choices=("sc", "sp", "both"),
        default="sp",
        help="Formulation to sweep (default: sp)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write JSONL results (one object per line)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Gurobi TimeLimit in seconds per MIP solve (default: 300)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    log = _setup_logger()

    if any(t < 1 for t in args.threads):
        print("All --threads values must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.repeats < 1:
        print("--repeats must be >= 1.", file=sys.stderr)
        sys.exit(1)
    if args.time_limit < 1:
        print("--time-limit must be >= 1.", file=sys.stderr)
        sys.exit(1)

    snapshots_dir = args.snapshots_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    thread_counts = list(args.threads)
    modes = _resolve_modes(args.mode)

    manifest = _load_manifest(snapshots_dir)
    instance_name = str(manifest.get("instance", "unknown"))
    seed = manifest.get("seed")
    instance_path_raw = manifest.get("instance_path")
    if not instance_path_raw:
        print("Manifest is missing instance_path.", file=sys.stderr)
        sys.exit(1)
    instance = load_instance_from_vrp_path(Path(instance_path_raw))

    snapshots = list(manifest.get("snapshots") or [])
    if not snapshots:
        print("Manifest has no snapshots.", file=sys.stderr)
        sys.exit(1)

    valid_snaps: list[dict[str, Any]] = []
    for snap in snapshots:
        filename = str(snap.get("filename", ""))
        pkl_path = snapshots_dir / filename
        try:
            if not filename or not pkl_path.is_file():
                raise FileNotFoundError(f"Snapshot file missing: {pkl_path}")
            pool = _load_pool(pkl_path)
            if pool.size() == 0:
                raise ValueError("RoutePool is empty")
            snap = {**snap, "pool_size": pool.size()}
            valid_snaps.append(snap)
        except Exception as exc:
            log.warning(f"[  SKIP   ] {filename or '<unknown>'}  ({exc})")

    if not valid_snaps:
        print("No valid snapshots to sweep.", file=sys.stderr)
        sys.exit(1)

    jobs: list[tuple[dict[str, Any], int, str, int]] = []
    for snap in valid_snaps:
        for threads in thread_counts:
            for mode in modes:
                for repeat in range(args.repeats):
                    jobs.append((snap, threads, mode, repeat))

    total = len(jobs)
    log.info(
        f"[{_format_sweep_tag(0, total)}] starting  snaps={len(valid_snaps)}  "
        f"threads={thread_counts}  modes={modes}  repeats={args.repeats}"
    )

    # Truncate / create output for a fresh sweep run.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    for done, (snap, threads, mode, repeat) in enumerate(jobs, start=1):
        filename = str(snap["filename"])
        mark_min = int(snap["mark_min"])
        pool_size_meta = snap.get("pool_size")
        pkl_path = snapshots_dir / filename
        tag = _format_sweep_tag(done, total)

        base_row: dict[str, Any] = {
            "instance": instance_name,
            "mark_min": mark_min,
            "mode": mode,
            "threads": threads,
            "repeat": repeat,
            "filename": filename,
            "pool_size": pool_size_meta,
            "seed": seed,
            "time_limit_s": args.time_limit,
            "mip_gap_tol": _DEFAULT_MIP_GAP,
        }

        use_sp = mode == "sp"
        try:
            # Fresh deserialize every repeat — no shared / mutated pool across runs.
            pool = _load_pool(pkl_path)
            _lp, _raw, timed_out, sol_count, metrics = build_and_solve(
                pool,
                instance,
                use_sp=use_sp,
                time_limit=float(args.time_limit),
                mip_gap=_DEFAULT_MIP_GAP,
                threads=threads,
            )
            row = {
                **base_row,
                "status": "ok",
                "error": None,
                "lp_wall_s": round(metrics.lp_wall_s, 6),
                "mip_wall_s": round(metrics.mip_wall_s, 6),
                "total_wall_s": round(metrics.total_wall_s, 6),
                "mip_gap": metrics.mip_gap,
                "objective": metrics.objective,
                "timed_out": timed_out,
                "sol_count": sol_count,
                "pool_size": pool.size(),
            }
            gap_s = "—" if metrics.mip_gap is None else f"{metrics.mip_gap:.4g}"
            log.info(
                f"[{tag}] {filename}  {mode}  thr={threads}  r={repeat}  "
                f"total={metrics.total_wall_s:.2f}s  "
                f"lp={metrics.lp_wall_s:.2f}s  mip={metrics.mip_wall_s:.2f}s  "
                f"gap={gap_s}  timeout={timed_out}"
            )
        except Exception as exc:
            log.error(
                f"[{tag}] FAIL {filename}  {mode}  thr={threads}  r={repeat}  ({exc})"
            )
            row = {
                **base_row,
                "status": "error",
                "error": str(exc),
                "lp_wall_s": None,
                "mip_wall_s": None,
                "total_wall_s": None,
                "mip_gap": None,
                "objective": None,
                "timed_out": None,
                "sol_count": None,
            }

        _append_jsonl(output_path, row)
        result_rows.append(row)

    _print_summary(result_rows, thread_counts)
    ok = sum(1 for r in result_rows if r.get("status") == "ok")
    print(f"\nWrote {len(result_rows)} rows ({ok} ok) → {output_path}")


if __name__ == "__main__":
    main()
