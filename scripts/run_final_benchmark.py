#!/usr/bin/env python3
"""Final benchmark — 100 XL instances × 3 seeds, 6+1+1 packed on 32 cores.

Production knobs (greedy + k, ω=10, divisor 35, async BG + async SP). Four
exclusive 8-core slices per wave. Jobs are ordered by instance size then seed
so three copies of XL-n10001 never share a wave.

Default: 100 × 3 = 300 runs, 2 h each → 75 waves ≈ 150 h wall.

Usage (orchestrator)::

    python scripts/run_final_benchmark.py --dry-run
    python scripts/run_final_benchmark.py --cpu-base 0 --total-cores 32

Usage (single slice)::

    python scripts/run_final_benchmark.py --single \\
        --instance data/instances/xl/XL-n2307-k34.vrp \\
        --seed 100 --cpus 0-7
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drispi.pipeline.config_io import load_config  # noqa: E402
from drispi.pipeline.cores import parse_cpu_list  # noqa: E402
from drispi.pipeline.pipeline import DRISPIPipeline, make_run_label  # noqa: E402
from drispi.pipeline.runner import DEFAULT_BKS_FILE, load_instance_from_vrp_path  # noqa: E402
from drispi.utils.metrics import gap_to_bks, resolve_bks_cost  # noqa: E402

_STAGNATION_DISABLED = 10**9
_DEFAULT_SEEDS = [100, 200, 300]
_DEFAULT_MARKS_MIN = [30, 60, 90, 120]
_DEFAULT_INSTANCES_DIR = _ROOT / "data/instances/xl"
_DEFAULT_CONFIG = _ROOT / "configs/final_benchmark.yaml"
_N_RE = re.compile(r"n(\d+)", re.IGNORECASE)


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("final_benchmark")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _format_wall_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _cpu_spec(cpus: list[int]) -> str:
    if not cpus:
        raise ValueError("empty CPU list")
    if cpus == list(range(cpus[0], cpus[-1] + 1)):
        return f"{cpus[0]}-{cpus[-1]}"
    return ",".join(str(c) for c in cpus)


def _slice_cpus(cpu_base: int, slot: int, cores_per_slice: int) -> list[int]:
    start = cpu_base + slot * cores_per_slice
    return list(range(start, start + cores_per_slice))


def _instance_n(path: Path) -> int:
    match = _N_RE.search(path.stem)
    return int(match.group(1)) if match else 0


def _discover_instances(raw: list[str] | None) -> list[Path]:
    if not raw:
        paths = sorted(_DEFAULT_INSTANCES_DIR.glob("*.vrp"), key=_instance_n)
        return paths
    seen: set[str] = set()
    out: list[Path] = []
    for item in raw:
        path = Path(item).expanduser()
        matches: list[Path]
        if path.is_dir():
            matches = sorted(path.glob("*.vrp"), key=_instance_n)
        elif path.is_file():
            matches = [path]
        else:
            matches = sorted((p for p in Path().glob(item) if p.suffix == ".vrp"), key=_instance_n)
            if not matches and "*" not in item:
                raise FileNotFoundError(item)
        for match in matches:
            resolved = str(match.expanduser().resolve()) if match.exists() else str(match)
            if resolved not in seen:
                seen.add(resolved)
                out.append(match.expanduser().resolve() if match.exists() else match)
    return out


def _append_campaign_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, payload)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _load_completed(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        done.add((str(row["instance_stem"]), int(row["seed"])))
    return done


def _gap_pct(cost: float, instance_name: str, bks_cost: float | None) -> float | None:
    if bks_cost is None or bks_cost <= 0 or not math.isfinite(cost):
        return None
    try:
        return gap_to_bks(cost, instance_name, {instance_name: bks_cost})
    except (KeyError, ZeroDivisionError):
        return (cost - bks_cost) / bks_cost * 100.0


def _watch_wall_marks(
    pipeline: DRISPIPipeline,
    marks_min: list[int],
    stop: threading.Event,
    out: list[dict[str, Any]],
    *,
    bks_cost: float | None,
    seed: int,
    instance_name: str,
) -> None:
    while pipeline._start_time == 0.0 and not stop.wait(0.05):
        pass
    t0 = pipeline._start_time
    remaining = list(marks_min)
    while remaining and not stop.is_set():
        elapsed = time.perf_counter() - t0
        due = remaining[0] * 60.0
        if elapsed < due:
            stop.wait(min(1.0, due - elapsed))
            continue
        mark = remaining.pop(0)
        cost = float(pipeline.best_cost)
        gap = _gap_pct(cost, instance_name, bks_cost)
        out.append(
            {
                "type": "wall_mark",
                "mark_min": mark,
                "wall_clock_s": round(elapsed, 3),
                "iteration": int(pipeline.iterations_completed),
                "best_cost": cost if math.isfinite(cost) else None,
                "gap_to_bks_pct": None if gap is None else round(gap, 4),
                "pool_size": int(pipeline._pool.size()),
                "no_improve": int(pipeline._no_improve_count),
                "seed": seed,
            }
        )


def _run_single(
    *,
    instance_path: Path,
    seed: int,
    cpus: str,
    config_path: Path,
    output_root: Path,
    duration_hours: float,
    marks_min: list[int],
    bks_file: Path,
    campaign_jsonl: Path | None,
) -> dict[str, Any]:
    instance_path = instance_path.expanduser().resolve()
    inst = load_instance_from_vrp_path(instance_path)
    bks_cost = resolve_bks_cost(inst.name, bks_file=bks_file)
    cfg = load_config(config_path)
    run_output = output_root / inst.name / f"seed{seed}"
    cfg = replace(
        cfg,
        seed=int(seed),
        time_limit=float(duration_hours) * 3600.0,
        max_no_improve=_STAGNATION_DISABLED,
        run_analysis=False,
        output_dir=run_output,
    )
    run_label = make_run_label(inst.name)
    pipeline = DRISPIPipeline(
        inst,
        cfg,
        bks_cost=bks_cost,
        run_label=run_label,
        instance_id=inst.name,
        cli_cpus=cpus,
    )
    marks: list[dict[str, Any]] = []
    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch_wall_marks,
        args=(pipeline, marks_min, stop, marks),
        kwargs={"bks_cost": bks_cost, "seed": seed, "instance_name": inst.name},
        daemon=True,
    )
    t0 = time.perf_counter()
    watcher.start()
    try:
        pipeline.run()
    finally:
        stop.set()
        watcher.join(timeout=5.0)
    elapsed = time.perf_counter() - t0
    cost = float(pipeline.best_cost)
    gap = _gap_pct(cost, inst.name, bks_cost)
    row: dict[str, Any] = {
        "campaign": "final_benchmark",
        "instance": inst.name,
        "instance_stem": instance_path.stem,
        "instance_path": str(instance_path),
        "seed": int(seed),
        "cpus": cpus,
        "duration_hours": float(duration_hours),
        "marks_min": list(marks_min),
        "wall_marks": marks,
        "final_best_cost": cost if math.isfinite(cost) else None,
        "gap_to_bks": gap,
        "wall_clock_s": round(elapsed, 3),
        "iterations": int(pipeline.iterations_completed),
        "stop_reason": pipeline.stop_reason,
        "run_dir": str(pipeline.run_dir),
        "status": "ok",
    }
    if campaign_jsonl is not None:
        _append_campaign_jsonl(campaign_jsonl, row)
    return row


def _build_jobs(
    instances: list[Path],
    seeds: list[int],
    *,
    cpu_base: int,
    cores_per_slice: int,
    slices_per_wave: int,
) -> list[dict[str, Any]]:
    if slices_per_wave < 1:
        raise ValueError(f"slices_per_wave must be >= 1, got {slices_per_wave}")
    ordered = sorted(instances, key=_instance_n)
    flat: list[tuple[Path, int]] = []
    for seed in seeds:
        for inst in ordered:
            flat.append((inst, int(seed)))
    jobs: list[dict[str, Any]] = []
    for i, (inst, seed) in enumerate(flat):
        wave_idx = i // slices_per_wave + 1
        slot = i % slices_per_wave
        cpus = _slice_cpus(cpu_base, slot, cores_per_slice)
        jobs.append(
            {
                "wave": wave_idx,
                "slot": slot,
                "instance_path": str(inst),
                "instance_stem": inst.stem,
                "seed": seed,
                "cpus": _cpu_spec(cpus),
            }
        )
    return jobs


def _fmt_mean_std(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:+.3f} (std n/a)"
    return f"{mean(vals):+.3f} ± {stdev(vals):.3f}"


def _print_summary(rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("gap_to_bks") is not None]
    print()
    print("=" * 78)
    print("Final benchmark summary (gap_to_bks %, lower is better)")
    print("=" * 78)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_fail = sum(1 for r in rows if r.get("status") != "ok")
    print(f"Rows: {len(rows)}  ok={n_ok}  failed={n_fail}  with_gap={len(ok)}")
    if not ok:
        print("No completed rows with gap_to_bks to summarize.")
        return
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in ok:
        by_seed[int(row["seed"])].append(float(row["gap_to_bks"]))
    print()
    print(f"  {'seed':>8}  {'n':>4}  {'mean ± std':>16}")
    for seed in sorted(by_seed):
        vals = by_seed[seed]
        print(f"  {seed:8d}  {len(vals):4d}  {_fmt_mean_std(vals):>16}")
    all_gaps = [float(r["gap_to_bks"]) for r in ok]
    print(f"  {'all':>8}  {len(all_gaps):4d}  {_fmt_mean_std(all_gaps):>16}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Final benchmark: 100 XL × seeds 100/200/300, 6+1+1×4 packed slices, "
            "2 h, SP on, time_limit only."
        )
    )
    p.add_argument("--single", action="store_true", help="Run one (instance, seed, cpus) slice")
    p.add_argument("--instance", type=Path, default=None, help="VRP path (required with --single)")
    p.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Instance files, directories, or globs (default: data/instances/xl/*.vrp)",
    )
    p.add_argument("--seed", type=int, default=None, help="Seed (required with --single)")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=_DEFAULT_SEEDS,
        help="Orchestrator seeds (default: 100 200 300)",
    )
    p.add_argument("--cpus", type=str, default=None, help='Slice CPUs, e.g. "0-7" (with --single)')
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "artifacts/final_benchmark/runs",
    )
    p.add_argument(
        "--campaign-jsonl",
        type=Path,
        default=_ROOT / "artifacts/final_benchmark/campaign.jsonl",
    )
    p.add_argument("--bks-file", type=Path, default=DEFAULT_BKS_FILE)
    p.add_argument("--duration-hours", type=float, default=2.0)
    p.add_argument("--marks-min", type=int, nargs="+", default=_DEFAULT_MARKS_MIN)
    p.add_argument("--cpu-base", type=int, default=0)
    p.add_argument("--total-cores", type=int, default=32)
    p.add_argument("--cores-per-slice", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip (instance, seed) pairs already ok in campaign.jsonl",
    )
    return p.parse_args(argv)


def _resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = _ROOT / path
    return path.resolve()


def _run_single_from_args(args: argparse.Namespace) -> None:
    if args.instance is None or args.seed is None or not args.cpus:
        raise SystemExit("--single requires --instance, --seed, and --cpus")
    parse_cpu_list(args.cpus)
    row = _run_single(
        instance_path=args.instance,
        seed=int(args.seed),
        cpus=args.cpus,
        config_path=_resolve_repo_path(args.config),
        output_root=_resolve_repo_path(args.output_dir),
        duration_hours=float(args.duration_hours),
        marks_min=list(args.marks_min),
        bks_file=_resolve_repo_path(args.bks_file),
        campaign_jsonl=_resolve_repo_path(args.campaign_jsonl),
    )
    print(json.dumps(row, default=str))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.single:
        _run_single_from_args(args)
        return

    log = _setup_logger()
    instances = _discover_instances(args.instances)
    if not instances:
        raise SystemExit("No .vrp instances found")
    seeds = [int(s) for s in args.seeds]
    if args.total_cores % args.cores_per_slice != 0:
        raise SystemExit("total-cores must be divisible by cores-per-slice")
    slices_per_wave = args.total_cores // args.cores_per_slice
    if len(instances) < slices_per_wave:
        log.warning(
            f"Only {len(instances)} instances for {slices_per_wave} slices/wave; "
            "the same instance may run more than one seed in a wave"
        )
    jobs = _build_jobs(
        instances,
        seeds,
        cpu_base=args.cpu_base,
        cores_per_slice=args.cores_per_slice,
        slices_per_wave=slices_per_wave,
    )
    campaign_jsonl = _resolve_repo_path(args.campaign_jsonl)
    completed = set() if args.no_resume else _load_completed(campaign_jsonl)
    pending = [
        j for j in jobs if (j["instance_stem"], int(j["seed"])) not in completed
    ]
    n_waves = max((j["wave"] for j in jobs), default=0)
    est = len(pending) / max(slices_per_wave, 1) * args.duration_hours
    log.info(
        f"{len(instances)} instances × {len(seeds)} seeds = {len(jobs)} runs, "
        f"{slices_per_wave}/wave, {n_waves} waves"
    )
    log.info(f"Seeds: {seeds}")
    log.info(f"Already ok (resume): {len(completed)}  pending: {len(pending)}")
    log.info(f"Estimated remaining wall: ~{est:.1f} h  jsonl={campaign_jsonl}")
    if args.dry_run:
        for job in pending[:12]:
            log.info(
                f"  wave {job['wave']:>3} slot {job['slot']}  "
                f"{job['instance_stem']:<22} seed={job['seed']}  {job['cpus']}"
            )
        if len(pending) > 12:
            log.info(f"  … {len(pending) - 12} more")
        return

    campaign_jsonl.parent.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    campaign_t0 = time.perf_counter()
    n_ok = 0
    n_fail = 0
    waves: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for job in pending:
        waves[int(job["wave"])].append(job)

    for wave_idx in sorted(waves):
        wave = sorted(waves[wave_idx], key=lambda j: int(j["slot"]))
        log.info(f"═══ Wave {wave_idx}/{n_waves} ({len(wave)} slices) ═══")
        procs: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
        for job in wave:
            cmd = [
                sys.executable,
                str(script_path),
                "--single",
                "--instance",
                job["instance_path"],
                "--seed",
                str(job["seed"]),
                "--cpus",
                job["cpus"],
                "--config",
                str(_resolve_repo_path(args.config)),
                "--output-dir",
                str(_resolve_repo_path(args.output_dir)),
                "--campaign-jsonl",
                str(campaign_jsonl),
                "--bks-file",
                str(_resolve_repo_path(args.bks_file)),
                "--duration-hours",
                str(args.duration_hours),
                "--marks-min",
                *[str(m) for m in args.marks_min],
            ]
            log.info(
                f"  slot {job['slot']}  {job['instance_stem']}  "
                f"seed={job['seed']}  cpus={job['cpus']}"
            )
            procs.append(
                (
                    job,
                    subprocess.Popen(
                        cmd,
                        cwd=str(_ROOT),
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    ),
                )
            )
        for job, proc in procs:
            rc = proc.wait()
            if rc == 0:
                n_ok += 1
            else:
                n_fail += 1
                fail_row = {
                    "campaign": "final_benchmark",
                    "instance": job["instance_stem"],
                    "instance_stem": job["instance_stem"],
                    "instance_path": job["instance_path"],
                    "seed": job["seed"],
                    "cpus": job["cpus"],
                    "status": "error",
                    "error": f"slice exited {rc}",
                }
                _append_campaign_jsonl(campaign_jsonl, fail_row)
                log.error(f"  FAIL {job['instance_stem']} seed={job['seed']} rc={rc}")
        log.info(f"═══ Wave {wave_idx}/{n_waves} complete ═══")

    loaded: list[dict[str, Any]] = []
    if campaign_jsonl.is_file():
        for line in campaign_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                loaded.append(json.loads(line))
        _print_summary(loaded)
    elapsed = time.perf_counter() - campaign_t0
    log.info(
        f"Final benchmark finished: {n_ok} ok / {n_fail} failed this session in "
        f"{_format_wall_time(elapsed)}  summary={campaign_jsonl}"
    )
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
