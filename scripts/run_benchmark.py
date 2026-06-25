#!/usr/bin/env python3
"""Benchmark campaign runner with queue-based parallel execution."""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.config_io import load_config, merge_cli_overrides
from drispi.pipeline.core_manager import CoreManager
from drispi.pipeline.pipeline import DRISPIPipeline, make_run_label
from drispi.pipeline.runner import (
    DEFAULT_BKS_FILE,
    _CLI_TO_FIELD,
    load_instance_from_vrp_path,
)
from drispi.utils.metrics import resolve_bks_cost


def _gap_pct(cost: float, bks_cost: float | None) -> float | None:
    if bks_cost is None or bks_cost <= 0 or not math.isfinite(cost):
        return None
    return (cost - bks_cost) / bks_cost * 100.0


def _discover_instances(patterns: list[str]) -> list[Path]:
    """Expand globs and explicit file paths into a sorted, deduplicated list."""
    seen: set[str] = set()
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            path = Path(pattern)
            if path.is_file():
                matches = [str(path)]
        for match in matches:
            if match not in seen:
                seen.add(match)
                paths.append(Path(match))
    return sorted(paths)


def _format_wall_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, _ = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _build_config(args: argparse.Namespace) -> DRISPIConfig:
    config = DRISPIConfig()
    if args.config:
        config = load_config(args.config)
    overrides: dict[str, object] = {}
    for arg_name, field_name in _CLI_TO_FIELD.items():
        value = getattr(args, arg_name, None)
        if field_name == "run_analysis":
            if value is True:
                overrides[field_name] = True
        elif value is not None:
            overrides[field_name] = value
    if args.time_limit is not None:
        overrides["time_limit"] = args.time_limit
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    return merge_cli_overrides(config, overrides)


def _run_single_instance(
    instance_path: Path,
    config: DRISPIConfig,
    bks_cost: float | None,
    core_manager: CoreManager,
    instance_id: str,
) -> dict:
    """Load instance, create pipeline, run. Catch all exceptions."""
    t0 = time.perf_counter()
    try:
        inst = load_instance_from_vrp_path(instance_path)
        run_label = make_run_label(inst.name)
        pipeline = DRISPIPipeline(
            inst,
            config,
            bks_cost=bks_cost,
            run_label=run_label,
            core_manager=core_manager,
            instance_id=instance_id,
        )
        pipeline.run()
        runtime_s = time.perf_counter() - t0
        gap_pct = _gap_pct(pipeline.best_cost, bks_cost)
        reason = pipeline.stop_reason
        if reason == "time_limit":
            stop_reason = "time"
        elif reason == "no_improve":
            stop_reason = "no_improve"
        else:
            stop_reason = reason or "unknown"
        return {
            "instance": inst.name,
            "n_customers": inst.n_customers,
            "best_cost": pipeline.best_cost,
            "bks": bks_cost,
            "gap_pct": gap_pct,
            "runtime_s": runtime_s,
            "iterations": pipeline.iterations_completed,
            "stop_reason": stop_reason,
            "run_dir": str(pipeline.run_dir),
            "status": "ok",
            "error": None,
        }
    except Exception as exc:
        return {
            "instance": instance_path.stem,
            "n_customers": 0,
            "best_cost": float("inf"),
            "bks": bks_cost,
            "gap_pct": None,
            "runtime_s": time.perf_counter() - t0,
            "iterations": 0,
            "stop_reason": "",
            "run_dir": "",
            "status": "error",
            "error": str(exc),
        }


def _print_progress(idx: int, total: int, result: dict) -> None:
    name = result["instance"]
    if result["status"] == "error":
        print(f"[{idx}/{total}] {name:<16}  ERROR: {result['error']}")
        return
    best = result["best_cost"]
    gap = result["gap_pct"]
    gap_str = f"{gap:+.2f}%" if gap is not None else "N/A"
    runtime = int(result["runtime_s"])
    print(
        f"[{idx}/{total}] {name:<16}  done  "
        f"best={best:,.0f}  gap={gap_str}  time={runtime}s  ✓"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DRISPI benchmark campaign.")
    parser.add_argument(
        "instance_patterns",
        nargs="+",
        help='Instance paths or globs, e.g. "data/instances/xl/*.vrp" or inst1.vrp inst2.vrp',
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--total-cores", type=int, required=True)
    parser.add_argument("--cores-per-instance", type=int, required=True)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--bks-file", type=Path, default=DEFAULT_BKS_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    instance_paths = _discover_instances(args.instance_patterns)
    if not instance_paths:
        joined = ", ".join(args.instance_patterns)
        print(f"No instances matched: {joined}", file=sys.stderr)
        sys.exit(1)

    config = _build_config(args)
    max_parallel = args.total_cores // args.cores_per_instance
    if max_parallel < 1:
        print("total_cores must be >= cores_per_instance", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(instance_paths)} instance(s):")
    for p in instance_paths:
        print(f"  {p}")

    est_runtime = len(instance_paths) / max_parallel * config.time_limit
    print(
        f"\nParallelism: {max_parallel} instance(s), "
        f"{args.total_cores} cores ({args.cores_per_instance} nominal/instance)"
    )
    print(f"Time limit per instance: {config.time_limit:.0f}s")
    print(f"Estimated wall time: {_format_wall_time(est_runtime)}")
    print(f"Output: {config.output_dir}")

    if args.dry_run:
        return

    core_manager = CoreManager(args.total_cores, args.cores_per_instance)
    total = len(instance_paths)
    results: list[dict] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for path in instance_paths:
            inst_id = path.stem
            bks = resolve_bks_cost(inst_id, bks_file=args.bks_file)
            fut = executor.submit(
                _run_single_instance,
                path,
                config,
                bks,
                core_manager,
                inst_id,
            )
            futures[fut] = path

        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            result = fut.result()
            results.append(result)
            _print_progress(done_count, total, result)

    wall_elapsed = time.perf_counter() - wall_start
    ok = sum(1 for r in results if r["status"] == "ok")
    errors = total - ok
    print("═" * 43)
    print(f"Benchmark complete: {total}/{total} instances")
    print(f"Successful: {ok}")
    print(f"Errors:      {errors}")
    print(f"Total wall time: {_format_wall_time(wall_elapsed)}")
    print(f"Results in: {config.output_dir}/")
    print("═" * 43)


if __name__ == "__main__":
    main()
