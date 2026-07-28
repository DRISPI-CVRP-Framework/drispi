#!/usr/bin/env python3
"""A/B test for BG-AILS pair selection + chain-count (bundled variants).

Compares two hardcoded configs with SC/SP disabled:
  baseline: stochastic pair selection, n_chains = k-1
  new:      greedy (argmax) pair selection, n_chains = k

Runs are parallelized like ``run_benchmark.py`` via a shared CoreManager:
  max_parallel = total_cores // cores_per_instance

Usage:
    python scripts/ab_test_bg_ails.py \\
        --total-cores 384 --cores-per-instance 8 \\
        --output artifacts/ab_bg_ails/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# Cap BLAS / OpenMP threads before importing numpy / spawning workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import colorlog

from drispi.core.types import Route
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.config_io import load_config
from drispi.pipeline.core_manager import CoreManager
from drispi.pipeline.pipeline import DRISPIPipeline, make_run_label
from drispi.pipeline.runner import DEFAULT_BKS_FILE, load_instance_from_vrp_path
from drispi.utils.metrics import gap_to_bks, resolve_bks_cost

_SP_SC_DISABLED_WARMUP = 10**9
_STAGNATION_DISABLED = 10**9
_SAMPLE_INTERVAL_S = 60.0

_DEFAULT_INSTANCES = [
    _ROOT / "data/instances/xl/XL-n3975-k687.vrp",
    _ROOT / "data/instances/xl/XL-n9571-k55.vrp",
    _ROOT / "data/instances/xl/XL-n8389-k2028.vrp",
]
_DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]

VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {
        "bg_ails_pair_selection": "stochastic",
        "bg_ails_n_chains_mode": "k_minus_1",
    },
    "new": {
        "bg_ails_pair_selection": "greedy",
        "bg_ails_n_chains_mode": "k",
    },
}

_jsonl_lock = threading.Lock()


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ab_test_bg_ails")
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


def _format_run_tag(done: int, total: int) -> str:
    return f"Run {done}/{total}"


def _format_wall_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, _ = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row) + "\n"
    with _jsonl_lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


class TrackingPipeline(DRISPIPipeline):
    """DRISPIPipeline that records best-so-far objective on a wall-clock cadence."""

    def __init__(
        self,
        *args: Any,
        sample_interval_s: float = _SAMPLE_INTERVAL_S,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sample_interval_s = float(sample_interval_s)
        self._best_so_far: list[dict[str, float]] = []
        self._next_sample_at = 0.0

    @property
    def best_so_far(self) -> list[dict[str, float]]:
        return list(self._best_so_far)

    def _record_sample(self, elapsed_s: float) -> None:
        cost = float(self._best_cost)
        if not math.isfinite(cost):
            return
        self._best_so_far.append(
            {"t_s": round(elapsed_s, 3), "best_cost": cost}
        )

    def _maybe_sample(self, elapsed_s: float) -> None:
        while elapsed_s >= self._next_sample_at:
            self._record_sample(elapsed_s)
            self._next_sample_at += self._sample_interval_s

    def run(self) -> list[Route]:
        self._start_time = time.perf_counter()
        self._write_status("running")
        iteration = 0
        self._best_so_far = []
        self._next_sample_at = 0.0
        try:
            while True:
                elapsed = time.perf_counter() - self._start_time
                if iteration > 0 and elapsed >= self._config.time_limit:
                    self._stopped_by_time = True
                    break
                if iteration > 0 and self._no_improve_count >= self._config.max_no_improve:
                    self._stopped_by_no_improve = True
                    break

                self._run_iteration(iteration)
                iteration += 1

                elapsed = time.perf_counter() - self._start_time
                self._maybe_sample(elapsed)
        except KeyboardInterrupt:
            self._cancelled = True

        self._iterations_completed = iteration
        final_elapsed = time.perf_counter() - self._start_time
        # Always record an endpoint sample (dedupe if last cadence sample is close).
        if (
            not self._best_so_far
            or abs(self._best_so_far[-1]["t_s"] - final_elapsed) > 0.5
        ):
            self._record_sample(final_elapsed)
        self._finalize()
        if self._best_solution is None:
            return []
        return self._best_solution


def _build_config(
    base: DRISPIConfig,
    *,
    seed: int,
    cores_per_instance: int,
    duration_minutes: float,
    output_dir: Path,
    variant_fields: dict[str, str],
) -> DRISPIConfig:
    return replace(
        base,
        warmup_iterations=_SP_SC_DISABLED_WARMUP,
        max_no_improve=_STAGNATION_DISABLED,
        seed=seed,
        n_workers=cores_per_instance,
        time_limit=float(duration_minutes) * 60.0,
        output_dir=output_dir,
        run_analysis=False,
        bg_ails_pair_selection=variant_fields["bg_ails_pair_selection"],
        bg_ails_n_chains_mode=variant_fields["bg_ails_n_chains_mode"],
    )


def _gap_pct(cost: float, instance_name: str, bks_file: Path) -> float | None:
    bks_cost = resolve_bks_cost(instance_name, bks_file=bks_file)
    if bks_cost is None:
        return None
    return gap_to_bks(cost, instance_name, {instance_name: bks_cost})


def _run_one(
    *,
    instance_path: Path,
    seed: int,
    variant: str,
    variant_fields: dict[str, str],
    base_config: DRISPIConfig,
    cores_per_instance: int,
    duration_minutes: float,
    run_output_dir: Path,
    bks_file: Path,
    core_manager: CoreManager,
) -> dict[str, Any]:
    instance = load_instance_from_vrp_path(instance_path)
    config = _build_config(
        base_config,
        seed=seed,
        cores_per_instance=cores_per_instance,
        duration_minutes=duration_minutes,
        output_dir=run_output_dir,
        variant_fields=variant_fields,
    )
    job_id = f"{instance.name}_{variant}_s{seed}"
    run_label = make_run_label(job_id)
    pipeline = TrackingPipeline(
        instance,
        config,
        run_label=run_label,
        core_manager=core_manager,
        instance_id=job_id,
        sample_interval_s=_SAMPLE_INTERVAL_S,
    )

    t0 = time.perf_counter()
    pipeline.run()
    wall_clock_s = time.perf_counter() - t0

    final_objective = float(pipeline.best_cost)
    if not math.isfinite(final_objective):
        final_objective = float("nan")

    return {
        "instance": instance.name,
        "instance_path": str(instance_path),
        "seed": seed,
        "variant": variant,
        "bg_ails_pair_selection": variant_fields["bg_ails_pair_selection"],
        "bg_ails_n_chains_mode": variant_fields["bg_ails_n_chains_mode"],
        "final_objective": final_objective,
        "gap_to_bks": _gap_pct(final_objective, instance.name, bks_file),
        "wall_clock_s": round(wall_clock_s, 3),
        "best_so_far": pipeline.best_so_far,
        "stop_reason": pipeline.stop_reason,
        "run_dir": str(pipeline.run_dir),
        "status": "ok",
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_instance[str(row["instance"])].append(row)

    if not by_instance:
        print("No rows to summarize.")
        return

    print()
    print("=" * 72)
    print("A/B summary (gap_to_bks %, lower is better)")
    print("=" * 72)

    for instance, group in sorted(by_instance.items()):
        baseline = {
            int(r["seed"]): r for r in group if r["variant"] == "baseline"
        }
        new = {int(r["seed"]): r for r in group if r["variant"] == "new"}
        seeds = sorted(set(baseline) & set(new))

        def _gaps(variant_map: dict[int, dict[str, Any]]) -> list[float]:
            out: list[float] = []
            for s in seeds:
                g = variant_map[s].get("gap_to_bks")
                if g is not None and math.isfinite(float(g)):
                    out.append(float(g))
            return out

        b_gaps = _gaps(baseline)
        n_gaps = _gaps(new)

        def _fmt_mean_std(vals: list[float]) -> str:
            if not vals:
                return "n/a"
            if len(vals) == 1:
                return f"{vals[0]:+.3f} (std n/a)"
            return f"{mean(vals):+.3f} ± {stdev(vals):.3f}"

        print()
        print(f"Instance: {instance}  (n_seeds={len(seeds)})")
        print(f"  baseline mean gap: {_fmt_mean_std(b_gaps)}")
        print(f"  new      mean gap: {_fmt_mean_std(n_gaps)}")
        print(f"  {'seed':>6}  {'base_gap':>10}  {'new_gap':>10}  {'new-base':>10}")
        for s in seeds:
            bg = baseline[s].get("gap_to_bks")
            ng = new[s].get("gap_to_bks")
            if bg is None or ng is None:
                diff_s = "n/a"
                bg_s = "n/a" if bg is None else f"{float(bg):+.3f}"
                ng_s = "n/a" if ng is None else f"{float(ng):+.3f}"
            else:
                diff = float(ng) - float(bg)
                bg_s = f"{float(bg):+.3f}"
                ng_s = f"{float(ng):+.3f}"
                diff_s = f"{diff:+.3f}"
            print(f"  {s:6d}  {bg_s:>10}  {ng_s:>10}  {diff_s:>10}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A/B test BG-AILS: stochastic+k-1 (baseline) vs greedy+k (new), "
            "SC/SP disabled. Parallelizes jobs like run_benchmark.py."
        ),
    )
    parser.add_argument(
        "--instances",
        type=Path,
        nargs="+",
        default=_DEFAULT_INSTANCES,
        help="VRP instance paths (default: three XL pilots)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=_DEFAULT_SEEDS,
        help="Random seeds (default: 42..49)",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=120.0,
        help="Wall-clock limit per run in minutes (default: 120)",
    )
    parser.add_argument(
        "--total-cores",
        type=int,
        required=True,
        help="Total physical cores available on the machine",
    )
    parser.add_argument(
        "--cores-per-instance",
        type=int,
        required=True,
        help="Nominal cores per concurrent run (n_workers / CoreManager grant)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "artifacts/ab_bg_ails/results.jsonl",
        help="JSONL output path (one line per instance/seed/variant)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/default.yaml",
        help="Base YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--bks-file",
        type=Path,
        default=DEFAULT_BKS_FILE,
        help=f"BKS JSON for gap_to_bks (default: {DEFAULT_BKS_FILE})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=_ROOT / "artifacts/ab_bg_ails/runs",
        help="Directory for per-run pipeline outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and estimated wall time, then exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    log = _setup_logger()

    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    runs_dir = args.runs_dir.expanduser().resolve()
    bks_file = args.bks_file.expanduser().resolve()
    instances = [p.expanduser().resolve() for p in args.instances]
    seeds = list(args.seeds)

    for path in instances:
        if not path.is_file():
            log.error(f"Instance not found: {path}")
            sys.exit(1)

    if args.total_cores < 1 or args.cores_per_instance < 1:
        log.error("--total-cores and --cores-per-instance must be >= 1")
        sys.exit(1)

    max_parallel = args.total_cores // args.cores_per_instance
    if max_parallel < 1:
        log.error("--total-cores must be >= --cores-per-instance")
        sys.exit(1)

    base_config = load_config(config_path)

    jobs: list[tuple[Path, int, str]] = []
    for inst in instances:
        for seed in seeds:
            for variant in ("baseline", "new"):
                jobs.append((inst, seed, variant))

    total = len(jobs)
    duration_s = float(args.duration_minutes) * 60.0
    waves = math.ceil(total / max_parallel)
    est_runtime = waves * duration_s

    log.info(
        f"A/B BG-AILS: {len(instances)} instance(s) × {len(seeds)} seed(s) × "
        f"2 variants = {total} runs, {args.duration_minutes:.0f} min each"
    )
    log.info(
        f"Parallelism: {max_parallel} concurrent "
        f"({args.total_cores} cores, {args.cores_per_instance}/run) → "
        f"~{waves} wave(s), est. wall {_format_wall_time(est_runtime)}"
    )

    if args.dry_run:
        for inst_path, seed, variant in jobs:
            log.info(f"  would run  {inst_path.stem}  seed={seed}  variant={variant}")
        return

    if output_path.exists():
        output_path.unlink()

    core_manager = CoreManager(args.total_cores, args.cores_per_instance)
    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for inst_path, seed, variant in jobs:
            run_output_dir = runs_dir / inst_path.stem / f"seed{seed}" / variant
            run_output_dir.mkdir(parents=True, exist_ok=True)
            fut = executor.submit(
                _run_one,
                instance_path=inst_path,
                seed=seed,
                variant=variant,
                variant_fields=VARIANTS[variant],
                base_config=base_config,
                cores_per_instance=args.cores_per_instance,
                duration_minutes=args.duration_minutes,
                run_output_dir=run_output_dir,
                bks_file=bks_file,
                core_manager=core_manager,
            )
            futures[fut] = (inst_path, seed, variant)

        for fut in as_completed(futures):
            inst_path, seed, variant = futures[fut]
            done_count += 1
            tag = _format_run_tag(done_count, total)
            try:
                row = fut.result()
            except Exception as exc:
                log.error(
                    f"[{tag}] {inst_path.stem}  seed={seed}  variant={variant}  "
                    f"FAILED: {exc}"
                )
                row = {
                    "instance": inst_path.stem,
                    "instance_path": str(inst_path),
                    "seed": seed,
                    "variant": variant,
                    "status": "error",
                    "error": str(exc),
                    "final_objective": None,
                    "gap_to_bks": None,
                    "wall_clock_s": None,
                    "best_so_far": [],
                }
            else:
                gap = row.get("gap_to_bks")
                gap_s = f"{gap:+.3f}%" if gap is not None else "n/a"
                log.info(
                    f"[{tag}] {inst_path.stem}  seed={seed}  variant={variant}  "
                    f"obj={row['final_objective']:.1f}  gap={gap_s}  "
                    f"wall={row['wall_clock_s']:.1f}s"
                )

            _append_jsonl(output_path, row)
            rows.append(row)

    wall_elapsed = time.perf_counter() - wall_start
    _print_summary([r for r in rows if r.get("status") == "ok"])
    ok = sum(1 for r in rows if r.get("status") == "ok")
    log.info(
        f"Done: {ok}/{total} ok in {_format_wall_time(wall_elapsed)} → {output_path}"
    )


if __name__ == "__main__":
    main()
