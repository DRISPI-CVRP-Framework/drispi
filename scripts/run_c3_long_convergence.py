#!/usr/bin/env python3
"""C3 long convergence / variance campaign (deployed 6+1+1 async).

Packs 3 instances × 4 seeds = 12 runs into 3 fully packed waves of 4 concurrent
8-core slices on a 32-core machine. Each slice gets a disjoint CPU island
(``--cpus``) so affinity matches the deployed 6+1+1 layout under real contention.

Wall-clock marks (default 1h/2h/4h/6h/8h) are written to each run's ``run.jsonl``
and to a campaign-level summary JSONL for trajectory / HAOS cold-start analysis.

Usage (orchestrator — preferred on host or inside one 32-core container)::

    python scripts/run_c3_long_convergence.py --dry-run
    python scripts/run_c3_long_convergence.py \\
        --cpu-base 0 --total-cores 32

Usage (single slice — for debugging or external launchers)::

    python scripts/run_c3_long_convergence.py --single \\
        --instance data/instances/xl/XL-n2307-k34.vrp \\
        --seed 42 --cpus 0-7
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drispi.core.types import Route
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.config_io import load_config
from drispi.pipeline.cores import parse_cpu_list
from drispi.pipeline.pipeline import DRISPIPipeline, make_run_label
from drispi.pipeline.runner import DEFAULT_BKS_FILE, load_instance_from_vrp_path
from drispi.utils.metrics import gap_to_bks, resolve_bks_cost

_STAGNATION_DISABLED = 10**9

_DEFAULT_INSTANCES = [
    _ROOT / "data/instances/xl/XL-n2307-k34.vrp",
    _ROOT / "data/instances/xl/XL-n3975-k687.vrp",
    _ROOT / "data/instances/xl/XL-n8389-k2028.vrp",
]
_DEFAULT_SEEDS = [42, 43, 44, 45]
_DEFAULT_MARKS_MIN = [60, 120, 240, 360, 480]  # 1h / 2h / 4h / 6h / 8h
_DEFAULT_CONFIG = _ROOT / "configs/c3_long_convergence.yaml"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("c3_long_convergence")
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
    # Compact consecutive ranges: [0,1,2,3,8,9] -> "0-3,8-9"
    parts: list[str] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def _slice_cpus(cpu_base: int, slice_idx: int, cores_per_slice: int) -> list[int]:
    start = cpu_base + slice_idx * cores_per_slice
    return list(range(start, start + cores_per_slice))


class MarkingPipeline(DRISPIPipeline):
    """Production pipeline that emits wall-clock trajectory marks to run.jsonl."""

    def __init__(
        self,
        *args: Any,
        marks_min: list[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._marks_min = sorted(int(m) for m in marks_min)
        self._pending_marks = list(self._marks_min)
        self._mark_records: list[dict[str, Any]] = []

    @property
    def mark_records(self) -> list[dict[str, Any]]:
        return list(self._mark_records)

    def run(self) -> list[Route]:
        self._start_time = time.perf_counter()
        self._write_status("running")
        iteration = 0
        self._pending_marks = list(self._marks_min)
        self._mark_records = []
        try:
            while True:
                elapsed = time.perf_counter() - self._start_time
                if iteration > 0 and elapsed >= self._config.time_limit:
                    self._emit_due_marks(elapsed, iteration)
                    self._stopped_by_time = True
                    break
                if iteration > 0 and self._no_improve_count >= self._config.max_no_improve:
                    self._stopped_by_no_improve = True
                    break

                self._run_iteration(iteration)
                iteration += 1

                elapsed = time.perf_counter() - self._start_time
                self._emit_due_marks(elapsed, iteration)
        except KeyboardInterrupt:
            self._cancelled = True

        self._iterations_completed = iteration
        final_elapsed = time.perf_counter() - self._start_time
        # Ensure the final mark fires if we stopped exactly on/after it.
        self._emit_due_marks(final_elapsed, iteration)
        self._finalize()
        if self._best_solution is None:
            return []
        return self._best_solution

    def _emit_due_marks(self, elapsed_s: float, iteration: int) -> None:
        while self._pending_marks and elapsed_s >= self._pending_marks[0] * 60.0:
            mark = self._pending_marks.pop(0)
            self._write_mark(mark, elapsed_s, iteration)

    def _write_mark(self, mark_min: int, elapsed_s: float, iteration: int) -> None:
        cost = float(self._best_cost)
        gap = None
        if self._bks_cost is not None and math.isfinite(cost) and self._bks_cost > 0:
            gap = (cost - self._bks_cost) / self._bks_cost * 100.0
        record = {
            "type": "wall_mark",
            "mark_min": mark_min,
            "wall_clock_s": round(elapsed_s, 3),
            "iteration": iteration,
            "best_cost": cost if math.isfinite(cost) else None,
            "gap_to_bks_pct": round(gap, 4) if gap is not None else None,
            "pool_size": self._pool.size(),
            "no_improve": self._no_improve_count,
            "seed": self._config.seed,
        }
        self._mark_records.append(record)
        if math.isfinite(cost):
            content = (
                f"mark={mark_min}min  wall={_format_wall_time(elapsed_s)}  "
                f"best={cost:.2f}"
            )
        else:
            content = (
                f"mark={mark_min}min  wall={_format_wall_time(elapsed_s)}  best=inf"
            )
        if gap is not None:
            content += f"  Δ_BKS={gap:+.2f}%"
        content += f"  iter={iteration}  pool={self._pool.size()}"
        self._logger._emit(  # noqa: SLF001 — reuse pipeline log channels
            iteration - 1 if iteration > 0 else None,
            " MARK   ",
            content,
            json_event=record,
        )


def _build_config(
    base: DRISPIConfig,
    *,
    seed: int,
    output_dir: Path,
    duration_hours: float,
) -> DRISPIConfig:
    return replace(
        base,
        seed=int(seed),
        output_dir=output_dir,
        time_limit=float(duration_hours) * 3600.0,
        max_no_improve=_STAGNATION_DISABLED,
        # Keep cores: block from YAML; n_workers ignored when present.
    )


def run_single(
    *,
    instance_path: Path,
    seed: int,
    cpus: str,
    config_path: Path,
    marks_min: list[int],
    duration_hours: float,
    output_dir: Path,
    bks_file: Path | None,
    campaign_jsonl: Path | None = None,
) -> dict[str, Any]:
    log = _setup_logger()
    base = load_config(config_path)
    config = _build_config(
        base,
        seed=seed,
        output_dir=output_dir,
        duration_hours=duration_hours,
    )
    instance = load_instance_from_vrp_path(instance_path)
    bks_cost = None
    if bks_file is not None:
        bks_cost = resolve_bks_cost(instance.name, bks_file=bks_file)

    run_label = make_run_label(f"{instance.name}_s{seed}")
    log.info(
        f"START  {instance.name}  seed={seed}  cpus={cpus}  "
        f"limit={duration_hours:g}h  marks_min={marks_min}"
    )
    t0 = time.perf_counter()
    pipeline = MarkingPipeline(
        instance,
        config,
        bks_cost=bks_cost,
        run_label=run_label,
        cli_cpus=cpus,
        marks_min=marks_min,
    )
    pipeline.run()
    wall_s = time.perf_counter() - t0

    final_cost = float(pipeline.best_cost)
    row: dict[str, Any] = {
        "campaign": "C3",
        "instance": instance.name,
        "instance_path": str(instance_path),
        "seed": seed,
        "cpus": cpus,
        "duration_hours": duration_hours,
        "marks_min": marks_min,
        "wall_marks": pipeline.mark_records,
        "final_best_cost": final_cost if math.isfinite(final_cost) else None,
        "gap_to_bks_pct": (
            gap_to_bks(final_cost, instance.name, {instance.name: bks_cost})
            if bks_cost is not None and math.isfinite(final_cost)
            else None
        ),
        "wall_clock_s": round(wall_s, 3),
        "iterations": pipeline.iterations_completed,
        "stop_reason": pipeline.stop_reason,
        "run_dir": str(pipeline.run_dir),
        "status": "ok",
    }

    if campaign_jsonl is not None:
        campaign_jsonl.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row) + "\n"
        # Serialize concurrent wave writers (4 slices append the same file).
        with campaign_jsonl.open("a", encoding="utf-8") as fh:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except ImportError:
                fh.write(line)

    log.info(
        f"DONE   {instance.name}  seed={seed}  "
        f"best={final_cost:.2f}  wall={_format_wall_time(wall_s)}  "
        f"marks={len(pipeline.mark_records)}  dir={pipeline.run_dir}"
    )
    return row


def _build_jobs(
    instances: list[Path],
    seeds: list[int],
    *,
    cpu_base: int,
    cores_per_slice: int,
    slices_per_wave: int,
) -> list[dict[str, Any]]:
    """One wave per instance (4 seeds) so every wave is fully packed."""
    if len(seeds) != slices_per_wave:
        raise ValueError(
            f"Need exactly {slices_per_wave} seeds for a full wave "
            f"(got {len(seeds)}: {seeds})"
        )
    jobs: list[dict[str, Any]] = []
    for wave_idx, inst in enumerate(instances):
        for slot, seed in enumerate(seeds):
            cpus = _slice_cpus(cpu_base, slot, cores_per_slice)
            jobs.append(
                {
                    "wave": wave_idx + 1,
                    "slot": slot,
                    "instance_path": str(inst),
                    "instance": inst.stem,
                    "seed": seed,
                    "cpus": _cpu_spec(cpus),
                }
            )
    return jobs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "C3 long convergence/variance campaign: 3×4 runs, 6+1+1×4 packed "
            "slices, wall-clock marks for trajectory."
        )
    )
    p.add_argument(
        "--single",
        action="store_true",
        help="Run one (instance, seed, cpus) slice and exit",
    )
    p.add_argument(
        "--instance",
        type=Path,
        default=None,
        help="VRP path (required with --single; else use --instances)",
    )
    p.add_argument(
        "--instances",
        type=Path,
        nargs="+",
        default=_DEFAULT_INSTANCES,
        help="Three VRP instances (default: n2307 / n3975 / n8389 pilots)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for --single mode",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=_DEFAULT_SEEDS,
        help="Four seeds → one full wave (default: 42 43 44 45)",
    )
    p.add_argument(
        "--cpus",
        type=str,
        default=None,
        help='CPU list for --single, e.g. "0-7"',
    )
    p.add_argument(
        "--cpu-base",
        type=int,
        default=0,
        help="First CPU id of the 32-core island (default: 0 → 0-31)",
    )
    p.add_argument(
        "--total-cores",
        type=int,
        default=32,
        help="Cores reserved for the campaign (default: 32)",
    )
    p.add_argument(
        "--cores-per-slice",
        type=int,
        default=8,
        help="Cores per 6+1+1 slice (default: 8)",
    )
    p.add_argument(
        "--duration-hours",
        type=float,
        default=8.0,
        help="Wall-clock limit per run in hours (default: 8)",
    )
    p.add_argument(
        "--marks",
        type=int,
        nargs="+",
        default=_DEFAULT_MARKS_MIN,
        help="Wall-clock marks in minutes (default: 60 120 240 360 480)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Deployed config YAML (default: {_DEFAULT_CONFIG.name})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "artifacts/c3_long_convergence/runs",
        help="Per-run pipeline output root",
    )
    p.add_argument(
        "--campaign-jsonl",
        type=Path,
        default=_ROOT / "artifacts/c3_long_convergence/campaign.jsonl",
        help="Append-only campaign summary (one line per finished run)",
    )
    p.add_argument(
        "--bks-file",
        type=Path,
        default=DEFAULT_BKS_FILE,
        help=f"BKS JSON (default: {DEFAULT_BKS_FILE})",
    )
    p.add_argument(
        "--no-bks",
        action="store_true",
        help="Skip BKS / gap_to_bks",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print wave/CPU plan and exit",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    log = _setup_logger()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        log.error(f"Config not found: {config_path}")
        sys.exit(1)

    marks = sorted(set(int(m) for m in args.marks))
    if any(m <= 0 for m in marks):
        log.error("All --marks must be positive minutes")
        sys.exit(1)

    bks_file: Path | None = None
    if not args.no_bks:
        bks_file = args.bks_file.expanduser().resolve()

    if args.single:
        if args.instance is None or args.seed is None or args.cpus is None:
            log.error("--single requires --instance, --seed, and --cpus")
            sys.exit(1)
        # Validate CPU list early.
        parse_cpu_list(args.cpus)
        run_single(
            instance_path=args.instance.expanduser().resolve(),
            seed=int(args.seed),
            cpus=args.cpus,
            config_path=config_path,
            marks_min=marks,
            duration_hours=float(args.duration_hours),
            output_dir=args.output_dir.expanduser().resolve(),
            bks_file=bks_file,
            campaign_jsonl=args.campaign_jsonl.expanduser().resolve(),
        )
        return

    instances = [p.expanduser().resolve() for p in args.instances]
    seeds = list(args.seeds)
    for path in instances:
        if not path.is_file():
            log.error(f"Instance not found: {path}")
            sys.exit(1)

    if args.total_cores < 1 or args.cores_per_slice < 1:
        log.error("--total-cores and --cores-per-slice must be >= 1")
        sys.exit(1)

    slices_per_wave = args.total_cores // args.cores_per_slice
    if slices_per_wave < 1:
        log.error("--total-cores must be >= --cores-per-slice")
        sys.exit(1)
    if slices_per_wave * args.cores_per_slice != args.total_cores:
        log.error(
            f"--total-cores ({args.total_cores}) must be divisible by "
            f"--cores-per-slice ({args.cores_per_slice})"
        )
        sys.exit(1)
    if len(seeds) != slices_per_wave:
        log.error(
            f"Need {slices_per_wave} seeds for a full packed wave "
            f"(got {len(seeds)}). Adjust --seeds or core layout."
        )
        sys.exit(1)
    if len(instances) < 1:
        log.error("Need at least one instance")
        sys.exit(1)

    jobs = _build_jobs(
        instances,
        seeds,
        cpu_base=args.cpu_base,
        cores_per_slice=args.cores_per_slice,
        slices_per_wave=slices_per_wave,
    )
    n_waves = len(instances)
    est_s = n_waves * float(args.duration_hours) * 3600.0

    log.info(
        f"C3 campaign: {len(instances)} instance(s) × {len(seeds)} seed(s) "
        f"= {len(jobs)} runs, {args.duration_hours:g}h each"
    )
    log.info(
        f"Packing: {slices_per_wave} concurrent 6+1+1 slices "
        f"(cpu_base={args.cpu_base}, {args.total_cores} cores) → "
        f"{n_waves} wave(s), est. wall {_format_wall_time(est_s)}"
    )
    log.info(f"Marks (min): {marks}")
    log.info(f"Config: {config_path}")

    for job in jobs:
        log.info(
            f"  wave {job['wave']}/{n_waves}  slot {job['slot']}  "
            f"{job['instance']:<20}  seed={job['seed']}  cpus={job['cpus']}"
        )

    if args.dry_run:
        log.info("Dry-run only — exiting")
        return

    output_dir = args.output_dir.expanduser().resolve()
    campaign_jsonl = args.campaign_jsonl.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Group jobs by wave; run each wave to completion before the next.
    by_wave: dict[int, list[dict[str, Any]]] = {}
    for job in jobs:
        by_wave.setdefault(int(job["wave"]), []).append(job)

    campaign_t0 = time.perf_counter()
    n_ok = 0
    n_fail = 0
    script_path = Path(__file__).resolve()

    for wave_idx in sorted(by_wave):
        wave_jobs = by_wave[wave_idx]
        log.info(f"═══ Wave {wave_idx}/{n_waves} — launching {len(wave_jobs)} slices ═══")
        # One subprocess per slice so sched_setaffinity / Gurobi / JVM stay isolated.
        procs: list[tuple[dict[str, Any], subprocess.Popen[bytes]]] = []
        for j in wave_jobs:
            cmd = [
                sys.executable,
                str(script_path),
                "--single",
                "--instance",
                j["instance_path"],
                "--seed",
                str(j["seed"]),
                "--cpus",
                j["cpus"],
                "--config",
                str(config_path),
                "--duration-hours",
                str(args.duration_hours),
                "--marks",
                *[str(m) for m in marks],
                "--output-dir",
                str(output_dir),
                "--campaign-jsonl",
                str(campaign_jsonl),
            ]
            if bks_file is None:
                cmd.append("--no-bks")
            else:
                cmd.extend(["--bks-file", str(bks_file)])
            log.info(f"  exec  {j['instance']}  seed={j['seed']}  cpus={j['cpus']}")
            procs.append(
                (
                    j,
                    subprocess.Popen(
                        cmd,
                        cwd=str(_ROOT),
                        env=os.environ.copy(),
                    ),
                )
            )

        for j, proc in procs:
            rc = proc.wait()
            if rc == 0:
                n_ok += 1
                log.info(f"  ok    {j['instance']}  seed={j['seed']}")
            else:
                n_fail += 1
                log.error(f"  FAIL  {j['instance']}  seed={j['seed']}  rc={rc}")
                fail_row = {
                    "campaign": "C3",
                    "instance": j["instance"],
                    "seed": j["seed"],
                    "cpus": j["cpus"],
                    "status": "error",
                    "error": f"subprocess exit {rc}",
                }
                with campaign_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(fail_row) + "\n")
        log.info(f"═══ Wave {wave_idx}/{n_waves} complete ═══")

    elapsed = time.perf_counter() - campaign_t0
    log.info(
        f"C3 finished: {n_ok} ok / {n_fail} failed in {_format_wall_time(elapsed)}  "
        f"summary={campaign_jsonl}"
    )
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
