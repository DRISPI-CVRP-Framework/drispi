#!/usr/bin/env python3
"""Screen B — BG-AILS initial_omega ∈ {1, 10, 30} on 32 cores.

Greedy + k frozen (Screen A winner). Three omega variants, SP/SC off,
async 6+1+1. Jobs fill every 8-core slice (4 concurrent); pairing is by
``(instance, seed)`` in the summary, not contemporaneous waves.

Default: 3 instances × 4 seeds × 3 variants = 36 runs, 2 h each → 9 waves
≈ 18 h wall on 32 cores.

Usage (orchestrator)::

    python scripts/run_bg_ails_screen_b.py --dry-run
    python scripts/run_bg_ails_screen_b.py --cpu-base 0 --total-cores 32

Usage (single slice)::

    python scripts/run_bg_ails_screen_b.py --single \\
        --instance data/instances/xl/XL-n2307-k34.vrp \\
        --seed 42 --variant omega_1 --cpus 0-7
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
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from drispi.core.types import Route
    from drispi.pipeline.config import DRISPIConfig

_STAGNATION_DISABLED = 10**9
_PAIR_SELECTION = "greedy"
_N_CHAINS_MODE = "k"

_DEFAULT_INSTANCES = [
    _ROOT / "data/instances/xl/XL-n2307-k34.vrp",
    _ROOT / "data/instances/xl/XL-n3975-k687.vrp",
    _ROOT / "data/instances/xl/XL-n8389-k2028.vrp",
]
_DEFAULT_SEEDS = [42, 43, 44, 45]
_DEFAULT_MARKS_MIN = [30, 60, 90, 120]
_DEFAULT_CONFIG = _ROOT / "configs/bg_ails_screen_b.yaml"

# Three omega cells packed densely onto 4-slice waves.
VARIANT_ORDER = ("omega_1", "omega_10", "omega_30")
VARIANTS: dict[str, dict[str, float]] = {
    "omega_1": {"bg_ails_initial_omega": 1.0},
    "omega_10": {"bg_ails_initial_omega": 10.0},
    "omega_30": {"bg_ails_initial_omega": 30.0},
}


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("bg_ails_screen_b")
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
    """Pipeline that emits wall-clock trajectory marks to run.jsonl."""

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
        self._logger._emit(  # noqa: SLF001
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
    variant_fields: dict[str, float],
) -> DRISPIConfig:
    return replace(
        base,
        seed=int(seed),
        output_dir=output_dir,
        time_limit=float(duration_hours) * 3600.0,
        max_no_improve=_STAGNATION_DISABLED,
        bg_ails_pair_selection=_PAIR_SELECTION,
        bg_ails_n_chains_mode=_N_CHAINS_MODE,
        bg_ails_initial_omega=float(variant_fields["bg_ails_initial_omega"]),
    )


def _append_campaign_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except ImportError:
            fh.write(line)


def run_single(
    *,
    instance_path: Path,
    seed: int,
    variant: str,
    cpus: str,
    config_path: Path,
    marks_min: list[int],
    duration_hours: float,
    output_dir: Path,
    bks_file: Path | None,
    campaign_jsonl: Path | None = None,
) -> dict[str, Any]:
    log = _setup_logger()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANT_ORDER}")
    variant_fields = VARIANTS[variant]
    base = load_config(config_path)
    instance = load_instance_from_vrp_path(instance_path)
    run_output_dir = output_dir / instance.name / f"seed{seed}" / variant
    run_output_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(
        base,
        seed=seed,
        output_dir=run_output_dir,
        duration_hours=duration_hours,
        variant_fields=variant_fields,
    )
    bks_cost = None
    if bks_file is not None:
        bks_cost = resolve_bks_cost(instance.name, bks_file=bks_file)

    job_id = f"{instance.name}_{variant}_s{seed}"
    run_label = make_run_label(job_id)
    log.info(
        f"START  {instance.name}  seed={seed}  variant={variant}  cpus={cpus}  "
        f"limit={duration_hours:g}h  pair={_PAIR_SELECTION}  "
        f"chains={_N_CHAINS_MODE}  omega={variant_fields['bg_ails_initial_omega']:g}"
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
    gap = None
    if bks_cost is not None and math.isfinite(final_cost):
        gap = gap_to_bks(final_cost, instance.name, {instance.name: bks_cost})

    row: dict[str, Any] = {
        "campaign": "screen_b",
        "instance": instance.name,
        "instance_path": str(instance_path),
        "seed": seed,
        "variant": variant,
        "bg_ails_pair_selection": _PAIR_SELECTION,
        "bg_ails_n_chains_mode": _N_CHAINS_MODE,
        "bg_ails_initial_omega": float(variant_fields["bg_ails_initial_omega"]),
        "cpus": cpus,
        "duration_hours": duration_hours,
        "marks_min": marks_min,
        "wall_marks": pipeline.mark_records,
        "final_best_cost": final_cost if math.isfinite(final_cost) else None,
        "gap_to_bks": gap,
        "wall_clock_s": round(wall_s, 3),
        "iterations": pipeline.iterations_completed,
        "stop_reason": pipeline.stop_reason,
        "run_dir": str(pipeline.run_dir),
        "status": "ok",
    }

    if campaign_jsonl is not None:
        _append_campaign_jsonl(campaign_jsonl, row)

    gap_s = f"{gap:+.3f}%" if gap is not None else "n/a"
    log.info(
        f"DONE   {instance.name}  seed={seed}  variant={variant}  "
        f"best={final_cost:.2f}  gap={gap_s}  wall={_format_wall_time(wall_s)}  "
        f"dir={pipeline.run_dir}"
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
    """Fill every slice: flatten (instance, seed, omega) and chunk into waves."""
    if slices_per_wave < 1:
        raise ValueError(f"slices_per_wave must be >= 1, got {slices_per_wave}")
    flat: list[tuple[Path, int, str]] = []
    for inst in instances:
        for seed in seeds:
            for variant in VARIANT_ORDER:
                flat.append((inst, int(seed), variant))
    jobs: list[dict[str, Any]] = []
    for i, (inst, seed, variant) in enumerate(flat):
        wave_idx = i // slices_per_wave + 1
        slot = i % slices_per_wave
        cpus = _slice_cpus(cpu_base, slot, cores_per_slice)
        jobs.append(
            {
                "wave": wave_idx,
                "slot": slot,
                "instance_path": str(inst),
                "instance": inst.stem,
                "seed": seed,
                "variant": variant,
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


def _gaps_for_variant(
    variant: str,
    complete: list[int],
    by_seed: dict[int, dict[str, dict[str, Any]]],
) -> list[float]:
    out: list[float] = []
    for s in complete:
        g = by_seed[s][variant].get("gap_to_bks")
        if g is not None and math.isfinite(float(g)):
            out.append(float(g))
    return out


def _print_summary(rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("gap_to_bks") is not None]
    if not ok:
        print("No completed rows with gap_to_bks to summarize.")
        return

    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        by_instance[str(row["instance"])].append(row)

    print()
    print("=" * 78)
    print("Screen B summary (gap_to_bks %, lower is better)")
    print("=" * 78)
    print("Frozen: pair_selection=greedy  n_chains_mode=k")
    print("Hypothesis: omega=1 wins (repair, not re-smash)")

    paired_10: list[float] = []
    paired_30: list[float] = []

    for instance, group in sorted(by_instance.items()):
        by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            by_seed[int(row["seed"])][str(row["variant"])] = row
        complete = [
            s for s, variants in by_seed.items() if all(v in variants for v in VARIANT_ORDER)
        ]
        complete.sort()

        print()
        print(f"Instance: {instance}  (complete seeds={len(complete)})")
        print(
            f"  {'omega_1':>16}  {'omega_10':>16}  {'omega_30':>16}"
        )
        g1 = _gaps_for_variant("omega_1", complete, by_seed)
        g10 = _gaps_for_variant("omega_10", complete, by_seed)
        g30 = _gaps_for_variant("omega_30", complete, by_seed)
        print(
            f"  {_fmt_mean_std(g1):>16}  {_fmt_mean_std(g10):>16}  "
            f"{_fmt_mean_std(g30):>16}"
        )
        print(
            f"  {'seed':>6}  {'omega_1':>10}  {'omega_10':>10}  {'omega_30':>10}"
        )
        for s in complete:
            cells = []
            for v in VARIANT_ORDER:
                g = by_seed[s][v].get("gap_to_bks")
                cells.append("n/a" if g is None else f"{float(g):+.3f}")
            print(f"  {s:6d}  {cells[0]:>10}  {cells[1]:>10}  {cells[2]:>10}")
            base = float(by_seed[s]["omega_1"]["gap_to_bks"])
            paired_10.append(float(by_seed[s]["omega_10"]["gap_to_bks"]) - base)
            paired_30.append(float(by_seed[s]["omega_30"]["gap_to_bks"]) - base)

    print()
    print("Paired deltas vs omega=1 over complete blocks (negative = that omega is better):")
    print(f"  omega_10 − omega_1 : {_fmt_mean_std(paired_10)}")
    print(f"  omega_30 − omega_1 : {_fmt_mean_std(paired_30)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Screen B: BG-AILS initial_omega ∈ {1, 10, 30}, greedy+k frozen, "
            "6+1+1×4 packed slices, 2 h, SP/SC off."
        )
    )
    p.add_argument(
        "--single",
        action="store_true",
        help="Run one (instance, seed, variant, cpus) slice and exit",
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
        help="VRP instances (default: n2307 / n3975 / n8389)",
    )
    p.add_argument("--seed", type=int, default=None, help="Seed for --single mode")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=_DEFAULT_SEEDS,
        help="Seeds (default: 42 43 44 45)",
    )
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=list(VARIANT_ORDER),
        help="Variant for --single mode",
    )
    p.add_argument("--cpus", type=str, default=None, help='CPU list for --single, e.g. "0-7"')
    p.add_argument(
        "--cpu-base",
        type=int,
        default=0,
        help="First CPU id of the 32-core island (default: 0 → 0-31)",
    )
    p.add_argument("--total-cores", type=int, default=32)
    p.add_argument("--cores-per-slice", type=int, default=8)
    p.add_argument(
        "--duration-hours",
        type=float,
        default=2.0,
        help="Wall-clock limit per run in hours (default: 2)",
    )
    p.add_argument(
        "--marks",
        type=int,
        nargs="+",
        default=_DEFAULT_MARKS_MIN,
        help="Wall-clock marks in minutes (default: 30 60 90 120)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Config YAML (default: {_DEFAULT_CONFIG.name})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "artifacts/bg_ails_screen_b/runs",
    )
    p.add_argument(
        "--campaign-jsonl",
        type=Path,
        default=_ROOT / "artifacts/bg_ails_screen_b/campaign.jsonl",
    )
    p.add_argument("--bks-file", type=Path, default=DEFAULT_BKS_FILE)
    p.add_argument("--no-bks", action="store_true")
    p.add_argument("--dry-run", action="store_true")
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
        if args.instance is None or args.seed is None or args.cpus is None or args.variant is None:
            log.error("--single requires --instance, --seed, --variant, and --cpus")
            sys.exit(1)
        parse_cpu_list(args.cpus)
        run_single(
            instance_path=args.instance.expanduser().resolve(),
            seed=int(args.seed),
            variant=str(args.variant),
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
    if len(instances) < 1 or len(seeds) < 1:
        log.error("Need at least one instance and one seed")
        sys.exit(1)

    jobs = _build_jobs(
        instances,
        seeds,
        cpu_base=args.cpu_base,
        cores_per_slice=args.cores_per_slice,
        slices_per_wave=slices_per_wave,
    )
    n_waves = math.ceil(len(jobs) / slices_per_wave) if jobs else 0
    est_s = n_waves * float(args.duration_hours) * 3600.0

    log.info(
        f"Screen B: {len(instances)} instance(s) × {len(seeds)} seed(s) × "
        f"{len(VARIANT_ORDER)} variants = {len(jobs)} runs, "
        f"{args.duration_hours:g}h each"
    )
    log.info(
        f"Packing: {slices_per_wave} concurrent 6+1+1 slices "
        f"(cpu_base={args.cpu_base}, {args.total_cores} cores) → "
        f"{n_waves} wave(s), est. wall {_format_wall_time(est_s)}"
    )
    log.info("Frozen: pair_selection=greedy  n_chains_mode=k")
    log.info(f"Marks (min): {marks}")
    log.info(f"Config: {config_path}")
    log.info("Variants: " + ", ".join(VARIANT_ORDER))

    for job in jobs:
        log.info(
            f"  wave {job['wave']}/{n_waves}  slot {job['slot']}  "
            f"{job['instance']:<20}  seed={job['seed']}  "
            f"variant={job['variant']:<11}  cpus={job['cpus']}"
        )

    if args.dry_run:
        log.info("Dry-run only — exiting")
        return

    output_dir = args.output_dir.expanduser().resolve()
    campaign_jsonl = args.campaign_jsonl.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_jsonl.parent.mkdir(parents=True, exist_ok=True)

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
                "--variant",
                j["variant"],
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
            log.info(
                f"  exec  {j['instance']}  seed={j['seed']}  "
                f"variant={j['variant']}  cpus={j['cpus']}"
            )
            procs.append(
                (
                    j,
                    subprocess.Popen(cmd, cwd=str(_ROOT), env=os.environ.copy()),
                )
            )

        for j, proc in procs:
            rc = proc.wait()
            if rc == 0:
                n_ok += 1
                log.info(
                    f"  ok    {j['instance']}  seed={j['seed']}  variant={j['variant']}"
                )
            else:
                n_fail += 1
                log.error(
                    f"  FAIL  {j['instance']}  seed={j['seed']}  "
                    f"variant={j['variant']}  rc={rc}"
                )
                fail_row = {
                    "campaign": "screen_b",
                    "instance": j["instance"],
                    "seed": j["seed"],
                    "variant": j["variant"],
                    "cpus": j["cpus"],
                    "status": "error",
                    "error": f"subprocess exit {rc}",
                }
                _append_campaign_jsonl(campaign_jsonl, fail_row)
        log.info(f"═══ Wave {wave_idx}/{n_waves} complete ═══")

    if campaign_jsonl.is_file():
        loaded: list[dict[str, Any]] = []
        for line in campaign_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                loaded.append(json.loads(line))
        _print_summary(loaded)

    elapsed = time.perf_counter() - campaign_t0
    log.info(
        f"Screen B finished: {n_ok} ok / {n_fail} failed in "
        f"{_format_wall_time(elapsed)}  summary={campaign_jsonl}"
    )
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
