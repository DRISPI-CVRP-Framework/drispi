#!/usr/bin/env python3
"""Capture RoutePool snapshots at wall-clock marks during a SC/SP-free DRI run.

Runs the production DRI loop (Decompose → Route → Improve) with SC/SP and
post-recombination AILS disabled, then pickles the live RoutePool at each
requested mark. Snapshots are intended for a later Gurobi thread-scaling sweep
over the SC/SP MIP, so the pool must match what SC/SP would see mid-campaign
without any prior SC/SP contamination.

No RoutePool persistence API exists in the codebase yet; pickle is the interim
snapshot format for this capture script (and the planned Gurobi sweep).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Cap BLAS / OpenMP threads before importing numpy / spawning workers so
# ProcessPoolExecutor children inherit the limit (same vars as Docker/compose).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drispi.core.types import Route
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.config_io import load_config
from drispi.pipeline.core_manager import CoreManager
from drispi.pipeline.pipeline import DRISPIPipeline, make_run_label
from drispi.pipeline.runner import load_instance_from_vrp_path
from drispi.route_pool.pool import RoutePool
from drispi.utils.time import local_now

# In-memory override: no enable_sp_sc flag exists; huge warmup never schedules SP/SC.
_SP_SC_DISABLED_WARMUP = 10**9
_STAGNATION_DISABLED = 10**9


def _fail_sp_sc(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "SC/SP fired during pool snapshot capture — snapshots would be invalid. "
        "warmup_iterations override failed to keep should_run_sp_sc False."
    )


def _fail_standard_ails(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "Post-recombination AILS fired during pool snapshot capture — "
        "snapshots would be invalid."
    )


def _unique_route_count(pool: RoutePool) -> int:
    """Number of distinct customer-set keys (equals pool.size() for RoutePool)."""
    return len({entry.customer_set for entry in pool})


def _snapshot_filename(instance_stem: str, mark_min: int, seed: int) -> str:
    return f"{instance_stem}_{mark_min}min_seed{seed}.pkl"


class SnapshottingPipeline(DRISPIPipeline):
    """DRISPIPipeline whose run loop dumps the RoutePool at wall-clock marks."""

    def __init__(
        self,
        *args: Any,
        marks_min: list[int],
        snapshot_dir: Path,
        instance_stem: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._marks_min = sorted(marks_min)
        self._snapshot_dir = snapshot_dir
        self._instance_stem = instance_stem
        self._pending_marks = list(self._marks_min)
        self._snapshot_records: list[dict[str, Any]] = []

    @property
    def snapshot_records(self) -> list[dict[str, Any]]:
        return list(self._snapshot_records)

    def run(self) -> list[Route]:
        """Run DRI until all marks are captured, then finalize like production."""
        self._start_time = time.perf_counter()
        self._write_status("running")
        iteration = 0
        try:
            while True:
                elapsed = time.perf_counter() - self._start_time
                if iteration > 0 and not self._pending_marks:
                    self._stopped_by_time = True
                    break
                if iteration > 0 and elapsed >= self._config.time_limit:
                    self._capture_due_snapshots(elapsed, iteration)
                    self._stopped_by_time = True
                    break
                if iteration > 0 and self._no_improve_count >= self._config.max_no_improve:
                    self._stopped_by_no_improve = True
                    break

                self._run_iteration(iteration)
                iteration += 1

                elapsed = time.perf_counter() - self._start_time
                self._capture_due_snapshots(elapsed, iteration)
                if not self._pending_marks:
                    self._stopped_by_time = True
                    break
        except KeyboardInterrupt:
            self._cancelled = True

        self._iterations_completed = iteration
        self._finalize()
        if self._best_solution is None:
            return []
        return self._best_solution

    def _capture_due_snapshots(self, elapsed_s: float, iteration: int) -> None:
        while self._pending_marks and elapsed_s >= self._pending_marks[0] * 60.0:
            mark = self._pending_marks.pop(0)
            self._write_pool_snapshot(mark, elapsed_s, iteration)

    def _write_pool_snapshot(self, mark_min: int, elapsed_s: float, iteration: int) -> None:
        pool = self._pool
        pool_size = pool.size()
        unique_count = _unique_route_count(pool)
        filename = _snapshot_filename(self._instance_stem, mark_min, self._config.seed)
        path = self._snapshot_dir / filename

        # Deep-copy so the on-disk object is a frozen view of this mark.
        snapshot_pool = copy.deepcopy(pool)
        with path.open("wb") as fh:
            pickle.dump(snapshot_pool, fh, protocol=pickle.HIGHEST_PROTOCOL)

        ts = local_now().isoformat(timespec="seconds")
        record = {
            "filename": filename,
            "mark_min": mark_min,
            "wall_clock_s": round(elapsed_s, 3),
            "pool_size": pool_size,
            "unique_route_count": unique_count,
            "timestamp": ts,
            "iteration": iteration,
        }
        self._snapshot_records.append(record)

        content = (
            f"mark={mark_min}min  wall={elapsed_s:.1f}s  "
            f"pool={pool_size}  unique={unique_count}  file={filename}"
        )
        self._logger._emit(  # noqa: SLF001 — reuse pipeline log channels
            iteration - 1 if iteration > 0 else None,
            "SNAPSHOT",
            content,
            json_event={
                "type": "pool_snapshot",
                "mark_min": mark_min,
                "wall_clock_s": elapsed_s,
                "pool_size": pool_size,
                "unique_route_count": unique_count,
                "filename": filename,
                "timestamp": ts,
                "iteration": iteration,
            },
        )


def _build_config(
    config_path: Path,
    *,
    seed: int,
    cores: int | None,
    last_mark_min: int,
    output_dir: Path,
) -> tuple[DRISPIConfig, int]:
    config = load_config(config_path)
    resolved_cores = cores if cores is not None else config.n_workers
    if resolved_cores < 1:
        raise ValueError(f"--cores must be >= 1, got {resolved_cores}")

    # Disable SC/SP (and thus post-SP AILS + deferred HAOS) without schema changes.
    config = replace(
        config,
        warmup_iterations=_SP_SC_DISABLED_WARMUP,
        seed=seed,
        n_workers=resolved_cores,
        # Stop promptly after the final mark; small slack for the last iteration.
        time_limit=float(last_mark_min * 60 + 1),
        # Avoid ending early on stagnation before all marks are hit.
        max_no_improve=_STAGNATION_DISABLED,
        output_dir=output_dir,
        run_analysis=False,
    )
    return config, resolved_cores


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SC/SP-free DRI and pickle RoutePool snapshots at wall-clock marks."
        ),
    )
    parser.add_argument(
        "--instance",
        type=Path,
        required=True,
        help="Path to the .vrp instance file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write snapshots and snapshot_manifest.json into",
    )
    parser.add_argument(
        "--marks",
        type=int,
        nargs="+",
        required=True,
        metavar="MIN",
        help="Wall-clock marks in minutes, e.g. --marks 10 60 90",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the DRI run (default: 42)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the existing YAML config (single source of truth)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Physical cores for subcluster solving (default: n_workers from config)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if any(m <= 0 for m in args.marks):
        print("All --marks must be positive integers (minutes).", file=sys.stderr)
        sys.exit(1)

    marks = sorted(set(args.marks))

    instance_path = args.instance.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    last_mark = marks[-1]
    config, cores = _build_config(
        config_path,
        seed=args.seed,
        cores=args.cores,
        last_mark_min=last_mark,
        output_dir=output_dir,
    )

    instance = load_instance_from_vrp_path(instance_path)
    instance_stem = instance_path.stem
    run_label = make_run_label(instance.name)
    core_manager = CoreManager(cores, cores)

    pipeline = SnapshottingPipeline(
        instance,
        config,
        run_label=run_label,
        core_manager=core_manager,
        instance_id=instance.name,
        marks_min=marks,
        snapshot_dir=output_dir,
        instance_stem=instance_stem,
    )

    with (
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fail_sp_sc),
        patch(
            "drispi.pipeline.pipeline.run_standard_improvement",
            side_effect=_fail_standard_ails,
        ),
    ):
        pipeline.run()

    manifest = {
        "instance": instance.name,
        "instance_stem": instance_stem,
        "instance_path": str(instance_path),
        "seed": args.seed,
        "config_path": str(config_path),
        "cores": cores,
        "marks_min": marks,
        "created_at": local_now().isoformat(timespec="seconds"),
        "run_dir": str(pipeline.run_dir),
        "snapshots": [
            {
                "filename": rec["filename"],
                "mark_min": rec["mark_min"],
                "wall_clock_s": rec["wall_clock_s"],
                "pool_size": rec["pool_size"],
                "unique_route_count": rec["unique_route_count"],
            }
            for rec in pipeline.snapshot_records
        ],
    }
    manifest_path = output_dir / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    captured = len(pipeline.snapshot_records)
    expected = len(marks)
    if captured < expected:
        print(
            f"Warning: captured {captured}/{expected} snapshots "
            f"(run stopped early: {pipeline.stop_reason})",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Wrote {captured} snapshot(s) and {manifest_path.name} → {output_dir}")


if __name__ == "__main__":
    main()
