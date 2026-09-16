#!/usr/bin/env python3
"""Run BG-AILS ablation arms on existing checkpoints.

Default --jobs=24 (memory cap: one JVM per cell, -Xmx4g). Each cell runs A/B/C
sequentially so the process count equals --jobs.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from drispi.ablation import config as ac  # noqa: E402
from drispi.ablation.arms import cell_job  # noqa: E402
from drispi.ablation.checkpoint import list_xl_instances  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", choices=("performance", "measurement"), required=True)
    p.add_argument("--wave", type=int, choices=(1, 2), default=1)
    p.add_argument("--instance", action="append", dest="instances")
    p.add_argument("--seed", type=int, action="append", dest="seeds")
    p.add_argument("--arm", action="append", dest="arms", choices=("A", "B", "C", "D", "E"))
    p.add_argument("--time-limit", type=float, default=None, help="override T (smoke tests)")
    p.add_argument("--jobs", type=int, default=ac.ABLATION_JOBS)
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "rerun selected --arm JVM(s). --arm D/E may refresh their inits; "
            "A/B/C inits are never overwritten. E also rewrites the mask file"
        ),
    )
    args = p.parse_args()
    instances = args.instances or list_xl_instances()
    seeds = args.seeds or list(ac.WAVE_SEEDS[args.wave])
    arms = tuple(args.arms) if args.arms else ("A", "B", "C")
    payloads: list[tuple[str, str, float | None, tuple[str, ...], bool]] = []
    for name in instances:
        for seed in seeds:
            ckpt = ac.CHECKPOINT_DIR / name / f"seed{seed}.json"
            if not ckpt.is_file():
                print(f"missing checkpoint {ckpt}")
                continue
            payloads.append(
                (str(ckpt), args.campaign, args.time_limit, arms, bool(args.force))
            )
    jobs = max(1, int(args.jobs))
    print(f"{args.campaign} cells={len(payloads)} jobs={jobs} arms={arms}")
    ctx = multiprocessing.get_context("spawn")
    failed = 0
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futs = {pool.submit(cell_job, payload): payload[0] for payload in payloads}
        for fut in as_completed(futs):
            ckpt = futs[fut]
            try:
                print(f"done {fut.result()}")
            except Exception:
                failed += 1
                print(f"FAILED {ckpt}")
                traceback.print_exc()
    if failed:
        raise SystemExit(f"{failed} cell(s) failed")


if __name__ == "__main__":
    main()
