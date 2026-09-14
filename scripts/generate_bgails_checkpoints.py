#!/usr/bin/env python3
"""Generate BG-AILS ablation checkpoints (HAOS/SC-SP off, pinned config).

Waves: --wave 1 uses seeds 101-103, --wave 2 uses 104-106.
Default --jobs=5 concurrent generators (each generator still uses 6 FILO2 workers).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from drispi.ablation import config as ac  # noqa: E402
from drispi.ablation.checkpoint import generate_checkpoint, list_xl_instances  # noqa: E402


def _job(name: str, seed: int, workers: int, force: bool) -> str:
    path = ac.CHECKPOINT_DIR / name / f"seed{seed}.json"
    if path.is_file() and not force:
        return f"skip existing {path}"
    payload = generate_checkpoint(name, seed, n_workers=workers)
    return (
        f"wrote {payload['_path']} cost={payload['cost']:.1f} "
        f"T={payload['bg_ails_budget_seconds']:.1f}s method={payload['pinned']['method']}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wave", type=int, choices=(1, 2), default=1)
    p.add_argument("--instance", action="append", dest="instances")
    p.add_argument("--seed", type=int, action="append", dest="seeds")
    p.add_argument("--workers", type=int, default=ac.GRANTED_DRI_WORKERS)
    p.add_argument("--jobs", type=int, default=ac.CHECKPOINT_JOBS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    instances = args.instances or list_xl_instances()
    seeds = args.seeds or list(ac.WAVE_SEEDS[args.wave])
    tasks = [(name, seed) for name in instances for seed in seeds]
    jobs = max(1, int(args.jobs))
    print(f"checkpoint tasks={len(tasks)} jobs={jobs}")
    failed = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(_job, name, seed, args.workers, args.force): (name, seed)
            for name, seed in tasks
        }
        for fut in as_completed(futs):
            name, seed = futs[fut]
            try:
                print(fut.result())
            except Exception:
                failed += 1
                print(f"FAILED {name} seed={seed}")
                traceback.print_exc()
    if failed:
        raise SystemExit(f"{failed} checkpoint(s) failed")


if __name__ == "__main__":
    main()
