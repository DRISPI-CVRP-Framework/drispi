#!/usr/bin/env python3
"""One-shot BG-AILS ablation: checkpoints → performance → measurement, both waves.

Wave 1 (seeds 101–103) finishes before wave 2 (104–106), so an overnight stop
still leaves a complete 100×3 design. Completed cells are skipped on resume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from drispi.ablation import config as ac  # noqa: E402
from drispi.ablation.stats import analyse_performance  # noqa: E402


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} | {msg}", flush=True)


def _run(argv: list[str], *, retries: int = 0) -> None:
    _log("+ " + " ".join(argv))
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            subprocess.check_call(argv)
            return
        except subprocess.CalledProcessError as exc:
            last = exc
            if attempt < retries:
                _log(f"retry {attempt + 1}/{retries} after exit {exc.returncode}")
    assert last is not None
    raise last


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--waves", type=int, nargs="+", default=[1, 2], choices=(1, 2))
    p.add_argument("--checkpoint-jobs", type=int, default=ac.CHECKPOINT_JOBS)
    p.add_argument("--jobs", type=int, default=ac.ABLATION_JOBS)
    args = p.parse_args()

    py = sys.executable
    if not ac.STOCK_JAR.is_file() or not ac.TOUCH_JAR.is_file():
        raise SystemExit(
            f"missing jars: stock={ac.STOCK_JAR.is_file()} touch={ac.TOUCH_JAR.is_file()}"
        )

    t0 = time.perf_counter()
    _log(
        f"campaign start waves={args.waves} "
        f"checkpoint_jobs={args.checkpoint_jobs} jvm_jobs={args.jobs}"
    )
    for wave in args.waves:
        _log(f"wave {wave} checkpoints")
        _run(
            [
                py,
                "scripts/generate_bgails_checkpoints.py",
                "--wave",
                str(wave),
                "--jobs",
                str(args.checkpoint_jobs),
            ],
            retries=1,
        )
        _log(f"wave {wave} performance")
        _run(
            [
                py,
                "scripts/run_bgails_ablation.py",
                "--campaign",
                "performance",
                "--wave",
                str(wave),
                "--jobs",
                str(args.jobs),
            ],
            retries=1,
        )
        _log(f"wave {wave} measurement")
        _run(
            [
                py,
                "scripts/run_bgails_ablation.py",
                "--campaign",
                "measurement",
                "--wave",
                str(wave),
                "--jobs",
                str(args.jobs),
            ],
            retries=1,
        )
        _log(f"wave {wave} complete")

    out = ac.ROOT / "artifacts/bg_ails_ablation/performance_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = analyse_performance()
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _log(f"wrote {out} n_instances={stats.get('n_instances')} n_cells={stats.get('n_cells')}")
    elapsed_h = (time.perf_counter() - t0) / 3600.0
    _log(f"campaign done in {elapsed_h:.2f} h")


if __name__ == "__main__":
    main()
