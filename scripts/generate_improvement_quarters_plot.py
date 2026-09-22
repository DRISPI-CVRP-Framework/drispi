#!/usr/bin/env python3
"""Count incumbent updates by wall-clock quarter and plot the totals.

An update is one ``[ IMPROVE ]`` line in a run log: a candidate that
strictly lowered the incumbent, including the first solution of the run.
Elapsed time is the log timestamp minus the first timestamp of that log.
A backward jump of more than 12 hours is a midnight wrap and adds one day.
Updates at or after 7,200 seconds are counted in the fourth quarter.

Reads data/results/lagrange_benchmark/*/seed*/*/run.log.
Writes thesis/figures/improvement_quarters.pdf and
data/results/improvement_quarters_lagrange.json.

Usage:
    python scripts/generate_improvement_quarters_plot.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.thesis_figstyle import apply, fig_size  # noqa: E402

import matplotlib.pyplot as plt

LOG_ROOT = ROOT / "data/results/lagrange_benchmark"
OUTPUT = ROOT / "thesis/figures/improvement_quarters.pdf"
COUNTS_JSON = ROOT / "data/results/improvement_quarters_lagrange.json"

# Half-open bins [0, 1800), [1800, 3600), [3600, 5400), [5400, inf).
EDGES_S = (1800, 3600, 5400)
QUARTER_LABELS = ("0--30", "30--60", "60--90", "90--120")

_TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) \|")
_IMPROVE = re.compile(r"\[ IMPROVE \] ([0-9.]+)  via (\S+)")


def _sec(hours: str, minutes: str, seconds: str) -> int:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def _quarter(elapsed_s: int) -> int:
    for index, edge in enumerate(EDGES_S):
        if elapsed_s < edge:
            return index
    return 3


def count_improvements(log_root: Path) -> dict:
    logs = sorted(log_root.glob("*/seed*/**/run.log"))
    if len(logs) != 300:
        raise SystemExit(f"expected 300 run logs, found {len(logs)} under {log_root}")

    quarters = [0, 0, 0, 0]
    first_solution = [0, 0, 0, 0]
    after_limit = 0
    total = 0

    for log in logs:
        start: int | None = None
        offset = 0
        prev: int | None = None
        seen_improve = False
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            stamp = _TS.match(line)
            if stamp is None:
                continue
            clock = _sec(*stamp.groups())
            if start is None:
                start = clock
                prev = clock
            else:
                assert prev is not None
                if clock < prev - 12 * 3600:
                    offset += 24 * 3600
                prev = clock
            if _IMPROVE.search(line) is None:
                continue
            elapsed = clock + offset - start
            if elapsed < 0:
                raise SystemExit(f"negative elapsed {elapsed}s in {log}")
            quarter = _quarter(elapsed)
            quarters[quarter] += 1
            total += 1
            if not seen_improve:
                first_solution[quarter] += 1
                seen_improve = True
            if elapsed >= 7200:
                after_limit += 1

    if sum(first_solution) != 300:
        raise SystemExit(f"expected one first solution per run, got {sum(first_solution)}")

    return {
        "n_runs": 300,
        "n_improvements": total,
        "quarters": quarters,
        "quarter_labels_min": ["0-30", "30-60", "60-90", "90-120"],
        "first_solution_by_quarter": first_solution,
        "after_7200_s_included_in_fourth": after_limit,
    }


def main() -> None:
    apply()
    counts = count_improvements(LOG_ROOT)
    quarters = counts["quarters"]

    fig, ax = plt.subplots(figsize=fig_size(0.72, 210))
    positions = range(len(quarters))
    bars = ax.bar(
        list(positions),
        quarters,
        color="#7f9fd6",
        edgecolor="black",
        linewidth=0.6,
        width=0.62,
    )
    ax.bar_label(bars, labels=[f"{n:,}" for n in quarters], padding=3, fontsize=9)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(QUARTER_LABELS)
    ax.set_xlabel("Wall-clock time (min)")
    ax.set_ylabel("Improvements")
    ax.set_ylim(0, max(quarters) * 1.22)
    ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    COUNTS_JSON.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {COUNTS_JSON}")
    print(counts)


if __name__ == "__main__":
    # This file used to write the quarter bars alone. The thesis figure now
    # also has the anytime-trajectory panel, produced by the script below.
    from scripts.generate_convergence_plot import main as convergence_main

    convergence_main()
