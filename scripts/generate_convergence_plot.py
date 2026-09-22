#!/usr/bin/env python3
"""Convergence figure: update counts by quarter, and the anytime gap trajectory.

Panel (a) recounts incumbent updates from the run logs. Panel (b) uses the
wall-clock marks in the campaign CSV. Writes thesis/figures/improvement_quarters.pdf.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.section_4_2_numbers import load  # noqa: E402
from scripts.thesis_figstyle import OKABE_ITO, apply, fig_size  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = ROOT / "thesis/figures/improvement_quarters.pdf"
LOG_ROOT = ROOT / "data/results/lagrange_benchmark"
_TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2}) \|")
_IMPROVE = re.compile(r"\[ IMPROVE \]")


def _sec(hours: str, minutes: str, seconds: str) -> int:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def _quarter_counts() -> list[int]:
    logs = sorted(LOG_ROOT.glob("*/seed*/**/run.log"))
    if len(logs) != 300:
        raise SystemExit(f"expected 300 run logs, found {len(logs)}")
    quarters = [0, 0, 0, 0]
    for log in logs:
        start = None
        offset = 0
        prev = None
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            stamp = _TS.match(line)
            if stamp is None:
                continue
            clock = _sec(*stamp.groups())
            if start is None:
                start = clock
                prev = clock
            else:
                if clock < prev - 12 * 3600:
                    offset += 24 * 3600
                prev = clock
            if _IMPROVE.search(line) is None:
                continue
            elapsed = clock + offset - start
            quarter = 0 if elapsed < 1800 else 1 if elapsed < 3600 else 2 if elapsed < 5400 else 3
            quarters[quarter] += 1
    return quarters


def main() -> None:
    apply()
    df = load()
    quarters = _quarter_counts()
    fig, axes = plt.subplots(1, 2, figsize=fig_size(1.0, 230))

    ax = axes[0]
    positions = np.arange(4)
    bars = ax.bar(
        positions,
        quarters,
        color="#7f9fd6",
        edgecolor="black",
        linewidth=0.6,
        width=0.62,
    )
    ax.bar_label(bars, labels=[f"{n:,}" for n in quarters], padding=2, fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(["0--30", "30--60", "60--90", "90--120"])
    ax.set_xlabel("Wall-clock time (min)")
    ax.set_ylabel("Improvements")
    ax.set_ylim(0, max(quarters) * 1.18)
    ax.set_title("(a) Incumbent updates")
    ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    x = np.array([30, 60, 90, 120, 140])
    series = (
        ("All instances", [df[f"gap_{m}"].mean() for m in (30, 60, 90, 120)] + [df.gap_final.mean()], OKABE_ITO[0], "-"),
        ("$r < 50$", [df.loc[~df.long, f"gap_{m}"].mean() for m in (30, 60, 90, 120)] + [df.loc[~df.long, "gap_final"].mean()], OKABE_ITO[5], "-"),
        ("$r \\geq 50$", [df.loc[df.long, f"gap_{m}"].mean() for m in (30, 60, 90, 120)] + [df.loc[df.long, "gap_final"].mean()], OKABE_ITO[6], "-"),
    )
    for label, values, color, style in series:
        ax.plot(x, values, color=color, linestyle=style, marker="o", markersize=3.5, linewidth=1.3, label=label)
    ax.axvline(130, color="0.75", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["30", "60", "90", "120", "final"])
    ax.set_xlabel("Wall-clock time (min)")
    ax.set_ylabel("Mean-of-3 gap (%)")
    ax.set_title("(b) Gap by route band")
    ax.set_ylim(0.15, 0.95)
    ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Wrote {OUTPUT}")
    print("quarters", quarters)


if __name__ == "__main__":
    main()
