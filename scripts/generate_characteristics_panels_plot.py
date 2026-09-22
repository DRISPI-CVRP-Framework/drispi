#!/usr/bin/env python3
"""Boxplots of the mean-of-3 gap by depot, customer distribution, and demand.

Demand levels are ordered by median. Whiskers are 1.5 IQR. Writes
thesis/figures/characteristics_panels.pdf.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.section_4_2_numbers import load  # noqa: E402
from scripts.thesis_figstyle import apply, fig_size  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = ROOT / "thesis/figures/characteristics_panels.pdf"

PANELS = (
    ("depot", ["R", "C", "E"], {"R": "Random", "C": "Central", "E": "Eccentric"}, "(a) Depot position"),
    (
        "cust_grp",
        ["R", "C", "RC"],
        {"R": "Random", "C": "Clustered", "RC": "Rand.\nclust."},
        "(b) Customer distribution",
    ),
    (
        "demand",
        ["U", "SL", "5-10", "50-100", "1-100", "1-10", "Q"],
        None,
        "(c) Demand distribution",
    ),
)


def _prepared(values: np.ndarray) -> dict:
    q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    low = values[values >= q1 - 1.5 * iqr]
    high = values[values <= q3 + 1.5 * iqr]
    fliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
    return {
        "med": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "whislo": float(low.min()),
        "whishi": float(high.max()),
        "fliers": fliers.tolist(),
    }


def main() -> None:
    apply()
    df = load()
    fig, axes = plt.subplots(1, 3, figsize=fig_size(1.0, 240), sharey=True)
    for ax, (col, order, labels, title) in zip(axes, PANELS):
        stats = []
        for level in order:
            stats.append(_prepared(df.loc[df[col] == level, "g"].to_numpy()))
        ax.bxp(
            stats,
            positions=list(range(1, len(order) + 1)),
            widths=0.55,
            showfliers=True,
            patch_artist=True,
            boxprops={"facecolor": "#7f9fd6", "edgecolor": "black", "linewidth": 0.6},
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "black", "linewidth": 0.6},
            capprops={"color": "black", "linewidth": 0.6},
            flierprops={
                "marker": "o",
                "markersize": 3.5,
                "markerfacecolor": "#a6304c",
                "markeredgecolor": "#a6304c",
                "linestyle": "none",
            },
        )
        ax.set_xticks(list(range(1, len(order) + 1)))
        tick_labels = [labels[level] if labels else level for level in order]
        ax.set_xticklabels(tick_labels)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean-of-3 gap (%)")
    axes[0].set_ylim(bottom=0)
    for label in axes[2].get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
