#!/usr/bin/env python3
"""Three-panel scatter of the mean-of-3 gap against standardized n, r, and log qbar.

All panels share the y-axis. Each line is the ordinary least-squares fit on
all 100 instances. qbar is the mean customer demand read from the instance file.

Writes thesis/figures/gap_scale_comparison.pdf.
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
from scipy import stats

OUTPUT = ROOT / "thesis/figures/gap_scale_comparison.pdf"


def _panel(ax, z_values: np.ndarray, gap: np.ndarray, title: str) -> None:
    fit = stats.linregress(z_values, gap)
    ax.scatter(z_values, gap, s=14, color="#a6304c", alpha=0.85, linewidths=0)
    order = np.argsort(z_values)
    ax.plot(z_values[order], fit.intercept + fit.slope * z_values[order], color="black", linewidth=1.1)
    ax.set_title(title)
    ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    ax.text(
        0.04,
        0.96,
        f"{fit.slope:+.3f} pp per SD\n$R^2 = {fit.rvalue**2:.3f}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def main() -> None:
    apply()
    df = load()
    gap = df.g.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=fig_size(1.0, 230), sharey=True, layout="constrained")
    panels = (
        (axes[0], df.zn.to_numpy(), "(a) Customer count $n$"),
        (axes[1], df.zr.to_numpy(), "(b) Average route length $r$"),
        (axes[2], df.zq.to_numpy(), r"(c) $\log \bar{q}$"),
    )
    for ax, z_values, title in panels:
        _panel(ax, z_values, gap, title)
    axes[1].set_xlabel("Standard deviations from the mean")
    axes[0].set_ylabel("Mean-of-3 gap (%)")
    axes[0].set_ylim(0, 1.75)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
