#!/usr/bin/env python3
"""Per-instance seed standard deviation against standardized n and r.

Both panels share the axes. Writes thesis/figures/seed_spread.pdf.
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

OUTPUT = ROOT / "thesis/figures/seed_spread.pdf"


def main() -> None:
    apply()
    df = load()
    y = df.seed_sd_pp.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=fig_size(0.98, 220), sharex=True, sharey=True)
    panels = (
        (axes[0], df.zn.to_numpy(), "(a) Customer count $n$"),
        (axes[1], df.zr.to_numpy(), "(b) Average route length $r$"),
    )
    for ax, z_values, title in panels:
        fit = stats.linregress(z_values, y)
        ax.scatter(z_values, y, s=14, color="#a6304c", alpha=0.85, linewidths=0)
        order = np.argsort(z_values)
        ax.plot(
            z_values[order],
            fit.intercept + fit.slope * z_values[order],
            color="black",
            linewidth=1.1,
        )
        ax.set_title(title)
        ax.set_xlabel("Standard deviations from the mean")
        ax.grid(True, linestyle=":", color="gray", alpha=0.6)
        ax.text(
            0.04,
            0.96,
            f"{fit.slope:+.3f} pp per SD\n$R^2 = {fit.rvalue**2:.3f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
        )
    axes[0].set_ylabel("Seed standard deviation (pp)")
    axes[0].set_ylim(0, max(y) * 1.12)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
