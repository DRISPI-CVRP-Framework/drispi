#!/usr/bin/env python3
"""Generate a matplotlib ECDF of the per-instance gap-to-BKS distribution for
six of the nine solvers in Section 5.2.1's summary table: AILS-II, FILO2,
DRISPI, FILO, SISRs, and HGS-CVRP. KGLSXXL, LKH-3, and OR-Tools are omitted,
same rationale as the paired-test exclusion in Section 5.2.1: their gaps are
large enough that including them would compress the visible range for the
other six.

Supersedes generate_gap_ecdf_tikz.py: the pgfplots-rendered version was hard
to read (thin default lines, an overlapping legend) so this now produces a
vector PDF directly with matplotlib and is placed with \\includegraphics
instead of \\input into a pgfplots axis.

Uses the mean-of-N metric for all six curves; Section 5.2.1 establishes
mean-of-N as the primary comparison, since DRISPI's mean-of-3 and each
baseline's mean-of-60 estimate the same population quantity (unlike
best-of-N, an order statistic over very different sample sizes).

Reads data/results/xl_solver_comparison.json.
Writes thesis/figures/gap_ecdf.pdf.

Usage:
    python scripts/generate_gap_ecdf_plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.thesis_figstyle import apply, fig_size  # noqa: E402

import matplotlib.pyplot as plt

DATA_FILE = ROOT / "data/results/xl_solver_comparison_lagrange.json"
OUTPUT = ROOT / "thesis/figures/gap_ecdf.pdf"

# Ordered by mean-of-N gap, ascending (Table 5.2), except DRISPI is kept last
# regardless of rank since it is the thesis's own method and is drawn on top
# deliberately. Draw order matters: earlier series are drawn first, so later
# (higher-gap) curves stay on top.
PLOTS = [
    ("AILS-II", "#1f77b4", "--", 1.2),
    ("FILO2", "#ff7f0e", "--", 1.2),
    ("FILO", "#2ca02c", "--", 1.2),
    ("SISRs", "#9467bd", "--", 1.2),
    ("HGS-CVRP", "#8c564b", ":", 1.4),
    ("DRISPI", "#d62728", "-", 2.0),
]
X_CLIP = 1.0  # percent; a handful of tail instances exceed this for every solver


def _gap_pct(cost: float, bks: float) -> float:
    return (cost - bks) / bks * 100.0


def _series(data: dict, solver: str) -> list[float]:
    vals = [
        _gap_pct(entry["solvers"][solver]["mean"], entry["bks"])
        for entry in data.values()
        if "mean" in entry["solvers"].get(solver, {})
    ]
    return sorted(vals)


def main() -> None:
    apply()
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=fig_size(0.85, 270))
    for solver, color, linestyle, linewidth in PLOTS:
        vals = _series(data, solver)
        n = len(vals)
        y = [(i + 1) / n for i in range(n)]
        # Right-continuous step ECDF, clipped at X_CLIP for readability; a
        # handful of tail instances exceed it for every solver (Section 5.2.1).
        ax.step(
            vals,
            y,
            where="post",
            label=solver,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    ax.set_xlim(0, X_CLIP)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean-of-$N$ gap to BKS (%)")
    ax.set_ylabel("Fraction of instances")
    ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
