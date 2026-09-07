#!/usr/bin/env python3
"""Generate a pgfplots ECDF of the per-instance gap-to-BKS distribution for
the five solvers in Section 5.2.1's summary table with a mean-of-N gap below
1%: AILS-II, FILO2, DRISPI, FILO, and SISRs.

Per STYLE.md \u00a78, thesis plots go through pgfplots (already loaded via
settings.tex), not a second, raster-based plotting mechanism -- so this
writes step-plot coordinates rather than a PNG.

Uses the mean-of-N metric for all five curves; Section 5.2.1 establishes
mean-of-N as the primary comparison, since DRISPI's mean-of-3 and each
baseline's mean-of-60 estimate the same population quantity (unlike
best-of-N, an order statistic over very different sample sizes).

Reads data/results/xl_solver_comparison.json.
Writes thesis/figures/gap_ecdf_data.tex, five \\addplot coordinate blocks
meant to be \\input inside a pgfplots axis environment in
chapters/5_Study.tex.

Usage:
    python scripts/generate_gap_ecdf_tikz.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/results/xl_solver_comparison.json"
OUTPUT = ROOT / "thesis/figures/gap_ecdf_data.tex"

# Ordered by mean-of-N gap, ascending (see Table 5.1). Draw order matters:
# earlier series get drawn first, so later (higher-gap) curves stay on top.
PLOTS = [
    ("AILS-II", "blue", "dashed", "thin"),
    ("FILO2", "orange", "dashed", "thin"),
    ("FILO", "green!60!black", "dashed", "thin"),
    ("SISRs", "violet", "dashed", "thin"),
    ("DRISPI", "red", "solid", "thick"),
]
X_CLIP = 1.0  # percent; a handful of tail instances exceed this for every solver


def _gap_pct(cost: float, bks: float) -> float:
    return (cost - bks) / bks * 100.0


def _series(data: dict[str, dict], solver: str) -> list[float]:
    vals = [
        _gap_pct(entry["solvers"][solver]["mean"], entry["bks"])
        for entry in data.values()
        if "mean" in entry["solvers"].get(solver, {})
    ]
    return sorted(vals)


def _coordinates(vals: list[float]) -> str:
    n = len(vals)
    # Right-continuous ECDF: at x = vals[i], F jumps to (i+1)/n. Emit one
    # coordinate per instance; "const plot mark left" (set on the axis)
    # draws the horizontal segment to the *left* of each listed point, i.e.
    # the standard right-continuous step, and clips cleanly at X_CLIP.
    coords = [f"({v:.4f},{(i + 1) / n:.4f})" for i, v in enumerate(vals)]
    return " ".join(coords)


def main() -> None:
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    blocks: list[str] = []
    for solver, color, linestyle, linewidth in PLOTS:
        vals = _series(data, solver)
        coords = _coordinates(vals)
        blocks.append(
            f"\\addplot+[const plot mark left, no markers, {linestyle}, {linewidth}, "
            f"color={color}] coordinates {{{coords}}};\n"
            f"\\addlegendentry{{{solver}}}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {len(blocks)} series to {OUTPUT}")


if __name__ == "__main__":
    main()
