#!/usr/bin/env python3
"""Generate the compact solver-comparison table for thesis Section 5.2.1.

Reads data/results/xl_comparison_stats.json (built by
scripts/analyze_xl_comparison.py) and writes thesis/tables/solver_summary_table.tex,
a standalone set of tabular rows meant to be \\input from chapters/5_Study.tex
inside a regular (non-long) table environment.

Column order is mean-of-N first, then best-of-N (mean-of-N is the primary
comparison, see Section 5.2.1), each split into All / n<3,400 / n>3,400,
matching the customer-count split Queiroga et al. (2026) use for their own
Table 2. Rows are sorted by ascending mean-of-N gap over all instances.

Usage:
    python scripts/generate_solver_summary_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS_FILE = ROOT / "data/results/xl_comparison_stats.json"
OUTPUT = ROOT / "thesis/tables/solver_summary_table.tex"


def _fmt(value: float) -> str:
    return f"+{value:.2f}"


def main() -> None:
    stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
    table = stats["summary_table"]

    solvers = sorted(table, key=lambda s: table[s]["mean_all"])

    lines: list[str] = []
    for solver in solvers:
        row = table[solver]
        label = solver
        if row["n_mean_all"] < 100:
            label = f"{solver} ($n = {row['n_mean_all']}$)"
        cells = [
            label,
            _fmt(row["mean_all"]),
            _fmt(row["mean_lo"]),
            _fmt(row["mean_hi"]),
            _fmt(row["best_all"]),
            _fmt(row["best_lo"]),
            _fmt(row["best_hi"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
