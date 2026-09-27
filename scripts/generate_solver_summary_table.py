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

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS_FILE = ROOT / "data/results/xl_comparison_stats_lagrange.json"
OUTPUT = ROOT / "thesis/tables/solver_summary_table_lagrange.tex"


def _fmt(value: float, digits: int) -> str:
    return f"+{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=STATS_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args()

    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    table = stats["summary_table"]

    solvers = sorted(table, key=lambda s: table[s]["mean_all"])

    lines: list[str] = []
    for solver in solvers:
        row = table[solver]
        label = r"\gls{drispi}" if solver == "DRISPI" else solver
        cells = [
            label,
            _fmt(row["mean_all"], args.digits),
            _fmt(row["mean_lo"], args.digits),
            _fmt(row["mean_hi"], args.digits),
            _fmt(row["best_all"], args.digits),
            _fmt(row["best_lo"], args.digits),
            _fmt(row["best_hi"], args.digits),
        ]
        # The trailing % eats the newline, so a following \bottomrule is not
        # preceded by a space token (which makes booktabs' \noalign fail).
        lines.append(" & ".join(cells) + r" \\%")

    # \bottomrule lives in this file. A space token after \input makes
    # booktabs' \noalign illegal, and the final % removes the newline before it.
    # No newline after \endinput. A token after the last rule becomes a blank row.
    lines.append(r"\bottomrule\endinput")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} rows to {args.output}")


if __name__ == "__main__":
    main()
