#!/usr/bin/env python3
"""Generate the four-panel figure of the mean-of-3 gap to BKS by instance
characteristic, for Section 5.2.2 (Performance by Instance Characteristics).

Panels: (a) depot position, (b) customer distribution, (c) demand
distribution -- each a bar of the per-category mean gap with a
one-sample-SD error bar -- and (d) a plain scatter of gap against n, with
no fitted line,
since Section 5.2.2's own regression coefficients are inserted separately
as \\todo{NUMBER: ...} placeholders pending the author's own regression run.

Supersedes generate_characteristics_panels_tikz.py: the pgfplots groupplot
version let panel (c)'s seven rotated tick labels overflow the panel and
collide with the figure caption. This now produces a vector PDF directly
with matplotlib, giving each label enough room to sit inside its own axes.

Reads data/results/xl_solver_comparison.json for n, bks, and DRISPI's
mean-of-3 cost, and thesis/tables/xl_summary_table.tex for the depot,
customer, and demand codes (Table 6.1's own source row order).

Writes thesis/figures/characteristics_panels.pdf.

Usage:
    python scripts/generate_characteristics_panels_plot.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.thesis_figstyle import apply, fig_size  # noqa: E402

import matplotlib.pyplot as plt

DATA_FILE = ROOT / "data/results/xl_solver_comparison_dantzig.json"
SUMMARY_TABLE = ROOT / "thesis/tables/xl_summary_table.tex"
OUTPUT = ROOT / "thesis/figures/characteristics_panels.pdf"


def _gap_pct(cost: float, bks: float) -> float:
    return (cost - bks) / bks * 100.0


def parse_summary_table() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with open(SUMMARY_TABLE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or "&" not in line:
                continue
            line = line.rstrip("\\").strip()
            parts = [p.strip() for p in line.split("&")]
            if len(parts) < 12:
                continue
            _, name, dep, cust, dem, _q, r = parts[:7]
            rows[name] = {"dep": dep, "cust": cust, "dem": dem, "r": r}
    return rows


def base_customer_code(cust: str) -> str:
    # "C (5)" -> "C", "RC (3)" -> "RC", "R" -> "R"
    return cust.split(" ")[0]


def category_stats(
    gaps: list[dict], key: str, order: list[str]
) -> list[tuple[str, float, float, int]]:
    out = []
    for cat in order:
        vals = [g["gap"] for g in gaps if g[key] == cat]
        if not vals:
            continue
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append((cat, mean, sd, len(vals)))
    return out


def plot_bar_panel(ax, stats, label_map, title):
    labels = [label_map.get(cat, cat) if label_map else cat for cat, _, _, _ in stats]
    means = [m for _, m, _, _ in stats]
    sds = [s for _, _, s, _ in stats]
    x = range(len(stats))
    ax.bar(x, means, yerr=sds, capsize=3, color="#7f9fd6", edgecolor="black", linewidth=0.6, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean-of-3 gap (%)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.6)


def main() -> None:
    apply()
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    rows = parse_summary_table()

    gaps = []
    for name, entry in data.items():
        r = rows[name]
        gap = _gap_pct(entry["solvers"]["DRISPI"]["mean"], entry["bks"])
        gaps.append(
            {
                "name": name,
                "n": entry["n"],
                "gap": gap,
                "dep": r["dep"],
                "cust": base_customer_code(r["cust"]),
                "dem": r["dem"],
            }
        )

    dep_order = ["R", "C", "E"]
    cust_order = ["R", "C", "RC"]
    dem_order = ["U", "1-10", "5-10", "1-100", "50-100", "Q", "SL"]

    dep_stats = category_stats(gaps, "dep", dep_order)
    cust_stats = category_stats(gaps, "cust", cust_order)
    dem_stats = category_stats(gaps, "dem", dem_order)

    dep_full = {"R": "Random", "C": "Central", "E": "Eccentric"}
    cust_full = {"R": "Random", "C": "Clustered", "RC": "Random-\nClustered"}

    fig, axes = plt.subplots(2, 2, figsize=fig_size(1.0, 430))

    plot_bar_panel(axes[0][0], dep_stats, dep_full, "(a) Depot position")
    plot_bar_panel(axes[0][1], cust_stats, cust_full, "(b) Customer distribution")
    plot_bar_panel(axes[1][0], dem_stats, None, "(c) Demand distribution")
    for label in axes[1][0].get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    ax_d = axes[1][1]
    ns = [g["n"] for g in gaps]
    gs = [g["gap"] for g in gaps]
    ax_d.scatter(ns, gs, s=12, color="#a6304c", alpha=0.8)
    ax_d.set_xlabel("$n$")
    ax_d.set_ylabel("Mean-of-3 gap (%)")
    ax_d.set_title("(d) Instance size $n$")
    ax_d.set_ylim(bottom=0)
    ax_d.grid(True, linestyle=":", color="gray", alpha=0.6)
    ax_d.xaxis.set_major_formatter(lambda v, pos: f"{int(v):,}")

    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
