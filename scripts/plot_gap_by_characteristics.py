#!/usr/bin/env python3
"""Figure 3-style gap-to-BKS plots by instance characteristic for a benchmark."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from drispi.pipeline.analysis import filter_by_type, load_jsonl
from drispi.pipeline.runner import DEFAULT_BKS_FILE
from scripts.generate_benchmark_table_tex import _parse_report_table

ROOT = Path(__file__).resolve().parent.parent


def _normalize_instance(name: str) -> str:
    if name.startswith("X-n"):
        return "XL" + name[1:]
    return name


def _instance_name(lines: list[dict], run_dir: Path) -> str:
    init_lines = filter_by_type(lines, "init")
    if init_lines and init_lines[0].get("instance"):
        return str(init_lines[0]["instance"])
    stem = run_dir.name
    match = re.match(r"^(.+)_\d{4}_\d{4}$", stem)
    return match.group(1) if match else stem


def _load_drispi_results(benchmark_dir: Path) -> dict[str, float]:
    results: dict[str, float] = {}
    for run_jsonl in sorted(benchmark_dir.glob("*/run.jsonl")):
        lines = load_jsonl(run_jsonl.parent)
        final_lines = filter_by_type(lines, "final")
        if not final_lines:
            continue
        best = final_lines[0].get("best_cost")
        if not isinstance(best, (int, float)) or best <= 0:
            continue
        instance = _normalize_instance(_instance_name(lines, run_jsonl.parent))
        results[instance] = float(best)
    return results


def _instance_size(name: str) -> int:
    match = re.search(r"XL-n(\d+)-k", name)
    if not match:
        raise ValueError(f"Cannot parse instance size from {name}")
    return int(match.group(1))


def _normalize_cust(cust: str) -> str:
    if cust.startswith("RC"):
        return "RC"
    if cust.startswith("C"):
        return "C"
    return "R"


def _gap_to_bks(cost: float, bks: int) -> float:
    return (cost - bks) / bks * 100.0


def _load_rows(benchmark_dir: Path, bks_file: Path) -> list[dict]:
    bks = {k: v for k, v in json.loads(bks_file.read_text()).items() if k.startswith("XL-")}
    meta = _parse_report_table()
    drispi = _load_drispi_results(benchmark_dir)

    rows: list[dict] = []
    for inst, cost in drispi.items():
        if inst not in meta or inst not in bks:
            continue
        rows.append(
            {
                "instance": inst,
                "gap_pct": _gap_to_bks(cost, bks[inst]),
                "n": _instance_size(inst),
                "r": float(meta[inst]["r"]),
                "dep": meta[inst]["dep"],
                "cust": _normalize_cust(meta[inst]["cust"]),
                "dem": meta[inst]["dem"],
            }
        )
    return rows


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ols_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.2)
    return r_value, p_value


def _plot_scatter_regression(
    rows: list[dict],
    x_key: str,
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    x = np.array([row[x_key] for row in rows], dtype=float)
    y = np.array([row["gap_pct"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor("white")
    _style_axis(ax)
    ax.scatter(x, y, color="#4C72B0", alpha=0.85, edgecolors="black", linewidths=0.4, s=42)

    r_value, p_value = _ols_line(ax, x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Gap to BKS (%)")
    ax.set_title(title, fontsize=11)
    ax.text(
        0.03,
        0.97,
        f"$n={len(rows)}$\n$r={r_value:.2f}$, $p={p_value:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_categorical(
    rows: list[dict],
    cat_key: str,
    category_order: list[str],
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    grouped: dict[str, list[float]] = {cat: [] for cat in category_order}
    for row in rows:
        cat = row[cat_key]
        if cat in grouped:
            grouped[cat].append(row["gap_pct"])

    labels = [cat for cat in category_order if grouped[cat]]
    data = [grouped[cat] for cat in labels]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor("white")
    _style_axis(ax)

    positions = np.arange(1, len(labels) + 1)
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "#DDDDDD", "edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    _ = box

    for pos, values in zip(positions, data, strict=True):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            color="#4C72B0",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
            s=36,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Gap to BKS (%)")
    ax.set_title(title, fontsize=11)
    ax.text(0.03, 0.97, f"$n={len(rows)}$", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_plots(benchmark_dir: Path, bks_file: Path = DEFAULT_BKS_FILE) -> None:
    benchmark_dir = Path(benchmark_dir)
    output_dir = benchmark_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(benchmark_dir, bks_file)
    if not rows:
        print("No completed benchmark instances with metadata found.", file=sys.stderr)
        sys.exit(1)

    plots = [
        (
            "gap_vs_instance_size.png",
            lambda: _plot_scatter_regression(
                rows,
                "n",
                "Instance size ($n$)",
                "Gap vs. Instance Size",
                output_dir / "gap_vs_instance_size.png",
            ),
        ),
        (
            "gap_vs_route_length.png",
            lambda: _plot_scatter_regression(
                rows,
                "r",
                "Average route length ($r$)",
                "Gap vs. Average Route Length",
                output_dir / "gap_vs_route_length.png",
            ),
        ),
        (
            "gap_vs_depot_position.png",
            lambda: _plot_categorical(
                rows,
                "dep",
                ["R", "C", "E"],
                "Depot position",
                "Gap vs. Depot Position",
                output_dir / "gap_vs_depot_position.png",
            ),
        ),
        (
            "gap_vs_customer_distribution.png",
            lambda: _plot_categorical(
                rows,
                "cust",
                ["R", "C", "RC"],
                "Customer distribution",
                "Gap vs. Customer Distribution",
                output_dir / "gap_vs_customer_distribution.png",
            ),
        ),
        (
            "gap_vs_demand_distribution.png",
            lambda: _plot_categorical(
                rows,
                "dem",
                ["U", "1-10", "5-10", "1-100", "50-100", "Q", "SL"],
                "Demand distribution",
                "Gap vs. Demand Distribution",
                output_dir / "gap_vs_demand_distribution.png",
            ),
        ),
    ]

    for name, plot_fn in plots:
        plot_fn()
        print(f"Wrote {output_dir / name}")

    summary = {
        "n_instances": len(rows),
        "instance_size": {
            "r": float(stats.pearsonr([r["n"] for r in rows], [r["gap_pct"] for r in rows]).statistic),
            "p": float(stats.pearsonr([r["n"] for r in rows], [r["gap_pct"] for r in rows]).pvalue),
        },
        "route_length": {
            "r": float(stats.pearsonr([r["r"] for r in rows], [r["gap_pct"] for r in rows]).statistic),
            "p": float(stats.pearsonr([r["r"] for r in rows], [r["gap_pct"] for r in rows]).pvalue),
        },
    }
    (output_dir / "gap_by_characteristics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gap-to-BKS by instance characteristics.")
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--bks-file", type=Path, default=DEFAULT_BKS_FILE)
    args = parser.parse_args()
    run_plots(args.benchmark_dir, bks_file=args.bks_file)


if __name__ == "__main__":
    main()
