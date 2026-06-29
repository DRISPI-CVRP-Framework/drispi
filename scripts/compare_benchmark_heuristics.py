"""Compare benchmark_50xl DRISPI results against published heuristic baselines.

Reads ``artifacts/comparison/xl_heuristic_comparison.json`` and completed runs
from a benchmark folder, then writes gap-to-DRISPI charts to
``<benchmark_dir>/comparison/``.

Usage:
    python scripts/compare_benchmark_heuristics.py \\
        artifacts/benchmarks/benchmark_50xl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from drispi.pipeline.analysis import load_jsonl, filter_by_type

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPARISON = ROOT / "artifacts/comparison/xl_heuristic_comparison.json"

HEURISTIC_ORDER = [
    "BKS",
    "AILS-II",
    "FILO",
    "FILO2",
    "KGLSXXL",
    "HGS-CVRP",
    "SISRs",
    "LKH-3",
    "OR-Tools",
    "DRSCI",
]


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


def load_drispi_results(benchmark_dir: Path) -> dict[str, float]:
    """Return final best_cost per instance for runs that completed successfully."""
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


def _gap_pct(cost: float, reference: float) -> float:
    return (cost - reference) / reference * 100.0


def _heuristic_costs(entry: dict, heuristic: str, metric: str) -> float | None:
    if heuristic == "BKS":
        value = entry.get("BKS")
        return float(value) if isinstance(value, (int, float)) else None
    metrics = (entry.get("heuristics") or {}).get(heuristic) or {}
    value = metrics.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def compute_gaps(
    comparison: dict[str, dict],
    drispi: dict[str, float],
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[str]]:
    """Return per-heuristic gap lists for best and avg metrics."""
    instances = sorted(set(comparison) & set(drispi))
    gaps_best: dict[str, list[float]] = {h: [] for h in HEURISTIC_ORDER}
    gaps_avg: dict[str, list[float]] = {h: [] for h in HEURISTIC_ORDER}

    for instance in instances:
        ref = drispi[instance]
        entry = comparison[instance]
        for heuristic in HEURISTIC_ORDER:
            best_cost = _heuristic_costs(entry, heuristic, "best")
            avg_cost = _heuristic_costs(entry, heuristic, "avg")
            if best_cost is not None:
                gaps_best[heuristic].append(_gap_pct(best_cost, ref))
            if avg_cost is not None:
                gaps_avg[heuristic].append(_gap_pct(avg_cost, ref))
    return gaps_best, gaps_avg, instances


def _summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), median(values)


def _plot_gap_comparison(
    gaps_best: dict[str, list[float]],
    gaps_avg: dict[str, list[float]],
    n_instances: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], gaps_best, "(a) Gap vs. Solver Best Result"),
        (axes[1], gaps_avg, "(b) Gap vs. Solver Mean Result"),
    ]

    x = np.arange(len(HEURISTIC_ORDER))
    width = 0.35

    for ax, gaps, title in panels:
        ax.set_facecolor("white")
        means: list[float] = []
        medians: list[float] = []
        for heuristic in HEURISTIC_ORDER:
            m, med = _summarize(gaps[heuristic])
            means.append(m if m is not None else float("nan"))
            medians.append(med if med is not None else float("nan"))

        mean_bars = ax.bar(
            x - width / 2,
            means,
            width,
            label="Mean",
            color="#4C72B0",
            edgecolor="black",
            linewidth=0.6,
        )
        median_bars = ax.bar(
            x + width / 2,
            medians,
            width,
            label="Median",
            color="#55A868",
            edgecolor="black",
            linewidth=0.6,
            hatch="///",
        )

        for bars in (mean_bars, median_bars):
            for bar in bars:
                height = bar.get_height()
                if np.isnan(height):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + (0.02 if height >= 0 else -0.08),
                    f"{height:+.2f}",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=7,
                )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_ylim(top=5.0)
        ax.set_xticks(x)
        ax.set_xticklabels(HEURISTIC_ORDER, rotation=35, ha="right")
        ax.set_ylabel("Gap to DRISPI (%)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(f"Mean and Median Gap to DRISPI Benchmark Results (n={n_instances})", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "gap_vs_drispi.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_summary_json(
    gaps_best: dict[str, list[float]],
    gaps_avg: dict[str, list[float]],
    instances: list[str],
    output_dir: Path,
) -> None:
    summary: dict[str, object] = {
        "n_instances": len(instances),
        "instances": instances,
        "heuristics": {},
    }
    for heuristic in HEURISTIC_ORDER:
        best_mean, best_median = _summarize(gaps_best[heuristic])
        avg_mean, avg_median = _summarize(gaps_avg[heuristic])
        summary["heuristics"][heuristic] = {
            "n_best": len(gaps_best[heuristic]),
            "n_avg": len(gaps_avg[heuristic]),
            "gap_vs_best": {"mean": best_mean, "median": best_median},
            "gap_vs_avg": {"mean": avg_mean, "median": avg_median},
        }
    (output_dir / "gap_vs_drispi.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def run_comparison(
    benchmark_dir: Path,
    comparison_file: Path = DEFAULT_COMPARISON,
) -> None:
    benchmark_dir = Path(benchmark_dir)
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    drispi = load_drispi_results(benchmark_dir)

    gaps_best, gaps_avg, instances = compute_gaps(comparison, drispi)
    if not instances:
        print("No overlapping completed benchmark instances found.", file=sys.stderr)
        sys.exit(1)

    output_dir = benchmark_dir / "comparison"
    _plot_gap_comparison(gaps_best, gaps_avg, len(instances), output_dir)
    _write_summary_json(gaps_best, gaps_avg, instances, output_dir)
    print(f"Compared {len(instances)} instances; output in {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark DRISPI results to published heuristics.")
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument(
        "--comparison-file",
        type=Path,
        default=DEFAULT_COMPARISON,
        help="Heuristic comparison JSON",
    )
    args = parser.parse_args()
    run_comparison(args.benchmark_dir, comparison_file=args.comparison_file)


if __name__ == "__main__":
    main()
