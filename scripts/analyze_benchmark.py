"""Aggregate post-run analysis across all runs in a benchmark folder.

Usage:
    python scripts/analyze_benchmark.py <benchmark_dir> [--bks-file <path>]

Discovers every subdirectory containing ``run.jsonl``, merges their telemetry,
and writes a single ``analysis/`` and ``deep_analysis/`` under ``<benchmark_dir>``.
Iteration-indexed series are averaged across runs at each iteration index.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

from drispi.pipeline.analysis import (  # noqa: E402
    HAOS_LEVELS,
    HTML_STYLE,
    TEXT_COLOR,
    filter_by_type,
    get_haos_config_from_jsonl,
    load_jsonl,
    make_figure,
    plot_haos_weights,
    plot_improvement_sources,
    plot_pool_size_over_time,
    plot_runtime_split,
    save_figure,
    style_axes,
    style_legend,
)
from drispi.pipeline.runner import DEFAULT_BKS_FILE  # noqa: E402
from drispi.utils.metrics import resolve_bks_cost  # noqa: E402
from scripts.analyze_run import (  # noqa: E402
    _DEEP_SECTIONS,
    plot_bg_ails_improvement,
    plot_cluster_size_distribution,
    plot_lp_fractionality,
    plot_pool_diversity_quality,
    plot_pool_eviction_balance,
    plot_sp_sc_usage,
)

_SUMMARY_AVG_FIELDS = (
    "pool_diversity_avg",
    "pool_quality_avg",
    "pool_size",
    "duplicates_rejected",
    "duplicates_replaced",
    "iter_cost",
    "best_cost",
)
_SP_SC_OP_AVG_FIELDS = (
    "min_coverage",
    "coverage_min",
    "coverage_median",
    "avg_coverage",
    "coverage_max",
)


def _discover_run_dirs(benchmark_dir: Path) -> list[Path]:
    runs = [
        p.parent
        for p in sorted(benchmark_dir.glob("*/run.jsonl"))
        if p.parent.is_dir()
    ]
    return runs


def _instance_name(lines: list[dict], run_dir: Path) -> str:
    init_lines = filter_by_type(lines, "init")
    if init_lines and init_lines[0].get("instance"):
        return str(init_lines[0]["instance"])
    stem = run_dir.name
    match = re.match(r"^(.+)_\d{4}_\d{4}$", stem)
    return match.group(1) if match else stem


def _resolve_bks(lines: list[dict], instance: str, bks_file: Path | None) -> float | None:
    init_lines = filter_by_type(lines, "init")
    if init_lines:
        bks = init_lines[0].get("bks")
        if isinstance(bks, (int, float)) and bks > 0:
            return float(bks)
    return resolve_bks_cost(instance, bks_file=bks_file)


def _load_runs(
    benchmark_dir: Path,
    bks_file: Path | None,
) -> list[tuple[str, Path, list[dict], float | None]]:
    runs: list[tuple[str, Path, list[dict], float | None]] = []
    for run_dir in _discover_run_dirs(benchmark_dir):
        lines = load_jsonl(run_dir)
        if not lines:
            print(f"[benchmark] skipping empty run: {run_dir.name}", file=sys.stderr)
            continue
        instance = _instance_name(lines, run_dir)
        bks = _resolve_bks(lines, instance, bks_file)
        runs.append((instance, run_dir, lines, bks))
    return runs


def _avg_numeric(values: list[Any]) -> float | int | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    avg = mean(nums)
    if all(isinstance(v, int) for v in values):
        return int(round(avg))
    return avg


def _aggregate_summaries(
    runs: list[tuple[str, Path, list[dict], float | None]],
) -> list[dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for _instance, _run_dir, lines, _bks in runs:
        for summary in filter_by_type(lines, "summary"):
            buckets[int(summary.get("iteration", 0))].append(summary)

    aggregated: list[dict] = []
    for iteration in sorted(buckets):
        group = buckets[iteration]
        row: dict[str, Any] = {"type": "summary", "iteration": iteration}
        for field in _SUMMARY_AVG_FIELDS:
            vals = [line[field] for line in group if line.get(field) is not None]
            avg = _avg_numeric(vals)
            if avg is not None:
                row[field] = avg
        aggregated.append(row)
    return aggregated


def _aggregate_haos_rolls(
    runs: list[tuple[str, Path, list[dict], float | None]],
) -> list[dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for _instance, _run_dir, lines, _bks in runs:
        for roll in filter_by_type(lines, "haos_roll"):
            if isinstance(roll.get("levels"), dict):
                buckets[int(roll.get("iteration", 0))].append(roll)

    aggregated: list[dict] = []
    for iteration in sorted(buckets):
        group = buckets[iteration]
        level_keys: set[str] = set()
        for roll in group:
            level_keys.update(roll["levels"].keys())

        merged_levels: dict[str, dict[str, Any]] = {}
        for level_key in level_keys:
            prob_sums: dict[str, list[float]] = defaultdict(list)
            template_values: list[Any] | None = None
            for roll in group:
                level = roll["levels"].get(level_key)
                if not isinstance(level, dict):
                    continue
                values = level.get("values", [])
                probs = level.get("probabilities", [])
                if not values or len(values) != len(probs):
                    continue
                if template_values is None:
                    template_values = list(values)
                for value, prob in zip(values, probs, strict=True):
                    prob_sums[str(value)].append(float(prob))
            if template_values is None:
                continue
            avg_probs = [
                mean(prob_sums[str(value)]) if prob_sums[str(value)] else 0.0
                for value in template_values
            ]
            total = sum(avg_probs)
            if total > 0:
                avg_probs = [p / total for p in avg_probs]
            merged_levels[level_key] = {
                "values": template_values,
                "probabilities": avg_probs,
            }

        if merged_levels:
            aggregated.append(
                {"type": "haos_roll", "iteration": iteration, "levels": merged_levels}
            )
    return aggregated


def _aggregate_sp_sc_phases(
    runs: list[tuple[str, Path, list[dict], float | None]],
) -> list[dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for _instance, _run_dir, lines, _bks in runs:
        for line in filter_by_type(lines, "phase_done"):
            if line.get("phase_name") == "sp_sc":
                buckets[int(line.get("iteration", 0))].append(line)

    aggregated: list[dict] = []
    for iteration in sorted(buckets):
        group = buckets[iteration]
        row: dict[str, Any] = {
            "type": "phase_done",
            "phase_name": "sp_sc",
            "iteration": iteration,
        }
        lp_vals = [line["lp_fractionality"] for line in group if line.get("lp_fractionality") is not None]
        lp_avg = _avg_numeric(lp_vals)
        if lp_avg is not None:
            row["lp_fractionality"] = float(lp_avg)

        elapsed_vals = [line["elapsed"] for line in group if line.get("elapsed") is not None]
        if elapsed_vals:
            # Sum (not average): runtime_split totals phase_done elapsed across all runs.
            row["elapsed"] = sum(float(v) for v in elapsed_vals)

        op_avg: dict[str, float] = {}
        for field in _SP_SC_OP_AVG_FIELDS:
            vals = [
                (line.get("operator_info") or {}).get(field)
                for line in group
                if isinstance(line.get("operator_info"), dict)
                and (line.get("operator_info") or {}).get(field) is not None
            ]
            avg = _avg_numeric(vals)
            if avg is not None:
                op_avg[field] = float(avg)
        if op_avg:
            row["operator_info"] = op_avg
        aggregated.append(row)
    return aggregated


def _merge_lines(runs: list[tuple[str, Path, list[dict], float | None]]) -> list[dict]:
    """Build a synthetic JSONL stream: avg iteration series + pooled event data."""
    merged: list[dict] = []
    if runs:
        merged.extend(filter_by_type(runs[0][2], "init"))
        merged.extend(filter_by_type(runs[0][2], "config"))

    merged.extend(_aggregate_summaries(runs))
    merged.extend(_aggregate_haos_rolls(runs))
    merged.extend(_aggregate_sp_sc_phases(runs))

    for _instance, _run_dir, lines, _bks in runs:
        for line in lines:
            line_type = line.get("type")
            if line_type in {"summary", "haos_roll", "init", "config"}:
                continue
            if line_type == "phase_done" and line.get("phase_name") == "sp_sc":
                continue
            merged.append(line)
    return merged


def _synthetic_final_for_route_ages(
    runs: list[tuple[str, Path, list[dict], float | None]],
) -> dict:
    all_ages: list[int] = []
    for _instance, _run_dir, lines, _bks in runs:
        final_lines = filter_by_type(lines, "final")
        if not final_lines:
            continue
        final = final_lines[0]
        tag_iters = final.get("pool_tag_iterations") or []
        last_iteration = max(int(final.get("iterations", 0)) - 1, 0)
        all_ages.extend(max(last_iteration - int(t), 0) for t in tag_iters)
    max_age = max(all_ages) if all_ages else 0
    return {
        "type": "final",
        "iterations": max_age + 1,
        "pool_tag_iterations": [max_age - age for age in all_ages],
    }


def _plot_benchmark_cost_trajectory(
    runs: list[tuple[str, Path, list[dict], float | None]],
    output_dir: Path,
) -> None:
    """Mean gap-to-BKS (%) at each iteration, averaged across runs."""
    gaps_by_iter: dict[int, list[float]] = defaultdict(list)

    for _instance, _run_dir, lines, bks in runs:
        if bks is None or bks <= 0:
            continue
        summaries = sorted(filter_by_type(lines, "summary"), key=lambda s: s.get("iteration", 0))
        if not summaries:
            continue
        running_best = float("inf")
        for summary in summaries:
            running_best = min(running_best, float(summary.get("best_cost", running_best)))
            iteration = int(summary.get("iteration", 0))
            gap = (running_best - bks) / bks * 100.0
            gaps_by_iter[iteration].append(gap)

    fig = make_figure((11.0, 5.5))
    ax = fig.add_subplot(111)
    style_axes(ax)

    if gaps_by_iter:
        iterations = sorted(gaps_by_iter)
        display_iters = [i + 1 for i in iterations]
        mean_gaps = [mean(gaps_by_iter[i]) for i in iterations]
        ax.plot(display_iters, mean_gaps, color="#4aa3ff", linewidth=1.8, label="mean gap to BKS")

        counts = [len(gaps_by_iter[i]) for i in iterations]
        ax.text(
            0.99,
            0.97,
            f"runs with BKS: {sum(1 for _, _, _, b in runs if b)}\n"
            f"final gap mean: {mean_gaps[-1]:+.2f}%  "
            f"(n={counts[-1]} at iter {display_iters[-1]})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=TEXT_COLOR,
            fontsize=9,
            fontfamily="monospace",
        )
        legend = ax.legend(loc="upper right", fontsize=8)
        style_legend(legend)
    else:
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            color=TEXT_COLOR,
            fontfamily="monospace",
            ha="center",
        )

    ax.set_title("Gap to BKS across benchmark runs (mean per iteration)", fontsize=11)
    ax.set_xlabel("iteration")
    ax.set_ylabel("gap to BKS (%)")
    save_figure(fig, output_dir / "cost_trajectory.png")


def _plot_benchmark_route_ages(
    runs: list[tuple[str, Path, list[dict], float | None]],
    output_dir: Path,
) -> None:
    final = _synthetic_final_for_route_ages(runs)
    lines = [final]
    from scripts.analyze_run import plot_route_age_distribution

    plot_route_age_distribution(lines, output_dir)


def _benchmark_meta_html(
    benchmark_dir: Path,
    runs: list[tuple[str, Path, list[dict], float | None]],
) -> str:
    n_runs = len(runs)
    gaps: list[float] = []
    runtimes: list[float] = []
    iterations: list[int] = []
    for _instance, _run_dir, lines, bks in runs:
        final_lines = filter_by_type(lines, "final")
        if not final_lines:
            continue
        final = final_lines[0]
        best = final.get("best_cost")
        if isinstance(best, (int, float)) and bks is not None and bks > 0:
            gaps.append((float(best) - bks) / bks * 100.0)
        elapsed = final.get("elapsed")
        if isinstance(elapsed, (int, float)):
            runtimes.append(float(elapsed))
        iters = final.get("iterations")
        if isinstance(iters, int):
            iterations.append(iters)

    gap_med = f"{median(gaps):+.2f}%" if gaps else "N/A"
    gap_mean = f"{mean(gaps):+.2f}%" if gaps else "N/A"
    runtime_med = f"{median(runtimes):,.0f}s" if runtimes else "N/A"
    iter_med = str(int(median(iterations))) if iterations else "N/A"

    rows = [
        ("benchmark", benchmark_dir.name),
        ("runs analyzed", str(n_runs)),
        ("median final gap", gap_med),
        ("mean final gap", gap_mean),
        ("median runtime", runtime_med),
        ("median iterations", iter_med),
    ]
    items = "<br>".join(f'<span class="key">{k}:</span> {v}' for k, v in rows)
    return f'<div class="meta">{items}</div>'


def _generate_benchmark_summary_report(
    benchmark_dir: Path,
    runs: list[tuple[str, Path, list[dict], float | None]],
) -> None:
    analysis_dir = benchmark_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    haos_imgs = "".join(
        f'<div class="col"><img src="{filename}" alt="{title}"></div>'
        for _, _, title, filename in HAOS_LEVELS
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DRISPI benchmark summary — {benchmark_dir.name}</title>
<style>{HTML_STYLE}</style>
</head>
<body>
<h1>DRISPI benchmark summary — {benchmark_dir.name}</h1>
{_benchmark_meta_html(benchmark_dir, runs)}
<h2>HAOS weight history (mean per iteration)</h2>
<div class="row">{haos_imgs}</div>
<h2>Gap to BKS trajectory</h2>
<img src="cost_trajectory.png" alt="cost trajectory">
<h2>Runtime &amp; improvement sources (pooled)</h2>
<div class="row">
  <div class="col"><img src="runtime_split.png" alt="runtime split"></div>
  <div class="col"><img src="improvement_sources.png" alt="improvement sources"></div>
</div>
<h2>Route pool (mean per iteration)</h2>
<img src="pool_size_over_time.png" alt="pool size over time">
</body>
</html>
"""
    (analysis_dir / "summary_report.html").write_text(html, encoding="utf-8")


def _generate_benchmark_deep_report(
    benchmark_dir: Path,
    runs: list[tuple[str, Path, list[dict], float | None]],
) -> None:
    deep_dir = benchmark_dir / "deep_analysis"
    deep_dir.mkdir(parents=True, exist_ok=True)

    sections_html = ""
    for section_title, images in _DEEP_SECTIONS:
        imgs = "".join(
            f'<img src="{filename}" alt="{alt}">' for filename, alt in images
        )
        sections_html += f"<h2>{section_title}</h2>\n{imgs}\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DRISPI deep analysis — {benchmark_dir.name}</title>
<style>{HTML_STYLE}</style>
</head>
<body>
<h1>DRISPI deep analysis — {benchmark_dir.name}</h1>
{_benchmark_meta_html(benchmark_dir, runs)}
{sections_html}
</body>
</html>
"""
    (deep_dir / "deep_report.html").write_text(html, encoding="utf-8")


def run_benchmark_analysis(
    benchmark_dir: Path,
    *,
    bks_file: Path | None = DEFAULT_BKS_FILE,
) -> None:
    benchmark_dir = Path(benchmark_dir)
    runs = _load_runs(benchmark_dir, bks_file)
    if not runs:
        print(f"No runs with run.jsonl found under {benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(runs)} run(s) from {benchmark_dir}")
    merged = _merge_lines(runs)
    haos_config = get_haos_config_from_jsonl(merged) or get_haos_config_from_jsonl(runs[0][2]) or {}

    analysis_dir = benchmark_dir / "analysis"
    deep_dir = benchmark_dir / "deep_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    deep_dir.mkdir(parents=True, exist_ok=True)

    analysis_tasks: list[tuple[str, Any]] = [
        ("plot_haos_weights", lambda: plot_haos_weights(merged, haos_config, analysis_dir)),
        ("plot_benchmark_cost_trajectory", lambda: _plot_benchmark_cost_trajectory(runs, analysis_dir)),
        ("plot_runtime_split", lambda: plot_runtime_split(merged, analysis_dir)),
        ("plot_improvement_sources", lambda: plot_improvement_sources(merged, analysis_dir)),
        ("plot_pool_size_over_time", lambda: plot_pool_size_over_time(merged, analysis_dir)),
        ("generate_benchmark_summary_report", lambda: _generate_benchmark_summary_report(benchmark_dir, runs)),
    ]
    deep_tasks: list[tuple[str, Any]] = [
        ("plot_pool_diversity_quality", lambda: plot_pool_diversity_quality(merged, deep_dir)),
        ("plot_pool_eviction_balance", lambda: plot_pool_eviction_balance(merged, deep_dir)),
        ("plot_cluster_size_distribution", lambda: plot_cluster_size_distribution(merged, deep_dir)),
        ("plot_benchmark_route_ages", lambda: _plot_benchmark_route_ages(runs, deep_dir)),
        ("plot_bg_ails_improvement", lambda: plot_bg_ails_improvement(merged, deep_dir)),
        ("plot_lp_fractionality", lambda: plot_lp_fractionality(merged, deep_dir)),
        ("plot_sp_sc_usage", lambda: plot_sp_sc_usage(merged, deep_dir)),
        ("generate_benchmark_deep_report", lambda: _generate_benchmark_deep_report(benchmark_dir, runs)),
    ]

    for name, task in analysis_tasks + deep_tasks:
        try:
            task()
        except Exception:
            print(f"[benchmark] {name} failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"Benchmark analysis saved to {analysis_dir}/")
    print(f"Benchmark deep analysis saved to {deep_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate post-run analysis for all runs in a benchmark folder."
    )
    parser.add_argument(
        "benchmark_dir",
        type=Path,
        help="Benchmark directory containing per-instance run subfolders",
    )
    parser.add_argument(
        "--bks-file",
        type=Path,
        default=DEFAULT_BKS_FILE,
        help="BKS lookup table for gap metrics",
    )
    args = parser.parse_args()
    run_benchmark_analysis(args.benchmark_dir, bks_file=args.bks_file)


if __name__ == "__main__":
    main()
