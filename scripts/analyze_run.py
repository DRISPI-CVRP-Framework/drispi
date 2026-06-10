"""Manual deep analysis for a finished DRISPI run.

Usage:
    python scripts/analyze_run.py <run_dir> [--bks <float>]

Reads ``run.jsonl`` from ``<run_dir>`` and generates a ``deep_analysis/``
subfolder with route-pool, subproblem, improvement, and SP/SC solver charts
plus ``deep_report.html``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

from drispi.pipeline.analysis import (
    AX_BG,
    HTML_STYLE,
    SPINE_COLOR,
    TEXT_COLOR,
    filter_by_type,
    get_haos_config_from_jsonl,
    load_jsonl,
    make_figure,
    save_figure,
    style_axes,
    style_legend,
)


def _summaries(lines: list[dict]) -> list[dict]:
    return sorted(filter_by_type(lines, "summary"), key=lambda s: s.get("iteration", 0))


def _no_data(ax: Any, msg: str = "no data") -> None:
    ax.text(
        0.5,
        0.5,
        msg,
        transform=ax.transAxes,
        color=TEXT_COLOR,
        fontfamily="monospace",
        ha="center",
    )


# ---------------------------------------------------------------------------
# Route pool health
# ---------------------------------------------------------------------------


def plot_pool_diversity_quality(lines: list[dict], output_dir: Path) -> None:
    """Diversity/quality lines over iterations + per-iteration tradeoff scatter."""
    summaries = [
        s
        for s in _summaries(lines)
        if s.get("pool_diversity_avg") is not None and s.get("pool_quality_avg") is not None
    ]

    fig = make_figure((10.0, 7.5))
    ax_lines = fig.add_subplot(211)
    ax_scatter = fig.add_subplot(212)
    style_axes(ax_lines)
    style_axes(ax_scatter)

    if summaries:
        iters = [s["iteration"] for s in summaries]
        div = [float(s["pool_diversity_avg"]) for s in summaries]
        qual = [float(s["pool_quality_avg"]) for s in summaries]
        ax_lines.plot(iters, div, color="#4aa3ff", linewidth=1.6, label="diversity")
        ax_lines.plot(iters, qual, color="#2ecc71", linewidth=1.6, label="quality")
        ax_lines.set_ylim(0, 1)
        legend = ax_lines.legend(loc="upper right", fontsize=8)
        style_legend(legend)

        sc = ax_scatter.scatter(div, qual, c=iters, cmap="plasma", s=24)
        cbar = fig.colorbar(sc, ax=ax_scatter)
        cbar.set_label("iteration", color=TEXT_COLOR, fontfamily="monospace")
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.outline.set_edgecolor(SPINE_COLOR)
    else:
        _no_data(ax_lines)
        _no_data(ax_scatter)

    ax_lines.set_title("Pool diversity / quality over iterations", fontsize=10)
    ax_lines.set_xlabel("iteration")
    ax_lines.set_ylabel("score")
    ax_scatter.set_title("Diversity vs quality (colored by iteration)", fontsize=10)
    ax_scatter.set_xlabel("diversity_avg")
    ax_scatter.set_ylabel("quality_avg")
    save_figure(fig, Path(output_dir) / "pool_diversity_quality.png")


def plot_pool_eviction_balance(lines: list[dict], output_dir: Path) -> None:
    """Per-iteration duplicates, cumulative trend, and pool size with evictions."""
    summaries = _summaries(lines)

    fig = make_figure((10.0, 10.0))
    ax_bars = fig.add_subplot(311)
    ax_cum = fig.add_subplot(312)
    ax_pool = fig.add_subplot(313)
    for ax in (ax_bars, ax_cum, ax_pool):
        style_axes(ax)

    if summaries:
        iters = [s.get("iteration", i) for i, s in enumerate(summaries)]
        rejected = [int(s.get("duplicates_rejected", 0) or 0) for s in summaries]
        replaced = [int(s.get("duplicates_replaced", 0) or 0) for s in summaries]

        width = 0.4
        ax_bars.bar(
            [i - width / 2 for i in iters], rejected, width=width,
            color="#e74c3c", label="rejected",
        )
        ax_bars.bar(
            [i + width / 2 for i in iters], replaced, width=width,
            color="#ff9f43", label="replaced",
        )
        legend = ax_bars.legend(loc="upper right", fontsize=8)
        style_legend(legend)

        cum_rej, cum_rep = [], []
        tr = tp = 0
        for r, p in zip(rejected, replaced, strict=True):
            tr += r
            tp += p
            cum_rej.append(tr)
            cum_rep.append(tp)
        ax_cum.plot(iters, cum_rej, color="#e74c3c", linewidth=1.6, label="cumulative rejected")
        ax_cum.plot(iters, cum_rep, color="#ff9f43", linewidth=1.6, label="cumulative replaced")
        legend = ax_cum.legend(loc="upper left", fontsize=8)
        style_legend(legend)

        sizes = [s.get("pool_size") for s in summaries]
        known = [(i, v) for i, v in zip(iters, sizes, strict=True) if v is not None]
        if known:
            xs = [i for i, _ in known]
            ys = [int(v) for _, v in known]
            ax_pool.plot(xs, ys, color="#4aa3ff", linewidth=1.6, label="pool size")
            evict_x = [xs[j] for j in range(1, len(ys)) if ys[j] < ys[j - 1]]
            evict_y = [ys[j] for j in range(1, len(ys)) if ys[j] < ys[j - 1]]
            if evict_x:
                ax_pool.scatter(
                    evict_x, evict_y, color="#e74c3c", s=40, zorder=5,
                    marker="v", label="eviction",
                )
            legend = ax_pool.legend(loc="lower right", fontsize=8)
            style_legend(legend)
    else:
        for ax in (ax_bars, ax_cum, ax_pool):
            _no_data(ax)

    ax_bars.set_title("Duplicates per iteration", fontsize=10)
    ax_bars.set_ylabel("count")
    ax_cum.set_title("Cumulative duplicates", fontsize=10)
    ax_cum.set_ylabel("count")
    ax_pool.set_title("Pool size with eviction events", fontsize=10)
    ax_pool.set_xlabel("iteration")
    ax_pool.set_ylabel("pool size")
    save_figure(fig, Path(output_dir) / "pool_eviction_balance.png")


# ---------------------------------------------------------------------------
# Subproblem structure
# ---------------------------------------------------------------------------


def plot_cluster_size_distribution(lines: list[dict], output_dir: Path) -> None:
    """Histogram of all cluster sizes + box plot grouped by k."""
    phase_lines = [
        line
        for line in filter_by_type(lines, "phase_done")
        if isinstance(line.get("cluster_sizes"), list)
    ]

    fig = make_figure((11.0, 4.5))
    ax_hist = fig.add_subplot(121)
    ax_box = fig.add_subplot(122)
    style_axes(ax_hist)
    style_axes(ax_box)

    all_sizes: list[int] = []
    by_k: dict[int, list[int]] = {}
    for line in phase_lines:
        sizes = [int(s) for s in line["cluster_sizes"]]
        all_sizes.extend(sizes)
        op = line.get("operator_info") or {}
        k = op.get("k")
        if k is None:
            k = len(sizes)
        by_k.setdefault(int(k), []).extend(sizes)

    if all_sizes:
        ax_hist.hist(all_sizes, bins=min(30, max(5, len(set(all_sizes)))), color="#4aa3ff",
                     edgecolor=AX_BG)
        ks = sorted(by_k)
        bp = ax_box.boxplot([by_k[k] for k in ks], patch_artist=True)
        ax_box.set_xticks(range(1, len(ks) + 1))
        ax_box.set_xticklabels([str(k) for k in ks])
        for patch in bp["boxes"]:
            patch.set_facecolor("#2a4a6e")
            patch.set_edgecolor(SPINE_COLOR)
        for part in ("whiskers", "caps", "medians", "fliers"):
            for artist in bp[part]:
                artist.set_color("#4aa3ff")
    else:
        _no_data(ax_hist)
        _no_data(ax_box)

    ax_hist.set_title("Cluster size distribution (all iterations)", fontsize=10)
    ax_hist.set_xlabel("cluster size (customers)")
    ax_hist.set_ylabel("count")
    ax_box.set_title("Cluster sizes by k", fontsize=10)
    ax_box.set_xlabel("k")
    ax_box.set_ylabel("cluster size")
    save_figure(fig, Path(output_dir) / "cluster_size_distribution.png")


def plot_k_selection(lines: list[dict], haos_config: dict, output_dir: Path) -> None:
    """Actual vs expected k selection frequency + k chosen over iterations."""
    haos_config = haos_config or {}
    rolls = filter_by_type(lines, "haos_roll")
    by_iter: dict[int, int] = {}
    for roll in rolls:
        it = roll.get("iteration")
        k = roll.get("k")
        if it is None or k is None:
            continue
        by_iter[int(it)] = int(k)

    fig = make_figure((11.0, 4.5))
    ax_bar = fig.add_subplot(121)
    ax_line = fig.add_subplot(122)
    style_axes(ax_bar, horizontal_grid_only=True)
    style_axes(ax_line)

    if by_iter:
        counts = Counter(by_iter.values())
        candidates = haos_config.get("k_candidates") or []
        k_values = sorted({int(k) for k in candidates} | set(counts))
        total = len(by_iter)
        expected = total / len(k_values) if k_values else 0.0
        ax_bar.bar(
            [str(k) for k in k_values],
            [counts.get(k, 0) for k in k_values],
            color="#4aa3ff",
            label="actual",
        )
        ax_bar.axhline(expected, color="#f5d442", linestyle=":", linewidth=1.4,
                       label="expected (uniform)")
        legend = ax_bar.legend(loc="upper right", fontsize=8)
        style_legend(legend)

        iters = sorted(by_iter)
        ax_line.plot(iters, [by_iter[i] for i in iters], color="#4aa3ff",
                     linewidth=1.2, marker=".", markersize=5)
    else:
        _no_data(ax_bar)
        _no_data(ax_line)

    ax_bar.set_title("k selection frequency", fontsize=10)
    ax_bar.set_xlabel("k")
    ax_bar.set_ylabel("times selected")
    ax_line.set_title("k selected over iterations", fontsize=10)
    ax_line.set_xlabel("iteration")
    ax_line.set_ylabel("k")
    save_figure(fig, Path(output_dir) / "k_selection.png")


# ---------------------------------------------------------------------------
# Improvement analysis
# ---------------------------------------------------------------------------


def plot_bg_ails_improvement(lines: list[dict], output_dir: Path) -> None:
    """Before/after scatter, improvement-% histogram, and improvement-% trend."""
    bg_lines = [
        line
        for line in filter_by_type(lines, "phase_done")
        if line.get("phase_name") == "bg_ails"
        and line.get("bg_cost_before") is not None
        and line.get("bg_cost_after") is not None
    ]
    bg_lines.sort(key=lambda b: b.get("iteration", 0))

    fig = make_figure((10.0, 10.0))
    ax_scatter = fig.add_subplot(311)
    ax_hist = fig.add_subplot(312)
    ax_trend = fig.add_subplot(313)
    for ax in (ax_scatter, ax_hist, ax_trend):
        style_axes(ax)

    if bg_lines:
        before = [float(b["bg_cost_before"]) for b in bg_lines]
        after = [float(b["bg_cost_after"]) for b in bg_lines]
        iters = [b.get("iteration", i) for i, b in enumerate(bg_lines)]
        improvement_pct = [
            ((b - a) / b * 100.0) if b > 0 else 0.0
            for b, a in zip(before, after, strict=True)
        ]

        sc = ax_scatter.scatter(before, after, c=improvement_pct, cmap="viridis", s=26)
        lo = min(min(before), min(after))
        hi = max(max(before), max(after))
        ax_scatter.plot([lo, hi], [lo, hi], color=SPINE_COLOR, linewidth=1.0,
                        linestyle="--")
        cbar = fig.colorbar(sc, ax=ax_scatter)
        cbar.set_label("improvement %", color=TEXT_COLOR, fontfamily="monospace")
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.outline.set_edgecolor(SPINE_COLOR)

        ax_hist.hist(improvement_pct, bins=min(30, max(5, len(improvement_pct))),
                     color="#2ecc71", edgecolor=AX_BG)
        ax_trend.plot(iters, improvement_pct, color="#2ecc71", linewidth=1.4,
                      marker=".", markersize=5)
        ax_trend.axhline(0.0, color=SPINE_COLOR, linewidth=0.8)
    else:
        for ax in (ax_scatter, ax_hist, ax_trend):
            _no_data(ax)

    ax_scatter.set_title("BG-AILS cost before vs after", fontsize=10)
    ax_scatter.set_xlabel("cost before")
    ax_scatter.set_ylabel("cost after")
    ax_hist.set_title("BG-AILS improvement % distribution", fontsize=10)
    ax_hist.set_xlabel("improvement %")
    ax_hist.set_ylabel("count")
    ax_trend.set_title("BG-AILS improvement % over iterations", fontsize=10)
    ax_trend.set_xlabel("iteration")
    ax_trend.set_ylabel("improvement %")
    save_figure(fig, Path(output_dir) / "bg_ails_improvement.png")


def plot_route_age_distribution(lines: list[dict], output_dir: Path) -> None:
    """Age distribution of the final route pool from the final JSONL block."""
    final_lines = filter_by_type(lines, "final")

    fig = make_figure((10.0, 4.5))
    ax = fig.add_subplot(111)
    style_axes(ax, horizontal_grid_only=True)

    ages: list[int] = []
    if final_lines:
        final = final_lines[0]
        tag_iters = final.get("pool_tag_iterations") or []
        last_iteration = max(int(final.get("iterations", 0)) - 1, 0)
        ages = [max(last_iteration - int(t), 0) for t in tag_iters]

    if ages:
        counts = Counter(ages)
        xs = sorted(counts)
        ax.bar([str(x) for x in xs], [counts[x] for x in xs], color="#4aa3ff")
    else:
        _no_data(ax)

    ax.set_title("Route age distribution in final pool", fontsize=10)
    ax.set_xlabel("route age (iterations)")
    ax.set_ylabel("routes")
    save_figure(fig, Path(output_dir) / "route_age_distribution.png")


# ---------------------------------------------------------------------------
# SP/SC solver behavior
# ---------------------------------------------------------------------------


def _sp_sc_lines(lines: list[dict]) -> list[dict]:
    out = [
        line
        for line in filter_by_type(lines, "phase_done")
        if line.get("phase_name") == "sp_sc"
    ]
    out.sort(key=lambda s: s.get("iteration", 0))
    return out


def plot_lp_fractionality(lines: list[dict], output_dir: Path) -> None:
    """LP fractionality per SP/SC solve + value histogram with mean/median."""
    sp_lines = [
        line for line in _sp_sc_lines(lines) if line.get("lp_fractionality") is not None
    ]

    fig = make_figure((11.0, 4.5))
    ax_line = fig.add_subplot(121)
    ax_hist = fig.add_subplot(122)
    style_axes(ax_line)
    style_axes(ax_hist)

    if sp_lines:
        iters = [s.get("iteration", i) for i, s in enumerate(sp_lines)]
        vals = [float(s["lp_fractionality"]) for s in sp_lines]
        ax_line.plot(iters, vals, color="#ff9f43", linewidth=1.5, marker=".",
                     markersize=5)
        ax_line.set_ylim(0, 1)
        ax_hist.hist(vals, bins=min(20, max(5, len(vals))), color="#ff9f43",
                     edgecolor=AX_BG)
        ax_hist.text(
            0.97,
            0.95,
            f"mean   {mean(vals):.3f}\nmedian {median(vals):.3f}",
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            color=TEXT_COLOR,
            fontsize=9,
            fontfamily="monospace",
        )
    else:
        _no_data(ax_line)
        _no_data(ax_hist)

    ax_line.set_title("LP fractionality over SP/SC iterations", fontsize=10)
    ax_line.set_xlabel("iteration")
    ax_line.set_ylabel("lp_fractionality")
    ax_hist.set_title("LP fractionality distribution", fontsize=10)
    ax_hist.set_xlabel("lp_fractionality")
    ax_hist.set_ylabel("count")
    save_figure(fig, Path(output_dir) / "lp_fractionality.png")


def plot_mip_solve_times(lines: list[dict], output_dir: Path) -> None:
    """SP/SC solve time per solve colored by budget load + histogram."""
    sp_lines = [line for line in _sp_sc_lines(lines) if line.get("elapsed") is not None]

    fig = make_figure((11.0, 4.5))
    ax_time = fig.add_subplot(121)
    ax_hist = fig.add_subplot(122)
    style_axes(ax_time)
    style_axes(ax_hist)

    if sp_lines:
        iters = [s.get("iteration", i) for i, s in enumerate(sp_lines)]
        elapsed = [float(s["elapsed"]) for s in sp_lines]
        budgets = [s.get("budget") for s in sp_lines]
        budget = next((float(b) for b in budgets if b is not None), None)

        colors = []
        over_count = 0
        for e, b in zip(elapsed, budgets, strict=True):
            if b is None or float(b) <= 0:
                colors.append("#888888")
                continue
            load = e / float(b)
            if load > 1.0:
                colors.append("#e74c3c")
                over_count += 1
            elif load >= 0.8:
                colors.append("#ff9f43")
            else:
                colors.append("#2ecc71")

        bar_width = max(0.8, (max(iters) - min(iters)) / max(len(iters) * 2, 1))
        ax_time.bar(iters, elapsed, color=colors, width=bar_width)
        if budget is not None:
            ax_time.axhline(budget, color="#f5d442", linestyle=":", linewidth=1.4,
                            label="budget")
            legend = ax_time.legend(loc="upper right", fontsize=8)
            style_legend(legend)

        ax_hist.hist(elapsed, bins=min(20, max(5, len(elapsed))), color="#4aa3ff",
                     edgecolor=AX_BG)
        pct_over = over_count / len(elapsed) * 100.0
        ax_hist.text(
            0.97,
            0.95,
            f"{pct_over:.1f}% hit time limit",
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            color=TEXT_COLOR,
            fontsize=9,
            fontfamily="monospace",
        )
    else:
        _no_data(ax_time)
        _no_data(ax_hist)

    ax_time.set_title("SP/SC solve time per iteration", fontsize=10)
    ax_time.set_xlabel("iteration")
    ax_time.set_ylabel("elapsed (s)")
    ax_hist.set_title("Solve time distribution", fontsize=10)
    ax_hist.set_xlabel("elapsed (s)")
    ax_hist.set_ylabel("count")
    save_figure(fig, Path(output_dir) / "mip_solve_times.png")


def plot_sp_sc_usage(lines: list[dict], output_dir: Path) -> None:
    """SP vs SC vs skipped totals + per-iteration model timeline."""
    sp_lines = _sp_sc_lines(lines)
    summaries = _summaries(lines)
    mode_by_iter: dict[int, str] = {}
    for line in sp_lines:
        op = line.get("operator_info") or {}
        mode = str(op.get("sc_or_sp", "")).upper()
        if mode in ("SP", "SC"):
            mode_by_iter[int(line.get("iteration", -1))] = mode

    all_iters = sorted({int(s.get("iteration", i)) for i, s in enumerate(summaries)})
    if not all_iters and mode_by_iter:
        all_iters = sorted(mode_by_iter)

    fig = make_figure((11.0, 4.5))
    ax_bar = fig.add_subplot(121)
    ax_timeline = fig.add_subplot(122)
    style_axes(ax_bar, horizontal_grid_only=True)
    style_axes(ax_timeline)

    mode_colors = {"SP": "#4aa3ff", "SC": "#ff9f43", "skipped": "#555555"}
    if all_iters:
        sp_count = sum(1 for m in mode_by_iter.values() if m == "SP")
        sc_count = sum(1 for m in mode_by_iter.values() if m == "SC")
        skipped = len(all_iters) - sp_count - sc_count
        ax_bar.bar(
            ["SP", "SC", "skipped"],
            [sp_count, sc_count, skipped],
            color=[mode_colors["SP"], mode_colors["SC"], mode_colors["skipped"]],
        )

        for mode, level in (("SP", 2), ("SC", 1), ("skipped", 0)):
            xs = [
                i
                for i in all_iters
                if mode_by_iter.get(i, "skipped") == mode
            ]
            if xs:
                ax_timeline.scatter(
                    xs, [level] * len(xs), color=mode_colors[mode], s=26,
                    marker="s", label=mode,
                )
        ax_timeline.set_yticks([0, 1, 2])
        ax_timeline.set_yticklabels(["skipped", "SC", "SP"])
        ax_timeline.set_ylim(-0.5, 2.5)
    else:
        _no_data(ax_bar)
        _no_data(ax_timeline)

    ax_bar.set_title("SP vs SC vs skipped iterations", fontsize=10)
    ax_bar.set_ylabel("iterations")
    ax_timeline.set_title("SP/SC model used per iteration", fontsize=10)
    ax_timeline.set_xlabel("iteration")
    save_figure(fig, Path(output_dir) / "sp_sc_usage.png")


# ---------------------------------------------------------------------------
# Report + entry point
# ---------------------------------------------------------------------------

_DEEP_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Route pool health",
        [
            ("pool_diversity_quality.png", "pool diversity / quality"),
            ("pool_eviction_balance.png", "pool eviction balance"),
        ],
    ),
    (
        "Subproblem structure",
        [
            ("cluster_size_distribution.png", "cluster size distribution"),
            ("k_selection.png", "k selection"),
        ],
    ),
    (
        "Improvement analysis",
        [
            ("bg_ails_improvement.png", "BG-AILS improvement"),
        ],
    ),
    (
        "SP/SC solver behavior",
        [
            ("lp_fractionality.png", "LP fractionality"),
            ("mip_solve_times.png", "MIP solve times"),
            ("sp_sc_usage.png", "SP vs SC usage"),
        ],
    ),
    (
        "Route age distribution",
        [
            ("route_age_distribution.png", "route age distribution"),
        ],
    ),
]


def generate_deep_report(run_dir: Path, lines: list[dict]) -> None:
    """Generate deep_analysis/deep_report.html linking all deep-analysis charts."""
    deep_dir = Path(run_dir) / "deep_analysis"
    deep_dir.mkdir(parents=True, exist_ok=True)

    init_lines = filter_by_type(lines, "init")
    instance = init_lines[0].get("instance", "?") if init_lines else "?"

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
<title>DRISPI deep analysis — {instance}</title>
<style>{HTML_STYLE}</style>
</head>
<body>
<h1>DRISPI deep analysis — {instance}</h1>
{sections_html}
</body>
</html>
"""
    (deep_dir / "deep_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep post-run analysis for a DRISPI run.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing run.jsonl")
    parser.add_argument("--bks", type=float, default=None, help="Known BKS cost")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    lines = load_jsonl(run_dir)
    haos_config = get_haos_config_from_jsonl(lines) or {}
    deep_dir = run_dir / "deep_analysis"
    deep_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, Any]] = [
        ("plot_pool_diversity_quality", lambda: plot_pool_diversity_quality(lines, deep_dir)),
        ("plot_pool_eviction_balance", lambda: plot_pool_eviction_balance(lines, deep_dir)),
        ("plot_cluster_size_distribution", lambda: plot_cluster_size_distribution(lines, deep_dir)),
        ("plot_route_age_distribution", lambda: plot_route_age_distribution(lines, deep_dir)),
        ("plot_bg_ails_improvement", lambda: plot_bg_ails_improvement(lines, deep_dir)),
        ("plot_lp_fractionality", lambda: plot_lp_fractionality(lines, deep_dir)),
        ("plot_mip_solve_times", lambda: plot_mip_solve_times(lines, deep_dir)),
        ("plot_k_selection", lambda: plot_k_selection(lines, haos_config, deep_dir)),
        ("plot_sp_sc_usage", lambda: plot_sp_sc_usage(lines, deep_dir)),
        ("generate_deep_report", lambda: generate_deep_report(run_dir, lines)),
    ]
    for name, task in tasks:
        try:
            task()
        except Exception:
            print(f"[deep_analysis] {name} failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"Deep analysis saved to {run_dir}/deep_analysis/")


if __name__ == "__main__":
    main()
