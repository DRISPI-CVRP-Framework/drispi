"""Post-run auto-analysis: HAOS weight history and run-level charts (Phases 9c/9d).

All charts are dark-themed matplotlib PNGs written to ``<run_dir>/analysis/``,
plus a self-contained ``summary_report.html`` linking them.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Dark theme matching the dashboard
FIG_BG = "#060606"
AX_BG = "#0c0c0c"
GRID_COLOR = "#1a1a1a"
TEXT_COLOR = "#e8e8e8"
SPINE_COLOR = "#333333"
DPI = 150

# Improvement-source colors (shared between cost trajectory and sources chart)
SOURCE_COLORS = {
    "bg_ails": "#2ecc71",  # green
    "sp_sc": "#ff9f43",  # orange
    "standard_ails": "#a55eea",  # purple
}

# JSONL phase_name -> display name
PHASE_DISPLAY_NAMES = {
    "decompose": "dissim+cluster",
    "route": "subclusters",
    "bg_ails": "bg_ails",
    "sp_sc": "sp_sc",
    "standard_ails": "standard_ails",
}
PHASE_ORDER = ["decompose", "route", "bg_ails", "sp_sc", "standard_ails"]
PHASE_COLORS = ["#4a90d9", "#2ecc71", "#ff9f43", "#a55eea", "#e74c3c"]

# (levels key, config key, title suffix, output filename)
HAOS_LEVELS: list[tuple[str, str, str, str]] = [
    ("level_1_k", "k_candidates", "HAOS Level 1 — clusters k", "haos_l1_k.png"),
    (
        "level_2_lambda_demand",
        "lambda_demand_values",
        "HAOS Level 2 — lambda demand",
        "haos_l2_lambda.png",
    ),
    ("level_3_paradigm", "paradigm_values", "HAOS Level 3 — paradigm", "haos_l3_paradigm.png"),
    (
        "level_4a_vertex_method",
        "vertex_method_values",
        "HAOS Level 4a — vertex method",
        "haos_l4a_vertex_method.png",
    ),
    (
        "level_4b_route_method",
        "route_method_values",
        "HAOS Level 4b — route method",
        "haos_l4b_route_method.png",
    ),
    ("level_5_solver", "solver_values", "HAOS Level 5 — solver", "haos_l5_solver.png"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(run_dir: Path) -> list[dict]:
    """Read and parse all lines of run.jsonl. Skip malformed lines."""
    path = Path(run_dir) / "run.jsonl"
    lines: list[dict] = []
    if not path.is_file():
        return lines
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                lines.append(obj)
    return lines


def filter_by_type(lines: list[dict], type_: str) -> list[dict]:
    """Filter JSONL lines by type field."""
    return [line for line in lines if line.get("type") == type_]


def get_haos_config_from_jsonl(lines: list[dict]) -> dict | None:
    """Extract HAOSConfig values from the init line."""
    init_lines = filter_by_type(lines, "init")
    if not init_lines:
        return None
    config = init_lines[0].get("config")
    if not isinstance(config, dict):
        return None
    haos_config = config.get("haos_config")
    return haos_config if isinstance(haos_config, dict) else None


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def make_figure(figsize: tuple[float, float] = (10.0, 5.0)) -> Figure:
    """Create a dark-themed figure with constrained layout (avoids panel overlap)."""
    fig = plt.figure(figsize=figsize, facecolor=FIG_BG, dpi=DPI, layout="constrained")
    return fig


def style_axes(ax: Axes, *, horizontal_grid_only: bool = False) -> None:
    """Apply the dark dashboard style to one axes object."""
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.tick_params(colors=SPINE_COLOR, labelcolor=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("monospace")
    ax.xaxis.label.set_fontfamily("monospace")
    ax.yaxis.label.set_fontfamily("monospace")
    ax.title.set_fontfamily("monospace")
    if horizontal_grid_only:
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
        ax.grid(False, axis="x")
    else:
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def style_legend(legend: Any) -> None:
    """Dark-theme an existing legend."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(AX_BG)
    frame.set_edgecolor(SPINE_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
        text.set_fontfamily("monospace")


def save_figure(fig: Figure, path: Path) -> None:
    """Save and close a figure with the dark background."""
    fig.savefig(path, dpi=DPI, facecolor=FIG_BG, bbox_inches="tight")
    plt.close(fig)


def thousands_formatter() -> FuncFormatter:
    return FuncFormatter(lambda x, _pos: f"{x:,.0f}")


# ---------------------------------------------------------------------------
# HAOS weight history (Phase 9d)
# ---------------------------------------------------------------------------


def _level_color_map(config_values: list[Any]) -> dict[str, str]:
    """Stable color per option, indexed by position in the config list."""
    cmap = plt.get_cmap("tab20" if len(config_values) > 10 else "tab10")
    return {
        str(v): matplotlib.colors.to_hex(cmap(i % cmap.N))
        for i, v in enumerate(config_values)
    }


def plot_haos_weights(
    lines: list[dict],
    haos_config: dict,
    output_dir: Path,
) -> None:
    """Generate 6 x 100% stacked area charts, one per HAOS level."""
    haos_config = haos_config or {}
    rolls = [
        line
        for line in filter_by_type(lines, "haos_roll")
        if isinstance(line.get("levels"), dict)
    ]
    rolls.sort(key=lambda r: r.get("iteration", 0))
    warmup = int(haos_config.get("haos_warmup", 0) or 0)

    for level_key, config_key, title, filename in HAOS_LEVELS:
        iterations: list[int] = []
        per_iter_probs: list[dict[str, float]] = []
        observed_values: list[str] = []
        for roll in rolls:
            level = roll["levels"].get(level_key)
            if not isinstance(level, dict):
                continue
            values = [str(v) for v in level.get("values", [])]
            probs = level.get("probabilities", [])
            if not values or len(values) != len(probs):
                continue
            for v in values:
                if v not in observed_values:
                    observed_values.append(v)
            iterations.append(int(roll.get("iteration", len(iterations))))
            per_iter_probs.append(dict(zip(values, probs, strict=True)))

        config_values = [str(v) for v in haos_config.get(config_key, []) or []]
        # Canonical stack order: config order first, then any extras observed.
        ordered = [v for v in config_values if v in observed_values]
        ordered += [v for v in observed_values if v not in ordered]
        color_source = config_values if config_values else ordered
        colors = _level_color_map(color_source)

        fig = make_figure((10.0, 5.0))
        ax = fig.add_subplot(111)
        style_axes(ax, horizontal_grid_only=True)

        if iterations and ordered:
            stack = []
            for value in ordered:
                stack.append([per_iter_probs[i].get(value, 0.0) for i in range(len(iterations))])
            # Normalize each iteration to exactly 100%.
            totals = [sum(col) for col in zip(*stack, strict=True)]
            norm = [
                [
                    (row[i] / totals[i] * 100.0) if totals[i] > 0 else 0.0
                    for i in range(len(iterations))
                ]
                for row in stack
            ]
            ax.stackplot(
                iterations,
                norm,
                labels=ordered,
                colors=[colors.get(v, "#888888") for v in ordered],
                alpha=0.9,
            )
            x_lo = min(iterations)
            x_hi = max(iterations) if max(iterations) > x_lo else x_lo + 1
            ax.set_xlim(x_lo, x_hi)
            if warmup > 0:
                ax.axvspan(min(iterations), warmup, color="white", alpha=0.08)
                ax.text(
                    min(iterations) + (warmup - min(iterations)) / 2.0,
                    95,
                    "warmup",
                    color=TEXT_COLOR,
                    fontsize=8,
                    fontfamily="monospace",
                    ha="center",
                    alpha=0.7,
                )
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

        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("iteration")
        ax.set_ylabel("selection probability (%)")
        if iterations and ordered:
            legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
            style_legend(legend)
        save_figure(fig, Path(output_dir) / filename)


# ---------------------------------------------------------------------------
# Auto-analysis charts (Phase 9c)
# ---------------------------------------------------------------------------


def plot_cost_trajectory(
    lines: list[dict],
    bks: float | None,
    output_dir: Path,
) -> None:
    """Cost trajectory over iterations with improvement-source markers."""
    summaries = sorted(filter_by_type(lines, "summary"), key=lambda s: s.get("iteration", 0))
    improves = filter_by_type(lines, "improve")

    fig = make_figure((11.0, 5.5))
    ax = fig.add_subplot(111)
    style_axes(ax)

    if summaries:
        # 1-based display, matching the dashboard and run.log.
        iters = [int(s.get("iteration", i)) + 1 for i, s in enumerate(summaries)]
        iter_costs = [s.get("iter_cost", float("nan")) for s in summaries]
        best_costs: list[float] = []
        running = float("inf")
        for s in summaries:
            running = min(running, float(s.get("best_cost", running)))
            best_costs.append(running)

        ax.plot(iters, iter_costs, color="#2a4a6e", linewidth=1.0, label="iteration cost")
        ax.plot(iters, best_costs, color="#4aa3ff", linewidth=1.8, label="best cost")

        for imp in improves:
            phase = str(imp.get("phase_name", ""))
            color = SOURCE_COLORS.get(phase)
            if color is None:
                continue
            ax.scatter(
                [int(imp.get("iteration", 0)) + 1],
                [imp.get("cost", float("nan"))],
                color=color,
                s=28,
                zorder=5,
            )

        if bks is not None:
            ax.axhline(bks, color="#f5d442", linestyle=":", linewidth=1.4, label="BKS")
            final_best = best_costs[-1]
            gap = (final_best - bks) / bks * 100.0
            ax.text(
                0.99,
                0.97,
                f"final gap to BKS: {gap:+.2f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=TEXT_COLOR,
                fontsize=9,
                fontfamily="monospace",
            )
        legend = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=8)
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

    ax.yaxis.set_major_formatter(thousands_formatter())
    ax.set_title("Cost trajectory", fontsize=11)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    save_figure(fig, Path(output_dir) / "cost_trajectory.png")


def plot_runtime_split(lines: list[dict], output_dir: Path) -> None:
    """Horizontal stacked bars: cumulative seconds and share per phase."""
    phase_lines = filter_by_type(lines, "phase_done")
    totals: dict[str, float] = {name: 0.0 for name in PHASE_ORDER}
    for line in phase_lines:
        name = str(line.get("phase_name", ""))
        if name in totals:
            totals[name] += float(line.get("elapsed", 0.0) or 0.0)

    total_runtime = sum(totals.values())
    final_lines = filter_by_type(lines, "final")
    n_iterations = final_lines[0].get("iterations") if final_lines else None
    if final_lines and final_lines[0].get("elapsed") is not None:
        total_runtime_display = float(final_lines[0]["elapsed"])
    else:
        total_runtime_display = total_runtime

    fig = make_figure((11.0, 4.0))
    ax_abs = fig.add_subplot(211)
    ax_pct = fig.add_subplot(212)
    for ax in (ax_abs, ax_pct):
        style_axes(ax, horizontal_grid_only=False)
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.8)
        ax.grid(False, axis="y")

    left_abs = 0.0
    left_pct = 0.0
    for name, color in zip(PHASE_ORDER, PHASE_COLORS, strict=True):
        seconds = totals[name]
        ax_abs.barh(["Total time (s)"], [seconds], left=[left_abs], color=color,
                    label=PHASE_DISPLAY_NAMES[name])
        left_abs += seconds
        pct = (seconds / total_runtime * 100.0) if total_runtime > 0 else 0.0
        ax_pct.barh(["Share (%)"], [pct], left=[left_pct], color=color)
        left_pct += pct

    ax_pct.set_xlim(0, 100)
    legend = ax_abs.legend(loc="center left", bbox_to_anchor=(1.02, -0.1), fontsize=8)
    style_legend(legend)

    iter_str = f"{n_iterations}" if n_iterations is not None else "?"
    fig.suptitle(
        "Runtime split by phase",
        color=TEXT_COLOR,
        fontsize=11,
        fontfamily="monospace",
    )
    ax_abs.set_title(
        f"total runtime: {total_runtime_display:,.1f}s  ·  iterations: {iter_str}",
        fontsize=8,
        loc="left",
    )
    save_figure(fig, Path(output_dir) / "runtime_split.png")


def plot_improvement_sources(lines: list[dict], output_dir: Path) -> None:
    """Bar + pie panels of new-best events per improvement source."""
    improves = filter_by_type(lines, "improve")
    sources = ["bg_ails", "sp_sc", "standard_ails"]
    counts = {s: 0 for s in sources}
    for imp in improves:
        phase = str(imp.get("phase_name", ""))
        if phase in counts:
            counts[phase] += 1

    fig = make_figure((11.0, 4.5))
    ax_bar = fig.add_subplot(121)
    ax_pie = fig.add_subplot(122)
    style_axes(ax_bar, horizontal_grid_only=True)
    ax_pie.set_facecolor(AX_BG)

    colors = [SOURCE_COLORS[s] for s in sources]
    ax_bar.bar(sources, [counts[s] for s in sources], color=colors)
    ax_bar.set_title("Improvement events by source", fontsize=10)
    ax_bar.set_ylabel("new-best events")

    total = sum(counts.values())
    if total > 0:
        nonzero = [(s, counts[s]) for s in sources if counts[s] > 0]
        wedge_texts = ax_pie.pie(
            [c for _, c in nonzero],
            labels=[s for s, _ in nonzero],
            colors=[SOURCE_COLORS[s] for s, _ in nonzero],
            autopct="%1.1f%%",
            textprops={"color": TEXT_COLOR, "fontfamily": "monospace", "fontsize": 8},
        )
        del wedge_texts
    else:
        ax_pie.text(
            0.5,
            0.5,
            "no improvements",
            transform=ax_pie.transAxes,
            color=TEXT_COLOR,
            fontfamily="monospace",
            ha="center",
        )
        ax_pie.set_xticks([])
        ax_pie.set_yticks([])
    ax_pie.set_title("Share of improvement events", fontsize=10, color=TEXT_COLOR,
                     fontfamily="monospace")
    save_figure(fig, Path(output_dir) / "improvement_sources.png")


def plot_pool_size_over_time(lines: list[dict], output_dir: Path) -> None:
    """Pool size line + per-iteration duplicate stats bars."""
    summaries = sorted(filter_by_type(lines, "summary"), key=lambda s: s.get("iteration", 0))

    fig = make_figure((11.0, 6.0))
    ax_size = fig.add_subplot(211)
    ax_dup = fig.add_subplot(212, sharex=ax_size)
    style_axes(ax_size)
    style_axes(ax_dup, horizontal_grid_only=True)

    if summaries:
        iters = [s.get("iteration", i) for i, s in enumerate(summaries)]
        sizes = [s.get("pool_size") for s in summaries]
        if any(v is not None for v in sizes):
            ax_size.plot(
                iters,
                [v if v is not None else float("nan") for v in sizes],
                color="#4aa3ff",
                linewidth=1.6,
            )
        rejected = [int(s.get("duplicates_rejected", 0) or 0) for s in summaries]
        replaced = [int(s.get("duplicates_replaced", 0) or 0) for s in summaries]
        ax_dup.bar(iters, rejected, color="#e74c3c", label="rejected")
        ax_dup.bar(iters, replaced, bottom=rejected, color="#ff9f43", label="replaced")
        legend = ax_dup.legend(loc="upper right", fontsize=8)
        style_legend(legend)
    else:
        ax_size.text(
            0.5,
            0.5,
            "no data",
            transform=ax_size.transAxes,
            color=TEXT_COLOR,
            fontfamily="monospace",
            ha="center",
        )

    ax_size.set_title("Pool size over iterations", fontsize=10)
    ax_size.set_ylabel("pool size")
    ax_dup.set_title("Duplicate routes per iteration", fontsize=10)
    ax_dup.set_xlabel("iteration")
    ax_dup.set_ylabel("count")
    save_figure(fig, Path(output_dir) / "pool_size_over_time.png")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_STYLE = """
body {
  background: #060606; color: #e8e8e8; font-family: monospace;
  margin: 0; padding: 24px;
}
h1 { font-size: 20px; border-bottom: 1px solid #333333; padding-bottom: 8px; }
h2 { font-size: 15px; color: #4aa3ff; margin-top: 32px; }
.meta { background: #0c0c0c; border: 1px solid #1a1a1a; padding: 12px 16px;
        border-radius: 6px; line-height: 1.7; }
.meta span.key { color: #888888; }
.row { display: flex; gap: 16px; flex-wrap: wrap; }
.row .col { flex: 1 1 45%; min-width: 320px; }
img { width: 100%; background: #0c0c0c; border: 1px solid #1a1a1a;
      border-radius: 6px; margin-top: 8px; }
"""


def _format_stop_reason(final: dict | None) -> str:
    if final is None:
        return "unknown"
    if final.get("stopped_by_time"):
        return "time limit"
    if final.get("stopped_by_no_improve"):
        return "no-improvement limit"
    return "unknown"


def _meta_html(lines: list[dict], bks: float | None) -> str:
    init_lines = filter_by_type(lines, "init")
    final_lines = filter_by_type(lines, "final")
    init = init_lines[0] if init_lines else {}
    final = final_lines[0] if final_lines else None

    instance = init.get("instance", "?")
    n_customers = init.get("n_customers", "?")
    iterations = final.get("iterations", "?") if final else "?"
    elapsed = final.get("elapsed") if final else None
    best_cost = final.get("best_cost") if final else None
    elapsed_s = f"{elapsed:,.1f}s" if isinstance(elapsed, (int, float)) else "?"
    best_s = f"{best_cost:,.2f}" if isinstance(best_cost, (int, float)) else "?"
    if bks is not None and isinstance(best_cost, (int, float)) and bks > 0:
        gap_s = f"{(best_cost - bks) / bks * 100.0:+.2f}%"
    else:
        gap_s = "N/A"

    rows = [
        ("instance", str(instance)),
        ("n_customers", str(n_customers)),
        ("runtime", elapsed_s),
        ("best cost", best_s),
        ("gap to BKS", gap_s),
        ("iterations", str(iterations)),
        ("stop reason", _format_stop_reason(final)),
    ]
    items = "<br>".join(f'<span class="key">{k}:</span> {v}' for k, v in rows)
    return f'<div class="meta">{items}</div>'


def generate_summary_report(
    run_dir: Path,
    lines: list[dict],
    bks: float | None,
) -> None:
    """Generate analysis/summary_report.html linking all PNG charts."""
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    haos_imgs = "".join(
        f'<div class="col"><img src="{filename}" alt="{title}"></div>'
        for _, _, title, filename in HAOS_LEVELS
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DRISPI run summary</title>
<style>{HTML_STYLE}</style>
</head>
<body>
<h1>DRISPI run summary</h1>
{_meta_html(lines, bks)}
<h2>HAOS weight history</h2>
<div class="row">{haos_imgs}</div>
<h2>Cost trajectory</h2>
<img src="cost_trajectory.png" alt="cost trajectory">
<h2>Runtime &amp; improvement sources</h2>
<div class="row">
  <div class="col"><img src="runtime_split.png" alt="runtime split"></div>
  <div class="col"><img src="improvement_sources.png" alt="improvement sources"></div>
</div>
<h2>Route pool</h2>
<img src="pool_size_over_time.png" alt="pool size over time">
</body>
</html>
"""
    (analysis_dir / "summary_report.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_auto_analysis(
    run_dir: Path,
    lines: list[dict],
    bks: float | None,
) -> None:
    """Run all auto-analysis charts; one chart failure does not abort the rest."""
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    haos_config = get_haos_config_from_jsonl(lines) or {}

    tasks: list[tuple[str, Any]] = [
        ("plot_haos_weights", lambda: plot_haos_weights(lines, haos_config, analysis_dir)),
        ("plot_cost_trajectory", lambda: plot_cost_trajectory(lines, bks, analysis_dir)),
        ("plot_runtime_split", lambda: plot_runtime_split(lines, analysis_dir)),
        ("plot_improvement_sources", lambda: plot_improvement_sources(lines, analysis_dir)),
        ("plot_pool_size_over_time", lambda: plot_pool_size_over_time(lines, analysis_dir)),
        ("generate_summary_report", lambda: generate_summary_report(run_dir, lines, bks)),
    ]
    for name, task in tasks:
        try:
            task()
        except Exception:
            print(f"[analysis] {name} failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
