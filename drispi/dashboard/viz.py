"""Plotly figure builders for the DRISPI monitor dashboard."""

from __future__ import annotations

import html
import math
from typing import Any

import plotly.graph_objects as go

BG = "#060606"
PANEL_BG = "#0c0c0c"
GRID = "#111111"
WHITE = "#e8e8e8"
YELLOW = "#facc15"
BLUE = "#3b82f6"
GREEN = "#4ade80"
RED = "#ef4444"
ORANGE = "#fb923c"
PURPLE = "#a78bfa"
GREY = "#6b7280"

CLUSTER_COLORS: list[str] = [
    "#378add",
    "#639522",
    "#ef9f27",
    "#d85a30",
    "#7f77dd",
    "#1d9e75",
    "#d4537e",
    "#888780",
    "#e74c3c",
    "#3498db",
    "#f1c40f",
    "#2ecc71",
]

CLUSTER_DIM: list[str] = [
    "#1e3a5f",
    "#2d4a1a",
    "#5c4210",
    "#5c2a18",
    "#3d3a5f",
    "#0f4a38",
    "#4a2038",
    "#3a3a38",
    "#5c2020",
    "#1a3a5c",
    "#5c5010",
    "#1a4a30",
]

_AXIS = dict(
    range=[0, 1000],
    showgrid=True,
    gridcolor=GRID,
    zeroline=False,
    showticklabels=False,
)

MAP_LAYOUT = dict(
    width=480,
    height=480,
    margin=dict(l=4, r=4, t=4, b=4),
)

PLOTLY_CHART_CONFIG: dict[str, Any] = {
    "displayModeBar": True,
    "responsive": False,
}


def apply_dark_theme(fig: go.Figure, *, square_map: bool = False) -> go.Figure:
    layout: dict[str, Any] = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL_BG,
        margin=dict(l=4, r=4, t=4, b=4),
        font=dict(color=WHITE, size=10),
    )
    if square_map:
        layout.update(MAP_LAYOUT)
    fig.update_layout(**layout)
    return fig


def _finalize_map_figure(fig: go.Figure) -> go.Figure:
    apply_dark_theme(fig, square_map=True)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS, scaleanchor="x", scaleratio=1)
    return fig


def _empty_map_figure(message: str = "") -> go.Figure:
    fig = go.Figure()
    _finalize_map_figure(fig)
    if message:
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=GREY, size=12),
        )
    return fig


def _add_depot(fig: go.Figure, depot: list[float]) -> None:
    fig.add_trace(
        go.Scatter(
            x=[depot[0]],
            y=[depot[1]],
            mode="markers",
            marker=dict(symbol="square", size=10, color=YELLOW),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _add_nodes(
    fig: go.Figure,
    customers: list[list[float]],
    color: str,
    size: int = 4,
) -> None:
    if not customers:
        return
    xs = [c[0] for c in customers]
    ys = [c[1] for c in customers]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=size, color=color),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _add_routes(
    fig: go.Figure,
    routes: list[list[int]],
    customer_ids: list[int],
    customers: list[list[float]],
    color: str,
    width: float = 0.8,
) -> None:
    if not routes or not customer_ids:
        return
    cid_to_xy = {cid: customers[i] for i, cid in enumerate(customer_ids)}
    for route in routes:
        if not route:
            continue
        xs: list[float | None] = []
        ys: list[float | None] = []
        for cid in route:
            if cid in cid_to_xy:
                xs.append(cid_to_xy[cid][0])
                ys.append(cid_to_xy[cid][1])
        if len(xs) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def build_bg_perturb_figure(snapshot: dict[str, Any]) -> go.Figure:
    """Before BG-AILS: red routes chosen for perturbation, dim cluster colors otherwise."""
    depot = snapshot.get("depot") or [500.0, 500.0]
    customer_ids = snapshot.get("customer_ids") or []
    customers = snapshot.get("customers") or []
    routes = snapshot.get("routes") or []
    route_cluster_ids = snapshot.get("route_cluster_ids") or []
    perturbed = set(snapshot.get("perturbed_route_indices") or [])

    fig = go.Figure()
    if routes:
        for i, route in enumerate(routes):
            if i in perturbed:
                continue
            cid = int(route_cluster_ids[i]) if i < len(route_cluster_ids) else 0
            color = CLUSTER_DIM[cid % len(CLUSTER_DIM)]
            _add_routes(fig, [route], customer_ids, customers, color, 0.8)
        for i, route in enumerate(routes):
            if i in perturbed:
                _add_routes(fig, [route], customer_ids, customers, RED, 1.6)
    _add_depot(fig, depot)
    return _finalize_map_figure(fig)


def build_bg_result_figure(snapshot: dict[str, Any]) -> go.Figure:
    """After BG-AILS: all routes white, routes modified by AILS in green."""
    depot = snapshot.get("depot") or [500.0, 500.0]
    customer_ids = snapshot.get("customer_ids") or []
    customers = snapshot.get("customers") or []
    routes = snapshot.get("routes") or []
    changed = set(snapshot.get("changed_route_indices") or [])

    fig = go.Figure()
    _add_nodes(fig, customers, WHITE, size=4)
    if routes:
        for i, route in enumerate(routes):
            if i in changed:
                continue
            _add_routes(fig, [route], customer_ids, customers, WHITE, 0.8)
        for i, route in enumerate(routes):
            if i in changed:
                _add_routes(fig, [route], customer_ids, customers, GREEN, 1.6)
    _add_depot(fig, depot)
    return _finalize_map_figure(fig)


def _build_bg_ails_legacy_figure(snapshot: dict[str, Any]) -> go.Figure:
    """Older single-file bg_ails snapshots (combined perturb + result layers)."""
    depot = snapshot.get("depot") or [500.0, 500.0]
    customer_ids = snapshot.get("customer_ids") or []
    customers = snapshot.get("customers") or []
    routes = snapshot.get("routes") or []
    route_cluster_ids = snapshot.get("route_cluster_ids") or []
    perturbed = set(snapshot.get("perturbed_route_indices") or [])
    changed = set(snapshot.get("changed_route_indices") or [])

    fig = go.Figure()
    if routes:
        for i, route in enumerate(routes):
            if i in changed or i in perturbed:
                continue
            cid = int(route_cluster_ids[i]) if i < len(route_cluster_ids) else 0
            _add_routes(fig, [route], customer_ids, customers, CLUSTER_DIM[cid % len(CLUSTER_DIM)], 0.8)
        for i, route in enumerate(routes):
            if i in perturbed and i not in changed:
                _add_routes(fig, [route], customer_ids, customers, RED, 1.6)
        for i, route in enumerate(routes):
            if i in changed:
                _add_routes(fig, [route], customer_ids, customers, GREEN, 1.6)
    _add_depot(fig, depot)
    return _finalize_map_figure(fig)


def build_iteration_figure(snapshot: dict[str, Any] | None) -> go.Figure:
    """Build the current-iteration VRP canvas from a snapshot dict."""
    if snapshot is None:
        return _empty_map_figure()

    depot = snapshot.get("depot") or [500.0, 500.0]
    customer_ids = snapshot.get("customer_ids") or []
    customers = snapshot.get("customers") or []
    phase_num = int(snapshot.get("phase_num") or 1)
    cluster_assignments = snapshot.get("cluster_assignments")
    routes = snapshot.get("routes")
    route_cluster_ids = snapshot.get("route_cluster_ids")
    perturbed = set(snapshot.get("perturbed_route_indices") or [])
    changed = set(snapshot.get("changed_route_indices") or [])
    selected = set(snapshot.get("selected_route_indices") or [])

    fig = go.Figure()

    if phase_num == 1:
        if cluster_assignments and customers and customer_ids:
            colors = [
                CLUSTER_COLORS[
                    int(cluster_assignments.get(str(cid), cluster_assignments.get(cid, 0))) % len(CLUSTER_COLORS)
                ]
                for cid in customer_ids
            ]
            xs = [c[0] for c in customers]
            ys = [c[1] for c in customers]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=4, color=colors),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            _add_nodes(fig, customers, GREY)
        _add_depot(fig, depot)

    elif phase_num == 2:
        if cluster_assignments and customers and customer_ids:
            colors = [
                CLUSTER_COLORS[
                    int(cluster_assignments.get(str(cid), cluster_assignments.get(cid, 0))) % len(CLUSTER_COLORS)
                ]
                for cid in customer_ids
            ]
            xs = [c[0] for c in customers]
            ys = [c[1] for c in customers]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=4, color=colors),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        if routes:
            for i, route in enumerate(routes):
                cid = 0
                if route_cluster_ids and i < len(route_cluster_ids):
                    cid = int(route_cluster_ids[i])
                color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                _add_routes(fig, [route], customer_ids, customers, color, 0.8)
        _add_depot(fig, depot)

    elif phase_num == 3:
        stage = snapshot.get("bg_stage")
        if stage == "pre":
            return build_bg_perturb_figure(snapshot)
        if stage == "post":
            return build_bg_result_figure(snapshot)
        return _build_bg_ails_legacy_figure(snapshot)

    elif phase_num == 4:
        _add_nodes(fig, customers, WHITE, size=4)
        if routes:
            for i, route in enumerate(routes):
                if i in selected or not selected:
                    _add_routes(fig, [route], customer_ids, customers, WHITE, 1.6 if i in selected else 0.8)
        _add_depot(fig, depot)

    elif phase_num == 5:
        _add_nodes(fig, customers, WHITE, size=4)
        if routes:
            for i, route in enumerate(routes):
                if i not in changed:
                    _add_routes(fig, [route], customer_ids, customers, "#888888", 0.8)
            for i, route in enumerate(routes):
                if i in changed:
                    _add_routes(fig, [route], customer_ids, customers, GREEN, 1.6)
        _add_depot(fig, depot)

    else:
        _add_nodes(fig, customers, GREY)
        _add_depot(fig, depot)

    return _finalize_map_figure(fig)


def build_best_solution_figure(best: dict[str, Any] | None) -> go.Figure:
    """Build the current best solution VRP canvas."""
    if best is None:
        return _empty_map_figure("No solution yet")

    depot = best.get("depot") or [500.0, 500.0]
    customer_ids = best.get("customer_ids") or []
    customers = best.get("customers") or []
    routes = best.get("routes") or []

    fig = go.Figure()
    _add_nodes(fig, customers, WHITE, size=3)
    _add_routes(fig, routes, customer_ids, customers, WHITE, 0.8)
    _add_depot(fig, depot)
    return _finalize_map_figure(fig)


def build_trajectory_figure(
    summary_lines: list[dict[str, Any]],
    bks: float | None,
    improve_events: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """Build cost trajectory chart (160px height)."""
    fig = go.Figure()
    apply_dark_theme(fig)
    fig.update_layout(height=160)

    if not summary_lines:
        fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
        return fig

    # 1-based display, matching the ITERATION metric and run.log.
    iters = [int(s.get("iteration", 0)) + 1 for s in summary_lines]
    iter_costs = [float(s.get("iter_cost", 0)) for s in summary_lines]
    best_costs: list[float] = []
    running = float("inf")
    for c in iter_costs:
        running = min(running, c)
        best_costs.append(running)

    fig.add_trace(
        go.Scatter(
            x=iters,
            y=iter_costs,
            mode="lines",
            line=dict(color="#1e3a5f", width=1),
            name="iter",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=best_costs,
            mode="lines",
            line=dict(color=BLUE, width=2),
            name="best",
            showlegend=False,
        )
    )

    if bks is not None and math.isfinite(bks):
        fig.add_hline(y=bks, line_dash="dot", line_color=YELLOW, line_width=1)

    phase_colors = {
        "bg_ails": GREEN,
        "sp_sc": ORANGE,
        "standard_ails": PURPLE,
    }
    if improve_events:
        for ev in improve_events:
            phase = str(ev.get("phase_name", ""))
            if phase not in phase_colors:
                continue
            it = ev.get("iteration")
            cost = ev.get("cost")
            if it is None or cost is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[int(it) + 1],
                    y=[float(cost)],
                    mode="markers",
                    marker=dict(size=6, color=phase_colors[phase]),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


def _value_matches(selected: Any, val: Any) -> bool:
    if selected is None:
        return False
    if val == selected:
        return True
    if str(val) == str(selected):
        return True
    if isinstance(selected, (int, float)) and isinstance(val, (int, float)):
        return abs(float(val) - float(selected)) < 1e-9
    # paradigm wheel uses vertex/route; jsonl may use vb/rb
    if str(val) in ("vertex", "route") and str(selected) in ("vb", "rb", "vertex", "route"):
        return (
            (str(val) == "vertex" and str(selected) in ("vb", "vertex"))
            or (str(val) == "route" and str(selected) in ("rb", "route"))
        )
    return False


def build_haos_figure(haos: dict[str, Any] | None) -> str:
    """
    Build HAOS weight bars as HTML for st.markdown().
    Six levels: L1 k, L2 λ, L3 paradigm, L4a vb, L4b rb, L5 solver.
    """
    if haos is None or "levels" not in haos:
        return "<p style='color:#6b7280'>Waiting for HAOS weights…</p>"

    levels = haos["levels"]
    if not isinstance(levels, dict):
        return "<p style='color:#6b7280'>No HAOS levels</p>"

    from drispi.dashboard.formatting import format_method, format_paradigm_short

    paradigm_raw = str(haos.get("paradigm") or "").lower()
    if paradigm_raw in ("vb", "vertex"):
        paradigm = "vertex"
    elif paradigm_raw in ("rb", "route"):
        paradigm = "route"
    else:
        paradigm = paradigm_raw
    sel_k = haos.get("k")
    sel_lam = haos.get("lambda_demand")
    sel_method = haos.get("method")
    sel_solver = haos.get("solver")

    level_defs: list[tuple[str, str, Any, str, bool]] = [
        ("L1", "k", sel_k, "level_1_k", False),
        ("L2", "λ", sel_lam, "level_2_lambda_demand", False),
        ("L3", "paradigm", paradigm, "level_3_paradigm", False),
        ("L4a", "vb", sel_method, "level_4a_vertex_method", paradigm == "route"),
        ("L4b", "rb", sel_method, "level_4b_route_method", paradigm == "vertex"),
        ("L5", "solver", sel_solver, "level_5_solver", False),
    ]

    def _render_level_block(
        label: str,
        short: str,
        selected: Any,
        key: str,
        dimmed: bool,
    ) -> str:
        level_data = levels.get(key)
        if level_data is None:
            return (
                f"<div style='margin:6px 0;color:#444'>"
                f"<span style='width:28px;display:inline-block'>{label}</span> "
                f"<span style='color:#555'>—</span></div>"
            )

        values = level_data.get("values") or []
        probs = level_data.get("probabilities") or []
        if len(probs) != len(values):
            probs = [1.0 / max(len(values), 1)] * len(values)

        header_suffix = " (inactive)" if dimmed else ""
        bars: list[str] = []
        for val, prob in zip(values, probs, strict=False):
            pct = float(prob) * 100.0
            highlight = (not dimmed) and _value_matches(selected, val)
            bar_color = YELLOW if highlight else "#333333"
            text_color = YELLOW if highlight else GREY
            if key in ("level_4a_vertex_method", "level_4b_route_method"):
                display = format_method(str(val))
            elif key == "level_3_paradigm":
                display = format_paradigm_short(str(val))
            else:
                display = str(val)
            display = html.escape(display)
            row_opacity = "opacity:0.35;" if dimmed else ""
            bars.append(
                f"<div style='display:flex;align-items:center;margin:1px 0;{row_opacity}'>"
                f"<span style='width:72px;font-size:10px;color:{text_color}'>{display}</span>"
                f"<div style='width:45%;background:#1a1a1a;height:5px;margin-left:4px'>"
                f"<div style='width:{pct:.1f}%;background:{bar_color};height:5px'></div>"
                f"</div>"
                f"<span style='width:36px;text-align:right;font-size:10px;color:{GREY};"
                f"margin-left:6px'>{pct:.1f}%</span>"
                f"</div>"
            )

        dim_style = "opacity:0.35;" if dimmed else ""
        return (
            f"<div style='margin:6px 0;{dim_style}'>"
            f"<div style='color:{WHITE};font-size:11px;margin-bottom:4px'>"
            f"{label} {short}{header_suffix}</div>"
            f"{''.join(bars)}</div>"
        )

    left_col = "".join(
        _render_level_block(label, short, selected, key, dimmed)
        for label, short, selected, key, dimmed in level_defs[:3]
    )
    right_col = "".join(
        _render_level_block(label, short, selected, key, dimmed)
        for label, short, selected, key, dimmed in level_defs[3:]
    )

    return (
        f"<div style='font-family:monospace;font-size:11px;background:#0c0c0c;padding:8px;"
        f"display:flex;gap:20px'>"
        f"<div style='flex:1;min-width:0'>{left_col}</div>"
        f"<div style='flex:1;min-width:0'>{right_col}</div>"
        f"</div>"
    )
