"""
DRISPI Monitor — Real-time dashboard.

Launch:
    streamlit run drispi/dashboard/app.py -- --run-dir artifacts/runs/X-n1001-k43_0528_1432
Or via pipeline:
    python -m drispi.pipeline.runner instance.vrp --gui
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

import html

from drispi.dashboard import data, formatting, phase_hold, viz

PHASE_DISPLAY_NAMES: dict[str, str] = {
    "dissim+cluster": "clustering",
    "subclusters": "subcluster routing",
    "bg_ails": "boundary improvement",
    "sp_sc": "set covering / partitioning",
    "standard_ails": "standard improvement",
}

st.set_page_config(
    page_title="DRISPI Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 100%;}
    [data-testid="stMetricValue"] {font-size: 1.4rem;}
    .haos-roll-inline {color:#facc15;font-family:monospace;font-size:11px;text-align:center;margin:4px 0 8px 0;}
    .haos-roll-inline.muted {color:#888;}
    .map-plot-wrap [data-testid="stPlotlyChart"] {width:480px !important;max-width:100%;}
    div[data-testid="column"] div[data-testid="stVerticalBlock"] {gap: 0.25rem;}
    .metric-value-row {display:flex;align-items:center;gap:10px;margin:0;}
    .metric-value-main {font-size:1.4rem;color:#e8e8e8;line-height:1.2;}
    .metric-value-gap {font-size:0.95rem;color:#ef4444;line-height:1.2;}
    @keyframes statusPulse {0%, 100% {opacity:1;} 50% {opacity:0.25;}}
    .status-dot-pulse {animation: statusPulse 1.6s ease-in-out infinite;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _parse_run_dir() -> Path | None:
    if "--run-dir" in sys.argv:
        idx = sys.argv.index("--run-dir") + 1
        if idx < len(sys.argv):
            return Path(sys.argv[idx])
    return None


def _instance_name_from_run_dir(run_dir: Path, meta: dict[str, Any] | None) -> str:
    if meta is not None and meta.get("instance"):
        return str(meta["instance"])
    name = run_dir.name
    parts = name.rsplit("_", 2)
    if (
        len(parts) == 3
        and len(parts[1]) == 4
        and parts[1].isdigit()
        and len(parts[2]) == 4
        and parts[2].isdigit()
    ):
        return parts[0]
    return name


def _format_cost(val: float | None) -> str:
    if val is None or not (val == val):
        return "—"
    return f"{val:,.0f}"


def _format_pct(val: float | None) -> str:
    if val is None:
        return ""
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _runtime_str(run_dir: Path, status: dict[str, Any]) -> str:
    # Frozen final runtime once the run has ended.
    if status.get("status") != "running":
        elapsed = status.get("elapsed")
        if isinstance(elapsed, (int, float)):
            return _format_hms(float(elapsed))
    ts = data.parse_run_start_timestamp(run_dir)
    if ts is None:
        return "00:00:00"
    try:
        start = datetime.strptime(ts, "%H:%M:%S")
        now = datetime.now()
        start = start.replace(year=now.year, month=now.month, day=now.day)
        if start > now:
            start = start.replace(day=now.day - 1)
        return _format_hms((now - start).total_seconds())
    except ValueError:
        return "00:00:00"


def _status_badge(status: dict[str, Any]) -> str:
    state = str(status.get("status", "running"))
    if state == "cancelled":
        label, color, bg = "CANCELLED", "#f87171", "rgba(239,68,68,0.12)"
    elif state == "finished":
        reason = {"time_limit": "TIME LIMIT", "no_improve": "STAGNATION"}.get(
            str(status.get("stop_reason"))
        )
        label = f"FINISHED · {reason}" if reason else "FINISHED"
        color, bg = "#60a5fa", "rgba(59,130,246,0.12)"
    else:
        label, color, bg = "RUNNING", "#4ade80", "rgba(74,222,128,0.10)"
    dot_cls = ' class="status-dot-pulse"' if state == "running" else ""
    return (
        f'<span style="margin-left:auto;display:inline-flex;align-items:center;gap:8px;'
        f"padding:4px 14px;border-radius:14px;background:{bg};"
        f"border:1px solid {color}55;color:{color};font-size:0.8rem;font-weight:600;"
        f'letter-spacing:0.06em;white-space:nowrap">'
        f'<span{dot_cls} style="width:8px;height:8px;border-radius:50%;'
        f'background:{color};display:inline-block"></span>'
        f"{label}</span>"
    )


def _config_chips(
    meta: dict[str, Any] | None,
    *,
    time_limit_s: str,
    no_improve: int,
    max_no_improve: int,
) -> str:
    if meta is None:
        return f"time_limit {time_limit_s} · no_improve {no_improve}/{max_no_improve}"
    cfg = meta.get("config") or {}
    parts = []
    if "n_customers" in meta:
        parts.append(f"n={meta['n_customers']}")
    if "n_workers" in cfg:
        parts.append(f"workers={cfg['n_workers']}")
    haos_cfg = cfg.get("haos_config") or {}
    if "decay" in haos_cfg:
        parts.append(f"decay={haos_cfg['decay']}")
    for key, label in (
        ("max_no_improve", "max_no_improve"),
        ("warmup_iterations", "warmup"),
        ("sp_interval", "sp_interval"),
    ):
        val = cfg.get(key)
        if val is not None:
            parts.append(f"{label}={val}")
    parts.append(f"time_limit {time_limit_s}")
    parts.append(f"no_improve {no_improve}/{max_no_improve}")
    return " · ".join(parts)


def _metric_cost_with_gap(label: str, cost: float | None, gap_pct: float | None) -> None:
    gap_html = (
        f'<span class="metric-value-gap">{_format_pct(gap_pct)}</span>'
        if _format_pct(gap_pct)
        else ""
    )
    st.markdown(
        f'<p style="color:#888;font-size:0.75rem;margin:0 0 4px 0">{label}</p>'
        f'<div class="metric-value-row">'
        f'<span class="metric-value-main">{_format_cost(cost)}</span>'
        f"{gap_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _plot_map(fig: Any, key: str) -> None:
    st.markdown('<div class="map-plot-wrap">', unsafe_allow_html=True)
    st.plotly_chart(
        fig,
        use_container_width=False,
        config=viz.PLOTLY_CHART_CONFIG,
        key=key,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_inline_caption(line: str, *, muted: bool = False) -> None:
    if not line:
        return
    cls = "haos-roll-inline muted" if muted else "haos-roll-inline"
    st.markdown(
        f'<p class="{cls}">{html.escape(line)}</p>',
        unsafe_allow_html=True,
    )


def _render_haos_roll(roll: dict[str, Any] | None, *, muted: bool = False) -> None:
    _render_inline_caption(formatting.haos_roll_inline(roll), muted=muted)


def _phase_nav_buttons(
    key_prefix: str,
    iter_0: int,
    idx: int,
    n_phases: int,
    session_key: str,
    *,
    manual_key: str | None = None,
) -> None:
    if n_phases <= 1:
        return
    _, btn_c, _ = st.columns([1, 1, 1])
    with btn_c:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("◀", key=f"{key_prefix}_back_{iter_0}", disabled=idx <= 0):
                st.session_state[session_key] = idx - 1
                if manual_key is not None:
                    st.session_state[manual_key] = True
                st.rerun()
        with b2:
            if st.button("▶", key=f"{key_prefix}_fwd_{iter_0}", disabled=idx >= n_phases - 1):
                st.session_state[session_key] = idx + 1
                if manual_key is not None:
                    st.session_state[manual_key] = True
                st.rerun()


def _current_iteration_display_snapshot(
    run_dir: Path,
    current_iter_0: int,
) -> dict[str, Any] | None:
    hold_key = "curr_phase_hold"
    hold = st.session_state.get(hold_key)
    display, st.session_state[hold_key] = phase_hold.resolve_current_iteration_display(
        data.list_iteration_snapshots(run_dir, current_iter_0),
        current_iter_0,
        hold,
        time.time(),
    )
    return display


def _snapshot_map_key(prefix: str, snap: dict[str, Any] | None) -> str:
    if snap is None:
        return f"{prefix}_empty"
    stage = snap.get("bg_stage") or "main"
    return (
        f"{prefix}_i{snap.get('iteration')}_"
        f"p{snap.get('phase_num')}_{stage}_{snap.get('timestamp', '')}"
    )


def _plot_phase_snapshot(snap: dict[str, Any] | None, map_key: str) -> None:
    if snap is None:
        _plot_map(viz._empty_map_figure(), map_key)
    else:
        _plot_map(viz.build_iteration_figure(snap), map_key)


def _phase_display(snapshot: dict[str, Any]) -> str:
    return PHASE_DISPLAY_NAMES.get(
        str(snapshot.get("phase_name", "")),
        str(snapshot.get("phase_name", "—")),
    )


run_dir = _parse_run_dir()

if run_dir is None or not run_dir.exists():
    st.title("DRISPI Monitor")
    st.markdown(
        "**Waiting for run to start…**  \n"
        "Point dashboard at a run directory with `--run-dir <path>`"
    )
    time.sleep(1)
    st.rerun()

meta = data.get_run_metadata(run_dir)
run_status = data.get_run_status(run_dir)
state = data.get_latest_run_state(run_dir)
snapshot = data.get_latest_snapshot(run_dir)
best = data.get_best_solution(run_dir)
summaries = data.get_all_summary_lines(run_dir)

improve_events = data.get_improve_events(run_dir)

bks = meta.get("bks") if meta else None
instance_name = _instance_name_from_run_dir(run_dir, meta)
max_no_improve = (meta.get("config") or {}).get("max_no_improve", 100) if meta else 100

current_iter_0 = data.get_current_iteration_0(run_dir, state)
iteration_disp = current_iter_0 + 1
last_iter_0 = current_iter_0 - 1

cfg = meta.get("config") if meta else {}
time_limit = cfg.get("time_limit")
time_limit_s = f"{float(time_limit):.0f}s" if time_limit is not None else "—"
no_imp = int(state.get("no_improve", 0)) if state else 0
config_line = _config_chips(
    meta,
    time_limit_s=time_limit_s,
    no_improve=no_imp,
    max_no_improve=max_no_improve,
)

st.markdown(
    f"""
<div style="display:flex;align-items:center;gap:28px;margin-bottom:12px;flex-wrap:wrap">
  <div style="font-size:1.5rem;font-weight:700;color:#e8e8e8;white-space:nowrap">
    DRISPI Monitor <code style="color:#4ade80">{instance_name}</code>
  </div>
  <div style="color:#888;font-size:0.8rem;line-height:1.5">{config_line}</div>
  {_status_badge(run_status)}
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("RUNTIME", _runtime_str(run_dir, run_status))
with m2:
    st.metric("ITERATION", str(iteration_disp) if state else "—")
with m3:
    st.markdown(
        f"<p style='color:#888;font-size:0.75rem;margin:0 0 4px 0'>BKS</p>"
        f"<p style='color:#facc15;font-size:1.4rem;margin:0'>{_format_cost(bks)}</p>",
        unsafe_allow_html=True,
    )
with m4:
    _metric_cost_with_gap(
        "CURRENT BEST",
        state.get("best_cost") if state else None,
        state.get("delta_best_to_bks_pct") if state else None,
    )
with m5:
    _metric_cost_with_gap(
        "LAST ITER",
        state.get("iter_cost") if state else None,
        state.get("delta_to_bks_pct") if state else None,
    )

# Middle row: current iter | last iter | best
col_curr, col_prev, col_best = st.columns(3, gap="small")

curr_snap = _current_iteration_display_snapshot(run_dir, current_iter_0)
if curr_snap is None:
    curr_snap = snapshot

prev_snaps: list[dict[str, Any]] = []
if last_iter_0 >= 0:
    prev_snaps = data.list_iteration_snapshots(run_dir, last_iter_0)

prev_phase_key = f"prev_phase_idx_{last_iter_0}"
if prev_phase_key not in st.session_state:
    st.session_state[prev_phase_key] = max(0, len(prev_snaps) - 1) if prev_snaps else 0

with col_curr:
    phase_label = _phase_display(curr_snap) if curr_snap else "—"
    st.markdown(f"**current iteration · {phase_label}**")
    _render_haos_roll(data.get_haos_roll_for_iteration(run_dir, current_iter_0))
    _plot_phase_snapshot(curr_snap, _snapshot_map_key("curr", curr_snap))

with col_prev:
    if prev_snaps:
        prev_idx = max(0, min(int(st.session_state[prev_phase_key]), len(prev_snaps) - 1))
        st.session_state[prev_phase_key] = prev_idx
        prev_snap = prev_snaps[prev_idx]
        st.markdown(f"**last iteration · {_phase_display(prev_snap)}**")
    else:
        prev_idx = 0
        prev_snap = None
        st.markdown("**last iteration**")
    if last_iter_0 >= 0:
        _render_haos_roll(data.get_haos_roll_for_iteration(run_dir, last_iter_0), muted=True)
    _plot_phase_snapshot(
        prev_snap,
        "prev_empty" if prev_snap is None else f"prev_map_{last_iter_0}_{prev_idx}",
    )
    if prev_snaps:
        _phase_nav_buttons("prev", last_iter_0, prev_idx, len(prev_snaps), prev_phase_key)

with col_best:
    st.markdown("**current best solution**")
    _render_inline_caption(formatting.best_origin_inline(best), muted=True)
    if best:
        _plot_map(viz.build_best_solution_figure(best), "best_map")
    else:
        _plot_map(viz.build_best_solution_figure(None), "best_empty")

col_traj, col_haos_panel = st.columns(2)

with col_traj:
    st.markdown("**cost trajectory**")

    @st.cache_data(ttl=1)
    def _trajectory_fig(
        summaries_key: str,
        bks_val: float | None,
        improve_key: str,
    ) -> Any:
        del summaries_key, improve_key
        return viz.build_trajectory_figure(summaries, bks_val, improve_events)

    traj_key = str(len(summaries))
    imp_key = str(len(improve_events))
    st.plotly_chart(
        _trajectory_fig(traj_key, bks, imp_key),
        use_container_width=True,
        key="trajectory",
    )

with col_haos_panel:
    st.markdown("**HAOS operator weights**")

    @st.cache_data(ttl=1)
    def _haos_html(run_dir_str: str, mtime: float) -> str:
        del mtime
        haos = data.get_haos_weights(Path(run_dir_str))
        return viz.build_haos_figure(haos)

    jsonl_path = run_dir / "run.jsonl"
    mtime = jsonl_path.stat().st_mtime if jsonl_path.is_file() else 0.0
    st.markdown(_haos_html(str(run_dir), mtime), unsafe_allow_html=True)

# Keep polling only while the run is alive; freeze the page once it has
# finished or been cancelled.
if run_status.get("status", "running") == "running":
    time.sleep(1)
    st.rerun()
