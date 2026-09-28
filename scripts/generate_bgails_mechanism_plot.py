#!/usr/bin/env python3
"""BG-AILS mechanism, 2×2 panels. Instance-agnostic JSON consumer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.thesis_figstyle import OKABE_ITO, TEXTWIDTH_PT, apply  # noqa: E402

DATA = ROOT / "data/results/bg_ails_ablation/figure1_mechanism.json"
OUT = ROOT / "thesis/figures/bgails_mechanism.pdf"

TITLES = (
    "(a) Concatenated input",
    "(b) Ranks and selected pairs",
    "(c) After cross-reconnect",
    "(d) After AILS-II",
)
# Six pair colors for panels (b) and (c). Light blue, yellow, orange, and pink
# stay; the blue that disappeared into the rank colormap and the dark green
# are replaced by light green and light burgundy.
PAIR_COLORS = (
    "#56B4E9",
    "#7DCE8A",
    "#F0E442",
    "#B84A5C",
    "#D55E00",
    "#CC79A7",
)


def _polyline(ax, seqs, coords, color, lw=0.6, z=1):
    """Customer-to-customer legs only. Depot spokes hide the route interior."""
    segs = []
    for seq in seqs:
        if len(seq) < 2:
            continue
        segs.append([coords[str(c)] for c in seq])
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=z))


def _offset_points(pts: list, delta: float) -> np.ndarray:
    """Shift a polyline sideways so a route used by two pairs stays two colors."""
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) < 2 or abs(delta) < 1e-9:
        return arr
    tangents = np.diff(arr, axis=0)
    lengths = np.maximum(np.linalg.norm(tangents, axis=1), 1e-9)
    tangents = tangents / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    vertex_n = np.empty_like(arr)
    vertex_n[0] = normals[0]
    vertex_n[-1] = normals[-1]
    vertex_n[1:-1] = normals[:-1] + normals[1:]
    nlen = np.maximum(np.linalg.norm(vertex_n, axis=1, keepdims=True), 1e-9)
    return arr + delta * vertex_n / nlen


def main() -> None:
    apply()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    depot = data["depot"]
    coords = data["coordinates"]
    customers = data["customers"]
    labels = {int(k): v for k, v in data["labels"].items()}
    ranks = np.array(data["ranks_hat"], dtype=np.float64)
    xy = np.array([coords[str(c)] for c in customers])
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    pad = 0.04 * max(xmax - xmin, ymax - ymin)
    lims = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)
    n_clusters = 1 + max(labels.values())
    cluster_colors = OKABE_ITO[1 : 1 + n_clusters]
    # Cells match the map, which is nearly square. A taller cell plus
    # equal aspect leaves a blank band between the rows and under (c)/(d).
    fig_w = TEXTWIDTH_PT / 72.27
    data_aspect = (lims[1] - lims[0]) / (lims[3] - lims[2])
    # Room for the colorbar label, a title clear of each border, and a
    # modest gap between the rows. The previous pack put titles on the
    # spines and clipped the colorbar; the one before that left empty bands.
    left, col_gap, cbar_gap, cbar_w, label_w = 0.08, 0.20, 0.14, 0.08, 1.00
    panel_w = (fig_w - left - col_gap - cbar_gap - cbar_w - label_w) / 2
    panel_h = panel_w / data_aspect
    title_band, bottom, top = 0.46, 0.20, 0.34
    fig_h = top + panel_h + title_band + panel_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _rect(x_in: float, y_in: float, w_in: float, h_in: float) -> list[float]:
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    y_bottom = bottom
    y_top = bottom + panel_h + title_band
    x_left = left
    x_right = left + panel_w + col_gap
    axes = np.empty((2, 2), dtype=object)
    for (row, col), (x_in, y_in) in {
        (0, 0): (x_left, y_top),
        (0, 1): (x_right, y_top),
        (1, 0): (x_left, y_bottom),
        (1, 1): (x_right, y_bottom),
    }.items():
        axes[row, col] = fig.add_axes(_rect(x_in, y_in, panel_w, panel_h))

    ax = axes[0, 0]
    cols = [cluster_colors[labels[c] % len(cluster_colors)] for c in customers]
    ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=8, zorder=3, linewidths=0)
    _polyline(ax, data["input_seqs"], coords, "#888888")
    ax.plot(*depot, marker="s", color="black", markersize=6, zorder=4)

    ax = axes[0, 1]
    ax.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=8, zorder=2, linewidths=0)
    sizes = 8 + 40 * ranks
    rank_grey = LinearSegmentedColormap.from_list("rank_grey", ["#F2F2F2", "#4A4A4A"])
    ax.scatter(xy[:, 0], xy[:, 1], c=ranks, cmap=rank_grey, s=sizes, zorder=3, linewidths=0)
    pair_colors = PAIR_COLORS
    input_seqs = data["input_seqs"]
    # Later pairs reuse some routes. Draw those twice, offset, so both colors show.
    route_uses: dict[int, int] = {}
    for step in data["guided_trace"]:
        for ridx in (step["i"], step["j"]):
            route_uses[ridx] = route_uses.get(ridx, 0) + 1
    seen_uses: dict[int, int] = {}
    pair_offset = 0.006 * max(xmax - xmin, ymax - ymin)
    for step_i, step in enumerate(data["guided_trace"]):
        color = pair_colors[step_i % len(pair_colors)]
        for ridx in (step["i"], step["j"]):
            seq = input_seqs[ridx]
            pts = [coords[str(c)] for c in seq]
            if route_uses[ridx] > 1:
                side = -1.0 if seen_uses.get(ridx, 0) % 2 == 0 else 1.0
                pts = _offset_points(pts, side * pair_offset)
            seen_uses[ridx] = seen_uses.get(ridx, 0) + 1
            if len(pts) >= 2:
                ax.add_collection(
                    LineCollection([pts], colors=color, linewidths=1.4, zorder=4)
                )
    ax.plot(*depot, marker="s", color="black", markersize=6, zorder=5)

    ax = axes[1, 0]
    ax.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=8, zorder=2, linewidths=0)
    pert = data["perturbed_seqs"]
    _polyline(ax, pert, coords, "#bbbbbb", lw=0.5, z=1)
    for step_i, step in enumerate(data["guided_trace"]):
        color = pair_colors[step_i % len(pair_colors)]
        for ridx in (step["i"], step["j"]):
            seq = step["seq_i"] if ridx == step["i"] else step["seq_j"]
            a = step["a"] if ridx == step["i"] else step["b"]
            if 0 < a < len(seq):
                x, y = coords[str(seq[a])]
                ax.plot(x, y, "x", color=color, markersize=6, zorder=5)
            _polyline(ax, [seq], coords, color, lw=1.2, z=4)
    ax.plot(*depot, marker="s", color="black", markersize=6, zorder=6)

    ax = axes[1, 1]
    touches = np.array(data["eval_touches"], dtype=np.float64)
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.log1p(touches),
        cmap="YlOrRd",
        s=10,
        zorder=3,
        linewidths=0,
    )
    _polyline(ax, data["result_seqs"], coords, "#888888", lw=0.5)
    ax.plot(*depot, marker="s", color="black", markersize=6, zorder=4)

    for ax, title in zip(axes.ravel(), TITLES):
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=8)
        for sp in ax.spines.values():
            sp.set_visible(True)

    # Colorbar sits in the right margin at the height of (d).
    cax = fig.add_axes(_rect(x_right + panel_w + cbar_gap, y_bottom, cbar_w, panel_h))
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"log(1 + evaluated touches)", labelpad=6)
    cbar.ax.tick_params(which="both", length=3, width=0.6, direction="out", pad=2)
    cbar.ax.minorticks_off()
    cbar.outline.set_linewidth(0.6)
    # Continuous strip without anti-aliased segment gaps.
    cbar.solids.set_edgecolor("face")
    cbar.solids.set_linewidth(0)
    cbar.solids.set_rasterized(True)

    # The colorbar label is narrower than the slot reserved for it, so the
    # panels sit left of the text block. Shift the whole group to center.
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in (*axes.ravel(), cax)]
    union = Bbox.union(boxes)
    shift = (fig.bbox.width - union.x1 - union.x0) / 2
    dx = shift / fig.bbox.width
    for ax in (*axes.ravel(), cax):
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if not DATA.is_file():
        raise SystemExit(f"missing {DATA}; run analyze_bgails_ablation.py first")
    main()
