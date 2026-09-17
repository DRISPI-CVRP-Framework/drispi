#!/usr/bin/env python3
"""BG-AILS mechanism, 2×2 panels. Instance-agnostic JSON consumer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.thesis_figstyle import OKABE_ITO, apply, fig_size  # noqa: E402

DATA = ROOT / "data/results/bg_ails_ablation/figure1_mechanism.json"
OUT = ROOT / "thesis/figures/bgails_mechanism.pdf"

TITLES = (
    "(a) Concatenated input",
    "(b) Ranks and selected pairs",
    "(c) After cross-reconnect",
    "(d) After AILS-II",
)


def _polyline(ax, seqs, coords, depot, color, lw=0.6, z=1):
    segs = []
    for seq in seqs:
        pts = [depot, *[coords[str(c)] for c in seq], depot]
        segs.append(pts)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=z))


def main() -> None:
    apply()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    depot = data["depot"]
    coords = data["coordinates"]
    customers = data["customers"]
    labels = {int(k): v for k, v in data["labels"].items()}
    ranks = np.array(data["ranks_hat"], dtype=np.float64)
    xy = np.array([coords[str(c)] for c in customers])
    fig, axes = plt.subplots(2, 2, figsize=fig_size(1.0, 430))
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    pad = 0.04 * max(xmax - xmin, ymax - ymin)
    lims = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)
    n_clusters = 1 + max(labels.values())
    cluster_colors = OKABE_ITO[1 : 1 + n_clusters]

    ax = axes[0, 0]
    cols = [cluster_colors[labels[c] % len(cluster_colors)] for c in customers]
    ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=8, zorder=3, linewidths=0)
    _polyline(ax, data["input_seqs"], coords, depot, "#888888")
    ax.plot(*depot, marker="*", color="black", markersize=8, zorder=4)

    ax = axes[0, 1]
    ax.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=8, zorder=2, linewidths=0)
    sizes = 8 + 40 * ranks
    ax.scatter(xy[:, 0], xy[:, 1], c=ranks, cmap="Blues", s=sizes, zorder=3, linewidths=0)
    pair_colors = OKABE_ITO[2:]
    input_seqs = data["input_seqs"]
    for step_i, step in enumerate(data["guided_trace"]):
        color = pair_colors[step_i % len(pair_colors)]
        for ridx in (step["i"], step["j"]):
            _polyline(ax, [input_seqs[ridx]], coords, depot, color, lw=1.4, z=4)
    ax.plot(*depot, marker="*", color="black", markersize=8, zorder=5)

    ax = axes[1, 0]
    ax.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=8, zorder=2, linewidths=0)
    pert = data["perturbed_seqs"]
    _polyline(ax, pert, coords, depot, "#bbbbbb", lw=0.5, z=1)
    for step_i, step in enumerate(data["guided_trace"]):
        color = pair_colors[step_i % len(pair_colors)]
        for ridx in (step["i"], step["j"]):
            seq = step["seq_i"] if ridx == step["i"] else step["seq_j"]
            a = step["a"] if ridx == step["i"] else step["b"]
            if 0 < a < len(seq):
                x, y = coords[str(seq[a])]
                ax.plot(x, y, "x", color=color, markersize=6, zorder=5)
            _polyline(ax, [seq], coords, depot, color, lw=1.2, z=4)
    ax.plot(*depot, marker="*", color="black", markersize=8, zorder=6)

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
    _polyline(ax, data["result_seqs"], coords, depot, "#888888", lw=0.5)
    ax.plot(*depot, marker="*", color="black", markersize=8, zorder=4)

    for ax, title in zip(axes.ravel(), TITLES):
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=6)
        for sp in ax.spines.values():
            sp.set_visible(True)

    # Equal panel sizes; colorbar sits in the right margin at the height of (d).
    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.04, wspace=0.18, hspace=0.22)
    pos = axes[1, 1].get_position()
    cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.025, pos.height])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"log(1 + evaluated touches)")
    cbar.ax.tick_params(which="both", length=3, width=0.6, direction="out", pad=2)
    cbar.ax.minorticks_off()
    cbar.outline.set_linewidth(0.6)
    # Continuous strip without anti-aliased segment gaps.
    cbar.solids.set_edgecolor("face")
    cbar.solids.set_linewidth(0)
    cbar.solids.set_rasterized(True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if not DATA.is_file():
        raise SystemExit(f"missing {DATA}; run analyze_bgails_ablation.py first")
    main()
