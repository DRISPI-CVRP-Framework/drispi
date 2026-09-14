#!/usr/bin/env python3
"""Figure 2 — evaluated-touch maps, arm A vs arm C, shared LogNorm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.thesis_figstyle import apply, fig_size  # noqa: E402

DATA = ROOT / "data/results/bg_ails_ablation/figure2_touchmaps.json"
OUT = ROOT / "thesis/figures/bgails_touchmaps.pdf"


def _hulls(ax, data: dict) -> None:
    labels = {int(k): int(v) for k, v in data["labels"].items()}
    coords = data["coordinates"]
    by = {}
    for cid, lab in labels.items():
        by.setdefault(lab, []).append(coords[str(cid)])
    for pts in by.values():
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) < 3:
            continue
        hull = ConvexHull(arr)
        poly = arr[hull.vertices]
        poly = np.vstack([poly, poly[0]])
        ax.plot(poly[:, 0], poly[:, 1], color="#222222", lw=0.6, zorder=2)


def main() -> None:
    apply()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    depot = data["depot"]
    customers = data["customers"]
    xy = np.array([data["coordinates"][str(c)] for c in customers])
    eval_a = np.array(data["eval_A"], dtype=np.float64)
    eval_c = np.array(data["eval_C"], dtype=np.float64)
    vmin = max(1.0, float(np.min(np.concatenate([eval_a, eval_c]).clip(min=1))))
    vmax = float(np.max(np.concatenate([eval_a, eval_c]).clip(min=1)))
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))
    fig, axes = plt.subplots(1, 2, figsize=fig_size(1.0, 220))
    for ax, touches in ((axes[0], eval_a), (axes[1], eval_c)):
        _hulls(ax, data)
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=np.clip(touches, 1, None),
            cmap="YlOrRd",
            norm=norm,
            s=10,
            zorder=3,
            linewidths=0,
        )
        ax.plot(*depot, marker="*", color="black", markersize=8, zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if not DATA.is_file():
        raise SystemExit(f"missing {DATA}")
    main()
