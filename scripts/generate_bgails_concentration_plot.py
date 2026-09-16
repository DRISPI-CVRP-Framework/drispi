#!/usr/bin/env python3
"""Figure 3 — touch concentration by boundary-rank decile, arms A/B/E."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.thesis_figstyle import OKABE_ITO, apply, fig_size  # noqa: E402

DATA = ROOT / "data/results/bg_ails_ablation/figure3_concentration.json"
OUT = ROOT / "thesis/figures/bgails_concentration.pdf"
ARMS = ("A", "B", "E")
COLORS = {"A": OKABE_ITO[5], "B": OKABE_ITO[1], "E": OKABE_ITO[6]}


def main() -> None:
    apply()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=fig_size(1.0, 200))
    for arm in ARMS:
        mean = np.array(data["mean_share"][arm], dtype=np.float64)
        lo = np.array(data["ci_lo"][arm], dtype=np.float64)
        hi = np.array(data["ci_hi"][arm], dtype=np.float64)
        ax.plot(x, mean, color=COLORS[arm], lw=1.6, label=arm)
        ax.fill_between(x, lo, hi, color=COLORS[arm], alpha=0.18, linewidth=0)
    tau_x = data.get("tau_decile_position", 5.5)
    ax.axvline(tau_x, color="#333333", ls="--", lw=0.8)
    ax.set_xlabel(r"boundary-rank decile (pre-threshold $\hat b_i$)")
    ax.set_ylabel("share of evaluated touches")
    ax.set_xticks(x)
    ax.set_ylim(0.0, None)
    ax.legend(frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if not DATA.is_file():
        raise SystemExit(f"missing {DATA}")
    main()
