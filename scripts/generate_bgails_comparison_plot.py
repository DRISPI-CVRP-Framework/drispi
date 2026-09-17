#!/usr/bin/env python3
"""BG-AILS ablation comparison: AILS-II, Control, and BG-AILS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.thesis_figstyle import OKABE_ITO, apply, fig_size  # noqa: E402

OUT = ROOT / "thesis/figures/bgails_comparison.pdf"
DATA = ROOT / "data/results/bg_ails_ablation/figure4_comparison.json"

ARM_ORDER = ("AILS-II", "Control", "BG-AILS")
ARM_FROM_LETTER = {"A": "AILS-II", "B": "Control", "E": "BG-AILS"}
COLORS = {
    "AILS-II": OKABE_ITO[5],
    "Control": OKABE_ITO[1],
    "BG-AILS": OKABE_ITO[6],
}
TOP_OUTLIERS_OMITTED = 3


def _from_artifacts() -> dict:
    from drispi.ablation.stats import (  # noqa: WPS433
        _collect_performance,
        _pair_block,
        holm,
        instance_means,
    )

    rows = _collect_performance()
    means = instance_means(rows)
    names = sorted(n for n in means if set(means[n]) >= {"A", "B", "E"})
    vectors = {
        label: [float(means[n][letter]) for n in names]
        for letter, label in ARM_FROM_LETTER.items()
    }
    a = np.array(vectors["AILS-II"], dtype=np.float64)
    b = np.array(vectors["Control"], dtype=np.float64)
    e = np.array(vectors["BG-AILS"], dtype=np.float64)
    block_ea = _pair_block(e - a)
    block_eb = _pair_block(e - b)
    adj = holm([float(block_ea["wilcoxon_p"]), float(block_eb["wilcoxon_p"])])
    block_ea["holm_p"] = adj[0]
    block_eb["holm_p"] = adj[1]
    out = {
        "n_instances": len(names),
        "response": "pct_improvement_vs_checkpoint",
        "arms": vectors,
        "BG-AILS_minus_AILS-II": block_ea,
        "BG-AILS_minus_Control": block_eb,
        "family_size": 2,
    }
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def load_data() -> dict:
    try:
        from drispi.ablation import config as ac  # noqa: WPS433

        if ac.PERF_DIR.exists() and any(ac.PERF_DIR.glob("*/*/performance_summary.json")):
            return _from_artifacts()
    except Exception:
        pass
    if DATA.is_file():
        return json.loads(DATA.read_text(encoding="utf-8"))
    raise SystemExit(f"missing {DATA} and no performance artifacts")


def main() -> None:
    apply()
    data = load_data()
    series = [np.array(data["arms"][name], dtype=np.float64) for name in ARM_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=fig_size(1.0, 230), constrained_layout=True)

    ax = axes[0]
    bp = ax.boxplot(
        series,
        tick_labels=list(ARM_ORDER),
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none"},
    )
    for patch, name in zip(bp["boxes"], ARM_ORDER):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    means = [float(np.mean(y)) for y in series]
    ax.scatter(
        np.arange(1, len(ARM_ORDER) + 1),
        means,
        marker="D",
        s=28,
        color="black",
        zorder=5,
        label="Mean",
    )
    flat = np.sort(np.concatenate(series))
    y_hi = float(flat[-(TOP_OUTLIERS_OMITTED + 1)]) * 1.05
    ax.set_ylim(0.0, y_hi)
    ax.text(
        0.98,
        0.98,
        f"top {TOP_OUTLIERS_OMITTED} outliers omitted",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
    )
    ax.set_ylabel("Cost improvement vs. checkpoint (%)")
    ax.set_title("(a) Per-instance mean")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)
    ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.6)

    ax = axes[1]
    ails = np.array(data["arms"]["AILS-II"], dtype=np.float64)
    control = np.array(data["arms"]["Control"], dtype=np.float64)
    bg = np.array(data["arms"]["BG-AILS"], dtype=np.float64)
    diffs = [bg - ails, bg - control]
    diff_labels = ["BG-AILS minus\nAILS-II", "BG-AILS minus\nControl"]
    diff_colors = [COLORS["AILS-II"], COLORS["Control"]]
    bp = ax.boxplot(
        diffs,
        tick_labels=diff_labels,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none"},
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], diff_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    ax.axhline(0.0, color="#333333", ls="--", lw=0.8)
    ax.set_ylabel("Paired difference (pp)")
    ax.set_title("(b) BG-AILS minus baseline")
    ax.grid(True, axis="y", linestyle=":", color="gray", alpha=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
