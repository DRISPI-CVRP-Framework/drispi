"""Chapter figures for the revised HAOS section.

Scatter: arm-mean immediate reward against final wheel probability.
Reward: immediate-reward histogram beside the deferred histogram.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.thesis_figstyle import BLUE_LIGHT, HAOS_WHEEL_COLORS, OKABE_ITO, apply, fig_size  # noqa: E402

CREDITS = ROOT / "data" / "results" / "haos_immediate_credits.csv"
STATE = ROOT / "data" / "results" / "haos_wheel_state.csv"
DEFERRED = ROOT / "data" / "results" / "haos_deferred_credits.csv"
OUT = ROOT / "thesis" / "figures"

WHEELS = (
    ("k", r"$k$"),
    ("lambda", r"$\lambda_q$"),
    ("paradigm", "Paradigm"),
    ("vertex_method", "Vertex Method"),
    ("route_method", "Route Method"),
    ("solver", "Subsolver"),
)
# Mean within-run Spearman from notes/haos_diagnostics.txt. Paradigm has
# fewer than three qualifying arms, so it is not a number.
SPEARMAN = {
    "k": "0.081",
    "lambda": "0.045",
    "paradigm": None,
    "vertex_method": "0.043",
    "route_method": "0.030",
    "solver": "0.042",
}


def arm_label(value: object) -> str:
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
    else:
        number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def selection_rewards(credits: pd.DataFrame) -> pd.DataFrame:
    return credits.groupby(["run_id", "credited_iteration"], as_index=False).agg(
        reward=("reward", "sum"),
        k=("k", "first"),
        lambda_demand=("lambda_demand", "first"),
        paradigm=("paradigm", "first"),
        method=("method", "first"),
        solver=("solver", "first"),
    )


def arm_draws(selections: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    def add(frame: pd.DataFrame, wheel: str, arm: pd.Series) -> None:
        parts.append(
            pd.DataFrame(
                {
                    "run_id": frame["run_id"].to_numpy(),
                    "reward": frame["reward"].to_numpy(),
                    "wheel": wheel,
                    "arm": arm.map(arm_label).to_numpy(),
                }
            )
        )

    add(selections, "k", selections["k"])
    add(selections, "lambda", selections["lambda_demand"])
    add(selections, "paradigm", selections["paradigm"].astype(str))
    add(selections, "solver", selections["solver"].astype(str))
    vertex = selections["paradigm"].eq("vertex")
    route = selections["paradigm"].eq("route")
    add(selections.loc[vertex], "vertex_method", selections.loc[vertex, "method"].astype(str))
    add(selections.loc[route], "route_method", selections.loc[route, "method"].astype(str))
    long = pd.concat(parts, ignore_index=True)
    return long.groupby(["run_id", "wheel", "arm"], as_index=False).agg(
        n=("reward", "size"),
        mean_reward=("reward", "mean"),
    )


def final_probabilities(state: pd.DataFrame) -> pd.DataFrame:
    last = state.groupby(["run_id", "wheel"])["iteration"].transform("max")
    final = state.loc[state["iteration"].eq(last), ["run_id", "wheel", "arm", "probability", "n_arms"]].copy()
    final["arm"] = final["arm"].map(arm_label)
    return final.drop_duplicates(["run_id", "wheel", "arm"])


def scatter_frame() -> pd.DataFrame:
    credits = pd.read_csv(CREDITS)
    state = pd.read_csv(STATE)
    qualified = arm_draws(selection_rewards(credits))
    qualified = qualified.loc[qualified["n"].ge(5)]
    merged = qualified.merge(
        final_probabilities(state),
        on=["run_id", "wheel", "arm"],
        how="left",
        indicator=True,
    )
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise SystemExit(f"{missing} qualifying arms have no final probability")
    return merged.drop(columns="_merge")


def plot_scatter(points: pd.DataFrame) -> None:
    apply()
    lo = float(points["mean_reward"].min())
    hi = float(points["mean_reward"].max())
    pad = max(0.05, 0.04 * (hi - lo))
    xlim = (lo - pad, hi + pad)
    print(f"scatter reward range {lo:.3f} to {hi:.3f}; axis {xlim[0]:.3f} to {xlim[1]:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=fig_size(1.0, 340), sharex=True)
    for ax, (wheel, title), color in zip(axes.ravel(), WHEELS, HAOS_WHEEL_COLORS):
        panel = points.loc[points["wheel"].eq(wheel)]
        ax.scatter(
            panel["mean_reward"],
            panel["probability"],
            s=9,
            color=color,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )
        arms = sorted(int(a) for a in panel["n_arms"].unique())
        for count in arms:
            ax.axhline(1.0 / count, color="0.35", linewidth=0.7, linestyle="--", zorder=1)
        if len(arms) == 1:
            ref = rf"$1/{arms[0]}$"
        else:
            ref = ", ".join(rf"$1/{count}$" for count in arms)
        note = SPEARMAN[wheel]
        text = "Spearman: not computed" if note is None else f"Spearman: {note}"
        ax.text(
            0.03,
            0.97,
            f"{text}\n{ref}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.75", "boxstyle": "round,pad=0.25", "alpha": 0.92},
            zorder=3,
        )
        ymax = max(float(panel["probability"].max()), max(1.0 / count for count in arms))
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.grid(True, linestyle=":", color="gray", alpha=0.6)
        ax.set_axisbelow(True)
        print(f"  {wheel}: {len(panel)} points, A={arms}")
    for ax in axes[1]:
        ax.set_xlabel("Mean immediate reward")
    for ax in axes[:, 0]:
        ax.set_ylabel("Final probability")
    fig.tight_layout()
    path = OUT / "haos_scatter.pdf"
    fig.savefig(path)
    print(f"Wrote {path}")


def plot_reward() -> None:
    credits = pd.read_csv(CREDITS)
    selections = selection_rewards(credits)
    counts = selections["reward"].value_counts().sort_index()
    n = int(len(selections))
    n5 = int(counts.get(5, 0))
    if n != 19803 or n5 != 16591:
        raise SystemExit(f"unexpected immediate histogram: n={n}, R=5 count={n5}")
    deferred = pd.read_csv(DEFERRED)
    dcounts = deferred["reward"].value_counts().sort_index()
    print(f"deferred n={len(deferred)} counts={dcounts.to_dict()}")

    apply()
    fig, axes = plt.subplots(1, 2, figsize=fig_size(1.0, 200))
    left, right = axes
    left.bar(counts.index.astype(float), counts.to_numpy(), width=0.65, color=BLUE_LIGHT, zorder=2)
    left.annotate(
        "16,591 (84%)",
        xy=(5, n5),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    left.set_ylim(0, n5 * 1.16)
    ticks = [int(value) for value in counts.index]
    left.set_xticks(ticks)
    left.set_xticklabels(
        [str(value) for value in ticks],
        fontsize=8,
    )
    for label, value in zip(left.get_xticklabels(), ticks):
        if value == 10:
            label.set_ha("right")
        elif value == 11:
            label.set_ha("left")
    left.set_xlabel(r"Immediate reward $R$")
    left.set_ylabel("Selections")
    left.set_title("(a) Immediate")
    left.yaxis.grid(True, linestyle=":", color="gray", alpha=0.6)
    left.set_axisbelow(True)

    right.bar(dcounts.index.astype(float), dcounts.to_numpy(), width=0.65, color=OKABE_ITO[1], zorder=2)
    right.set_xticks([0, 3, 7])
    right.set_xlim(-1.2, 8.2)
    right.set_xlabel("Deferred reward")
    right.set_ylabel("SC/SP applies")
    right.set_title("(b) Deferred")
    right.yaxis.grid(True, linestyle=":", color="gray", alpha=0.6)
    right.set_axisbelow(True)
    fig.tight_layout()
    path = OUT / "haos_reward.pdf"
    fig.savefig(path)
    print(f"Wrote {path}")


def main() -> None:
    plot_scatter(scatter_frame())
    plot_reward()


if __name__ == "__main__":
    main()
