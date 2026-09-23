#!/usr/bin/env python3
"""HAOS weight figures and the numbers quoted in Section 4.2.3.

Final weights come from haos_weights_final.json. Entropy trajectories come
from the haos_roll events in run.jsonl that carry a levels payload.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.section_4_2_numbers import load  # noqa: E402
from scripts.thesis_figstyle import HAOS_WHEEL_COLORS, apply, fig_size  # noqa: E402

LOG_ROOT = ROOT / "data/results/lagrange_benchmark"
OUT_JSON = ROOT / "data/results/haos_campaign_lagrange.json"
FIG_ENTROPY = ROOT / "thesis/figures/haos_entropy.pdf"
FIG_K = ROOT / "thesis/figures/haos_k_heatmap.pdf"
FIG_SEEDS = ROOT / "thesis/figures/haos_seed_agreement.pdf"
FIG_BUDGET = ROOT / "thesis/figures/haos_top_arm.pdf"

LEVELS = (
    ("level_1_k", r"$k$"),
    ("level_2_lambda_demand", r"$\lambda_q$"),
    ("level_3_paradigm", "Paradigm"),
    ("level_4a_vertex_method", "Vertex method"),
    ("level_4b_route_method", "Route method"),
    ("level_5_solver", "Subsolver"),
)
LEVEL_KEYS = [key for key, _ in LEVELS]


def norm_entropy(probs: list[float]) -> float:
    p = np.asarray(probs, dtype=float)
    n_arms = len(p)
    if n_arms <= 1:
        return 0.0
    positive = p[p > 0]
    positive = positive / positive.sum()
    return float(-np.sum(positive * np.log(positive)) / np.log(n_arms))


def tv(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys)


def _seed_count(cust: str) -> int | None:
    if "(" not in cust:
        return None
    return int(cust.split("(")[1].split(")")[0])


def _runs() -> list[Path]:
    paths = sorted(LOG_ROOT.glob("*/seed*/**/haos_weights_final.json"))
    if len(paths) != 300:
        raise SystemExit(f"expected 300 HAOS finals, found {len(paths)}")
    return paths


def _instance_name(path: Path) -> str:
    # .../X-n1048-k237/seed11/<run>/haos_weights_final.json
    return "XL-" + path.parents[2].name[2:]


def _seed(path: Path) -> int:
    return int(path.parents[1].name.removeprefix("seed"))


def load_finals() -> list[dict]:
    rows = []
    for path in _runs():
        payload = json.loads(path.read_text(encoding="utf-8"))
        levels = {}
        for key, _label in LEVELS:
            level = payload["levels"][key]
            levels[key] = {
                "values": level["values"],
                "prob": {
                    str(value): float(prob)
                    for value, prob in zip(level["values"], level["probabilities"])
                },
            }
        rows.append(
            {
                "instance": _instance_name(path),
                "seed": _seed(path),
                "iterations": int(payload["iterations_completed"]),
                "levels": levels,
            }
        )
    return rows


def load_entropy() -> dict[str, list[dict[int, float]]]:
    """Per level, one dict of iteration -> entropy for each of the 300 runs."""
    series: dict[str, list[dict[int, float]]] = {key: [] for key in LEVEL_KEYS}
    for path in _runs():
        jsonl = path.with_name("run.jsonl")
        by_iter: dict[str, dict[int, float]] = {key: {} for key in LEVEL_KEYS}
        with jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if '"levels"' not in line or '"haos_roll"' not in line:
                    continue
                event = json.loads(line)
                if event.get("type") != "haos_roll" or "levels" not in event:
                    continue
                iteration = int(event["iteration"])
                for key in LEVEL_KEYS:
                    by_iter[key][iteration] = norm_entropy(event["levels"][key]["probabilities"])
        for key in LEVEL_KEYS:
            series[key].append(by_iter[key])
    return series


def entropy_summary(series: dict[str, list[dict[int, float]]]) -> dict:
    lengths = [max(run) + 1 for run in series[LEVEL_KEYS[0]] if run]
    median_len = int(np.median(lengths))
    summary = {"median_completed_iterations": median_len, "levels": {}}
    for key, _label in LEVELS:
        runs = series[key]
        never = 0
        for run in runs:
            if not run or min(run.values()) >= 0.75:
                never += 1
        first = {"0.9": None, "0.75": None}
        medians = []
        for iteration in range(median_len):
            vals = [run[iteration] for run in runs if iteration in run]
            if not vals:
                continue
            medians.append(float(np.median(vals)))
        for threshold in (0.9, 0.75):
            for iteration, median in enumerate(medians):
                if median < threshold:
                    first[str(threshold)] = iteration + 1
                    break
        summary["levels"][key] = {
            "runs_never_below_0.75": never,
            "median_first_below_0.9": first["0.9"],
            "median_first_below_0.75": first["0.75"],
        }
    summary["median_curve_length"] = median_len
    return summary, median_len


def plot_entropy(series: dict[str, list[dict[int, float]]], median_len: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=fig_size(1.0, 320), sharex=True, sharey=True)
    x = np.arange(1, median_len + 1)
    titles = (
        r"$k$",
        r"$\lambda_q$",
        "Paradigm",
        "Vertex Method",
        "Route Method",
        "Subsolver",
    )
    for ax, (key, _label), title, color in zip(axes.ravel(), LEVELS, titles, HAOS_WHEEL_COLORS):
        med, lo, hi = [], [], []
        for iteration in range(median_len):
            vals = [run[iteration] for run in series[key] if iteration in run]
            med.append(np.median(vals))
            lo.append(np.quantile(vals, 0.25))
            hi.append(np.quantile(vals, 0.75))
        ax.fill_between(x, lo, hi, color=color, alpha=0.25, linewidth=0)
        ax.plot(x, med, color=color, linewidth=1.3)
        ax.axvline(10.5, color="0.45", linewidth=0.7, linestyle="--")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized entropy")
    fig.tight_layout()
    fig.savefig(FIG_ENTROPY)
    print(f"Wrote {FIG_ENTROPY}")


def k_alignment(finals: list[dict], chars) -> dict:
    by_instance: dict[str, list[dict]] = {}
    for row in finals:
        by_instance.setdefault(row["instance"], []).append(row)
    records = []
    for instance, runs in by_instance.items():
        char = chars.loc[chars.instance == instance].iloc[0]
        arms: dict[str, list[float]] = {}
        for run in runs:
            for arm, prob in run["levels"]["level_1_k"]["prob"].items():
                arms.setdefault(arm, []).append(prob)
        mean_prob = {arm: float(np.mean(vals)) for arm, vals in arms.items()}
        top = max(mean_prob, key=mean_prob.get)
        seed_n = _seed_count(str(char.cust))
        records.append(
            {
                "instance": instance,
                "cust": str(char.cust),
                "group": str(char.cust_grp),
                "seed_n": seed_n,
                "n": int(char.n),
                "probs": mean_prob,
                "top": int(float(top)),
                "iterations": float(np.mean([run["iterations"] for run in runs])),
            }
        )
    clustered = [row for row in records if row["seed_n"] is not None]
    exact = multiple = neither = 0
    chance = []
    for row in clustered:
        c = row["seed_n"]
        domain = [int(float(arm)) for arm in row["probs"]]
        if row["top"] == c:
            exact += 1
        elif c > 0 and row["top"] % c == 0:
            multiple += 1
        else:
            neither += 1
        chance.append(1.0 / len(domain) if c in domain else 0.0)
    summary = {
        "n_clustered": len(clustered),
        "top_equals_seed_count": exact,
        "top_other_multiple": multiple,
        "top_neither": neither,
        "uniform_hit_rate": float(np.mean(chance)) if chance else None,
    }
    return {"records": records, "summary": summary}


def plot_k(records: list[dict]) -> None:
    clustered = [row for row in records if row["seed_n"] is not None]
    random_rows = [row for row in records if row["seed_n"] is None]
    clustered.sort(key=lambda row: (row["seed_n"], row["group"] != "C", row["instance"]))
    random_rows.sort(key=lambda row: row["instance"])
    ordered = clustered + random_rows
    arms = sorted({int(float(arm)) for row in ordered for arm in row["probs"]}, key=lambda v: v)
    grid = np.full((len(ordered), len(arms)), np.nan)
    marks = []
    for i, row in enumerate(ordered):
        for arm, prob in row["probs"].items():
            grid[i, arms.index(int(float(arm)))] = prob
        if row["seed_n"] is not None and row["seed_n"] in arms:
            marks.append((i, arms.index(row["seed_n"])))
    fig_h = max(280, 4.2 * len(ordered))
    fig, ax = plt.subplots(figsize=fig_size(0.98, fig_h))
    cmap = plt.colormaps["Blues"].copy()
    cmap.set_bad("#b0b0b0")
    vmax = float(np.nanmax(grid))
    image = ax.imshow(
        grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest"
    )
    if marks:
        ax.scatter(
            [col for _, col in marks],
            [row for row, _ in marks],
            s=16,
            facecolors="none",
            edgecolors="black",
            linewidths=0.4,
        )
    if clustered:
        ax.axhline(len(clustered) - 0.5, color="black", linewidth=0.6)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([str(arm) for arm in arms])
    ax.set_xlabel("Subcluster count $k$")
    ax.set_yticks([])
    ax.set_ylabel("Instances, ordered by seed-point count")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Mean wheel share")
    fig.tight_layout()
    fig.savefig(FIG_K)
    print(f"Wrote {FIG_K}")


def seed_agreement(finals: list[dict]) -> dict:
    by_instance: dict[str, list[dict]] = {}
    for row in finals:
        by_instance.setdefault(row["instance"], []).append(row)
    per_level: dict[str, list[dict]] = {key: [] for key in LEVEL_KEYS}
    for instance, runs in by_instance.items():
        if len(runs) != 3:
            raise SystemExit(f"{instance} has {len(runs)} final weight files")
        iters = float(np.mean([run["iterations"] for run in runs]))
        for key in LEVEL_KEYS:
            probs = [run["levels"][key]["prob"] for run in runs]
            dist = float(np.mean([tv(probs[i], probs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]))
            per_level[key].append({"instance": instance, "iterations": iters, "tv": dist})
    summary = {}
    for key, rows in per_level.items():
        tvs = np.array([row["tv"] for row in rows])
        iters = np.array([row["iterations"] for row in rows])
        summary[key] = {
            "mean_tv": float(tvs.mean()),
            "mean_tv_over_60": float(tvs[iters > 60].mean()) if np.any(iters > 60) else None,
            "n_over_60": int(np.sum(iters > 60)),
            "mean_tv_under_40": float(tvs[iters < 40].mean()) if np.any(iters < 40) else None,
            "n_under_40": int(np.sum(iters < 40)),
        }
    return {"rows": per_level, "summary": summary}


def plot_agreement(agreement: dict) -> None:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(2, 3, figsize=fig_size(1.0, 320), sharex=True, sharey=True)
    for ax, (key, label) in zip(axes.ravel(), LEVELS):
        rows = agreement["rows"][key]
        x = np.array([row["iterations"] for row in rows])
        y = np.array([row["tv"] for row in rows])
        ax.scatter(x, y, s=12, color="#a6304c", alpha=0.75, linewidths=0)
        if len(x) > 5:
            smooth = lowess(y, x, frac=0.4, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], color="black", linewidth=1.1)
        ax.set_title(label)
        ax.grid(True, linestyle=":", color="gray", alpha=0.6)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Completed iterations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean pairwise TV")
    fig.tight_layout()
    fig.savefig(FIG_SEEDS)
    print(f"Wrote {FIG_SEEDS}")


def budget_cut(finals: list[dict], chars) -> dict:
    n_of = dict(zip(chars.instance, chars.n))
    summary = {}
    points: dict[str, list[dict]] = {key: [] for key in LEVEL_KEYS}
    for row in finals:
        for key in LEVEL_KEYS:
            share = max(row["levels"][key]["prob"].values())
            points[key].append(
                {
                    "iterations": row["iterations"],
                    "share": share,
                    "n": int(n_of[row["instance"]]),
                }
            )
    for key, rows in points.items():
        candidates = []
        for threshold in sorted({row["iterations"] for row in rows}):
            subset = [row for row in rows if row["iterations"] >= threshold]
            if len(subset) < 10:
                continue
            frac = np.mean([row["share"] > 0.5 for row in subset])
            if frac >= 0.80:
                candidates.append(threshold)
                break
        cut = candidates[0] if candidates else None
        if cut is None:
            summary[key] = {"cut_iterations": None, "median_n": None, "n_runs": 0}
        else:
            kept = [row for row in rows if row["iterations"] >= cut]
            summary[key] = {
                "cut_iterations": int(cut),
                "median_n": float(np.median([row["n"] for row in kept])),
                "n_runs": len(kept),
                "fraction": float(np.mean([row["share"] > 0.5 for row in kept])),
            }
    return {"points": points, "summary": summary}


def plot_budget(budget: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=fig_size(1.0, 320), sharex=True, sharey=True)
    for ax, (key, label) in zip(axes.ravel(), LEVELS):
        rows = budget["points"][key]
        ax.scatter(
            [row["iterations"] for row in rows],
            [row["share"] for row in rows],
            s=10,
            color="#a6304c",
            alpha=0.55,
            linewidths=0,
        )
        ax.axhline(0.5, color="0.4", linewidth=0.7, linestyle="--")
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Completed iterations")
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-arm share")
    fig.tight_layout()
    fig.savefig(FIG_BUDGET)
    print(f"Wrote {FIG_BUDGET}")


def main() -> None:
    apply()
    chars = load()
    finals = load_finals()
    series = load_entropy()
    ent_summary, median_len = entropy_summary(series)
    plot_entropy(series, median_len)
    aligned = k_alignment(finals, chars)
    plot_k(aligned["records"])
    agreement = seed_agreement(finals)
    plot_agreement(agreement)
    budget = budget_cut(finals, chars)
    plot_budget(budget)
    # Drop per-run coordinates from the JSON; the figures already carry them.
    payload = {
        "entropy": ent_summary,
        "k_alignment": aligned["summary"],
        "seed_agreement": agreement["summary"],
        "top_arm": budget["summary"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
