#!/usr/bin/env python3
"""Do HAOS arms differ in the cost of the candidate they actually return?

One row is one draw that produced a BG-AILS candidate. The opening
decompose-route solution is the baseline and is not a row. Iteration 0's
BG-AILS call is a row, scored against that opening cost. Launches skipped at
the wall-clock cap are omitted: they have no BG-AILS candidate.

Quality of the draw is 100 * (C_t - c_BG) / C_t, signed. C_t is the incumbent
facing the decompose-route candidate (the opening cost at iteration 0). A
second outcome is whether that draw installed a new incumbent. BG-AILS's
repair of its own decompose-route input is reported on its own and is not an
arm score.

Position is a linear covariate: within each instance, quality (and the
incumbent indicator) is regressed on arm indicators and the iteration index,
and the arm's fitted value at the instance's mean iteration is the adjusted
mean. Friedman tests use those means. Thirds of each run are a robustness
check, not the control.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data/results/lagrange_benchmark"
NOTES = ROOT / "notes/haos_arm_quality.txt"
FIG_DIR = ROOT / "figures"
FIG_PATH = FIG_DIR / "haos_arm_quality.tex"

N_BOOT = 2000
BOOT_SEED = 20260923
SEED_SD_PP = 0.037
NOISE = SEED_SD_PP

PARADIGM = {"vb": "vertex", "rb": "route", "vertex": "vertex", "route": "route"}
K_TEST = (2, 3, 4, 6, 8, 10)
LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
WHEELS = (
    ("k", "k", K_TEST),
    ("lambda", r"$\lambda_q$", LAMBDAS),
    ("paradigm", "paradigm", ("vertex", "route")),
    (
        "vertex",
        "vertex method",
        ("kmeans", "agglomerative_avg", "agglomerative_complete", "kmedoids", "fcm"),
    ),
    (
        "route",
        "route method",
        ("kmeans", "agglomerative_avg", "agglomerative_complete"),
    ),
    ("solver", "subsolver", ("pyvrp", "filo", "filo2", "ails2")),
)

# Tripwires from the stage-1 audit. A drift in the exclusion rules should fail
# here rather than silently change the test.
EXPECTED_RUNS = 300
EXPECTED_USABLE = 22173
EXPECTED_WARMUP = 3000
EXPECTED_SKIPPED = 630


def canon_lambda(value: float) -> float:
    for arm in LAMBDAS:
        if abs(float(value) - arm) <= 1e-6:
            return arm
    raise SystemExit(f"lambda_demand {value!r} is outside the wheel")


def arm_label(wheel: str, arm: object) -> str:
    if wheel == "k":
        return str(int(arm))
    if wheel == "lambda":
        number = float(arm)
        if abs(number - round(number)) <= 1e-9:
            return str(int(round(number)))
        return f"{number:.1f}"
    return {
        "agglomerative_avg": "agg-avg",
        "agglomerative_complete": "agg-complete",
    }.get(str(arm), str(arm))


def full_label(wheel: str, arm: object) -> str:
    if wheel in ("k", "lambda"):
        return arm_label(wheel, arm)
    return str(arm)


def parse_runs() -> dict[str, np.ndarray]:
    paths = sorted(LOG_ROOT.glob("*/seed*/**/run.jsonl"))
    if len(paths) != EXPECTED_RUNS:
        raise SystemExit(f"expected {EXPECTED_RUNS} run logs, found {len(paths)}")

    rows: list[tuple] = []
    skipped = 0
    domains: dict[str, tuple] = {}
    before_mismatch = 0
    credit_mismatch = 0
    adopt_mismatch = 0

    for path in paths:
        instance = path.parents[2].name
        seed = path.parents[1].name
        draws: dict[int, dict] = {}
        drs: dict[int, tuple] = {}
        bgs: dict[int, dict] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if (
                    '"haos_roll"' not in line
                    and '"dr_apply"' not in line
                    and '"bg_apply"' not in line
                ):
                    continue
                event = json.loads(line)
                kind = event.get("type")
                if kind == "haos_roll" and "levels" not in event:
                    iteration = int(event["iteration"])
                    draws[iteration] = {
                        "k": int(event["k"]),
                        "lambda": canon_lambda(event["lambda_demand"]),
                        "paradigm": PARADIGM[event["paradigm"]],
                        "method": event["method"],
                        "solver": event["solver"],
                    }
                elif kind == "haos_roll" and instance not in domains:
                    domains[instance] = tuple(event["levels"]["level_1_k"]["values"])
                elif kind == "dr_apply":
                    iteration = int(event["iteration"])
                    incumbent = event["incumbent_cost"]
                    drs[iteration] = (
                        None if incumbent is None else float(incumbent),
                        float(event["dr_cost"]),
                        bool(event["adopted"]),
                    )
                elif kind == "bg_apply":
                    if event.get("bg_cost") is None:
                        raise SystemExit(f"{path}: bg_apply without a cost")
                    launch = int(event["bg_ails_launch_iteration"])
                    credited = event.get("haos_credited_iteration")
                    if credited is not None and int(credited) != launch:
                        credit_mismatch += 1
                    bgs[launch] = {
                        "cost": float(event["bg_cost"]),
                        "before": float(event["bg_cost_before"]),
                        "adopted": bool(event["adopted"]),
                    }

        if not draws or set(draws) != set(range(max(draws) + 1)):
            raise SystemExit(f"{path}: iteration index is not 0..n-1")
        n = max(draws) + 1
        for iteration, theta in draws.items():
            if iteration not in drs:
                raise SystemExit(f"{path}: iteration {iteration} has no dr_apply")
            if iteration not in bgs:
                skipped += 1
                continue
            incumbent, dr_cost, dr_adopted = drs[iteration]
            bg = bgs[iteration]
            if abs(bg["before"] - dr_cost) > 1e-4:
                before_mismatch += 1
            if iteration == 0:
                if incumbent is not None:
                    raise SystemExit(f"{path}: iteration 0 has an incumbent")
                baseline = dr_cost
                dr_new = False
            else:
                if incumbent is None or incumbent <= 0:
                    raise SystemExit(f"{path}: iteration {iteration} has no incumbent")
                baseline = incumbent
                cost_says_new = dr_cost < incumbent
                if cost_says_new != dr_adopted:
                    adopt_mismatch += 1
                dr_new = dr_adopted
            if baseline <= 0 or dr_cost <= 0:
                raise SystemExit(f"{path}: non-positive cost at iteration {iteration}")
            quality = 100.0 * (baseline - bg["cost"]) / baseline
            repair = (dr_cost - bg["cost"]) / dr_cost
            bg_new = bg["adopted"]
            if iteration < n / 3.0:
                third = 0
            elif iteration < 2.0 * n / 3.0:
                third = 1
            else:
                third = 2
            rows.append(
                (
                    instance,
                    seed,
                    iteration,
                    third,
                    theta["k"],
                    theta["lambda"],
                    theta["paradigm"],
                    theta["method"],
                    theta["solver"],
                    quality,
                    int(dr_new or bg_new),
                    int(dr_new),
                    int(bg_new),
                    repair,
                )
            )

    if before_mismatch or credit_mismatch or adopt_mismatch:
        raise SystemExit(
            "log fields disagree: "
            f"bg_cost_before {before_mismatch}, credit {credit_mismatch}, "
            f"dr adopted {adopt_mismatch}"
        )
    if skipped != EXPECTED_SKIPPED or len(rows) != EXPECTED_USABLE:
        raise SystemExit(f"expected {EXPECTED_USABLE} rows and {EXPECTED_SKIPPED} skips, got {len(rows)} and {skipped}")
    warmup = sum(1 for row in rows if row[2] <= 9)
    if warmup != EXPECTED_WARMUP:
        raise SystemExit(f"expected {EXPECTED_WARMUP} warm-up rows, got {warmup}")

    instance = np.array([row[0] for row in rows])
    data = {
        "instance": instance,
        "seed": np.array([row[1] for row in rows]),
        "iteration": np.array([row[2] for row in rows], dtype=int),
        "third": np.array([row[3] for row in rows], dtype=int),
        "k": np.array([row[4] for row in rows], dtype=int),
        "lambda": np.array([row[5] for row in rows], dtype=float),
        "paradigm": np.array([row[6] for row in rows]),
        "method": np.array([row[7] for row in rows]),
        "solver": np.array([row[8] for row in rows]),
        "quality": np.array([row[9] for row in rows], dtype=float),
        "new_inc": np.array([row[10] for row in rows], dtype=float),
        "dr_new": np.array([row[11] for row in rows], dtype=int),
        "bg_new": np.array([row[12] for row in rows], dtype=int),
        "repair": np.array([row[13] for row in rows], dtype=float),
        "domains": domains,
        "skipped": skipped,
    }
    return data


def wheel_arm(data: dict[str, np.ndarray], wheel: str) -> np.ndarray:
    if wheel == "k":
        return data["k"]
    if wheel == "lambda":
        return data["lambda"]
    if wheel == "paradigm":
        return data["paradigm"]
    if wheel == "solver":
        return data["solver"]
    if wheel == "vertex":
        arm = np.array(data["method"], copy=True)
        arm[data["paradigm"] != "vertex"] = None
        return arm
    if wheel == "route":
        arm = np.array(data["method"], copy=True)
        arm[data["paradigm"] != "route"] = None
        return arm
    raise KeyError(wheel)


def instance_ids(data: dict[str, np.ndarray], mask: np.ndarray) -> list[str]:
    return sorted(set(data["instance"][mask].tolist()))


def fit_adjusted(y: np.ndarray, arm: np.ndarray, iteration: np.ndarray, levels: tuple) -> tuple[np.ndarray, float]:
    """Arm means at the group's mean iteration, plus the common slope.

    Columns are arm indicators and the centered iteration index, so each arm
    coefficient is the fitted value at the mean index. Rank must be full: the
    slope is the within-arm association, and it is unidentified only when
    iteration is constant inside every arm.
    """
    n = y.size
    k = len(levels)
    design = np.zeros((n, k + 1))
    index = {level: i for i, level in enumerate(levels)}
    for row, value in enumerate(arm.tolist()):
        design[row, index[value]] = 1.0
    centered = iteration.astype(float) - float(np.mean(iteration))
    design[:, -1] = centered
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < k + 1:
        raise SystemExit("iteration slope is unidentified within an instance")
    return coef[:k].astype(float), float(coef[-1])


def arm_matrix(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    wheel: str,
    levels: tuple,
    outcome: str,
    *,
    adjust: bool,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return instance ids, an (instance, arm) matrix, and the within-instance slopes.

    Instances that never drew a tested arm are omitted. With adjust=False the
    cell is the raw mean. The primary analysis does not omit anyone: every
    tested arm occurs in every instance.
    """
    arms = wheel_arm(data, wheel)
    y_all = data[outcome]
    iteration = data["iteration"]
    inst = data["instance"]
    kept: list[str] = []
    means: list[np.ndarray] = []
    slopes: list[float] = []
    level_set = set(levels)
    for instance in instance_ids(data, mask):
        member = (inst == instance) & mask
        arm_ok = np.fromiter((value in level_set for value in arms), dtype=bool, count=arms.size)
        take = member & arm_ok
        if not np.any(take):
            continue
        present = set(arms[take].tolist())
        if present != level_set:
            continue
        y = y_all[take]
        arm = arms[take]
        t = iteration[take]
        if adjust:
            adjusted, slope = fit_adjusted(y, arm, t, levels)
            slopes.append(slope)
        else:
            adjusted = np.array([float(np.mean(y[arm == level])) for level in levels])
            slopes.append(float("nan"))
        kept.append(instance)
        means.append(adjusted)
    if not means:
        raise SystemExit(f"no complete instances for {wheel} {outcome}")
    return kept, np.vstack(means), np.array(slopes)


def holm(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=lambda i: pvalues[i])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def summarize_matrix(matrix: np.ndarray, levels: tuple, rng: np.random.Generator) -> dict:
    point = matrix.mean(axis=0)
    best = int(np.argmax(point))
    worst = int(np.argmin(point))
    spread = float(point[best] - point[worst])
    n, k = matrix.shape
    boot_spread = np.empty(N_BOOT)
    boot_mean = np.empty((N_BOOT, k))
    for draw in range(N_BOOT):
        sample = matrix[rng.integers(0, n, n)].mean(axis=0)
        boot_spread[draw] = sample.max() - sample.min()
        boot_mean[draw] = sample
    spread_lo, spread_hi = np.percentile(boot_spread, [2.5, 97.5])
    mean_lo = np.percentile(boot_mean, 2.5, axis=0)
    mean_hi = np.percentile(boot_mean, 97.5, axis=0)
    if k < 2:
        stat, pvalue = float("nan"), float("nan")
        test = "none"
    elif k == 2:
        # Friedman needs three or more treatments. Two arms are a paired
        # difference, which the signed-rank test evaluates.
        diff = matrix[:, 0] - matrix[:, 1]
        if np.all(diff == 0):
            stat, pvalue = 0.0, 1.0
        else:
            stat, pvalue = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        test = "wilcoxon"
    else:
        stat, pvalue = friedmanchisquare(*[matrix[:, j] for j in range(k)])
        test = "friedman"
    return {
        "levels": levels,
        "mean": point,
        "lo": mean_lo,
        "hi": mean_hi,
        "best": levels[best],
        "worst": levels[worst],
        "spread": spread,
        "spread_lo": float(spread_lo),
        "spread_hi": float(spread_hi),
        "stat": float(stat),
        "test": test,
        "p": float(pvalue),
        "n": n,
        "draws_note": None,
    }


def analyze_block(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    outcome: str,
    *,
    adjust: bool,
    rng: np.random.Generator,
) -> list[dict]:
    results = []
    for wheel, _pretty, levels in WHEELS:
        _ids, matrix, slopes = arm_matrix(data, mask, wheel, levels, outcome, adjust=adjust)
        summary = summarize_matrix(matrix, levels, rng)
        summary["wheel"] = wheel
        summary["slope_median"] = float(np.nanmedian(slopes)) if adjust else float("nan")
        summary["slope_q25"] = float(np.nanpercentile(slopes, 25)) if adjust else float("nan")
        summary["slope_q75"] = float(np.nanpercentile(slopes, 75)) if adjust else float("nan")
        results.append(summary)
    pvalues = [row["p"] for row in results]
    for row, corrected in zip(results, holm(pvalues)):
        row["holm"] = corrected
    return results


def draw_counts(data: dict[str, np.ndarray], mask: np.ndarray, wheel: str, levels: tuple) -> np.ndarray:
    """Draws per tested arm per instance. Missing arms contribute a zero."""
    arms = wheel_arm(data, wheel)
    inst = data["instance"]
    level_set = set(levels)
    counts = []
    for instance in instance_ids(data, np.ones(len(inst), dtype=bool)):
        take = mask & (inst == instance)
        present = arms[take]
        for level in levels:
            if level not in level_set:
                continue
            counts.append(int(np.sum(present == level)))
    return np.array(counts, dtype=int)


def f4(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def f3(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def fp(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def noise_phrase(row: dict) -> str:
    point = row["spread"]
    lo = row["spread_lo"]
    hi = row["spread_hi"]
    if hi < NOISE:
        where = "entirely below"
    elif lo > NOISE:
        where = "entirely above"
    else:
        where = "crossing"
    relation = "below" if point < NOISE else "above"
    return f"point {relation} {NOISE:.3f}; CI {where} {NOISE:.3f}"


def write_result_block(
    lines: list[str], title: str, rows: list[dict], unit: str, means_label: str
) -> None:
    add = lines.append
    add(title)
    add(
        f"  {'wheel':<16} {'best':<22} {'worst':<22} {'spread':>8} {'95% CI':>24} "
        f"{'stat':>8} {'p':>9} {'holm':>9} {'n':>4}"
    )
    for row in rows:
        add(
            f"  {row['wheel']:<16} {full_label(row['wheel'], row['best']):<22} "
            f"{full_label(row['wheel'], row['worst']):<22} {f4(row['spread']):>8} "
            f"[{f4(row['spread_lo'])}, {f4(row['spread_hi'])}] "
            f"{f3(row['stat']):>8} {fp(row['p']):>9} {fp(row['holm']):>9} {row['n']:4d}"
        )
    add(f"  Spread unit: {unit}. Holm is across the six wheels.")
    if unit.startswith("pp"):
        for row in rows:
            add(f"  {row['wheel']:<16} {noise_phrase(row)}")
    add(f"  {means_label}")
    for row in rows:
        add(f"  {row['wheel']}")
        if np.isfinite(row["slope_median"]):
            add(
                f"    within-instance slope per iteration: median {f4(row['slope_median'])}, "
                f"IQR [{f4(row['slope_q25'])}, {f4(row['slope_q75'])}]"
            )
        for level, mean, lo, hi in zip(row["levels"], row["mean"], row["lo"], row["hi"]):
            add(
                f"    {full_label(row['wheel'], level):<22} {f4(float(mean)):>8}  "
                f"[{f4(float(lo))}, {f4(float(hi))}]"
            )
    add("")


def schedule_share_lines(data: dict[str, np.ndarray]) -> list[str]:
    lines = []
    for iteration in (0, 9, 30, 70):
        present = data["iteration"] == iteration
        bits = []
        for seed in sorted(set(data["seed"][present].tolist())):
            values = data["solver"][present & (data["seed"] == seed)]
            _levels, counts = np.unique(values, return_counts=True)
            bits.append(f"{seed} {counts.max() / values.size:.2f} of {values.size}")
        lines.append(f"    iteration {iteration}: " + ", ".join(bits))
    return lines


def per_seed_quality_lines(data: dict[str, np.ndarray]) -> list[str]:
    lines = []
    for wheel, _pretty, levels in WHEELS:
        arm_all = wheel_arm(data, wheel)
        keep = np.isin(arm_all, list(levels))
        for seed in sorted(set(data["seed"].tolist())):
            mask = keep & (data["seed"] == seed)
            arm = arm_all[mask]
            missing = [level for level in levels if not np.any(arm == level)]
            if missing:
                lines.append(f"  {wheel:<16} {seed} missing {missing}")
                continue
            coef, _slope = fit_adjusted(
                data["quality"][mask], arm, data["iteration"][mask], levels
            )
            best = int(np.argmax(coef))
            worst = int(np.argmin(coef))
            lines.append(
                f"  {wheel:<16} {seed} best {full_label(wheel, levels[best]):<22} "
                f"{f4(float(coef[best])):>8}  worst {full_label(wheel, levels[worst]):<22} "
                f"{f4(float(coef[worst])):>8}  spread {f4(float(coef[best] - coef[worst]))}"
            )
    return lines


def write_notes(data: dict[str, np.ndarray], blocks: dict) -> None:
    quality = data["quality"]
    repair = data["repair"]
    new_inc = data["new_inc"]
    iteration = data["iteration"]
    lines: list[str] = []
    add = lines.append
    add("HAOS arm quality")
    add("Campaign: data/results/lagrange_benchmark")
    add(f"Runs: {EXPECTED_RUNS}. Instances: {len(set(data['instance'].tolist()))}.")
    add("")
    add("Definition")
    add("  One row is one draw whose BG-AILS candidate came back.")
    add("  C_t for iteration t >= 1 is dr_apply.incumbent_cost, the live best after")
    add("  the previous draw's BG-AILS result has been drained and before this draw's")
    add("  decompose-route candidate. At iteration 0 the opening decompose-route cost")
    add("  is C_0. That opening candidate is not a row. Iteration 0's BG-AILS call is.")
    add("  Quality = 100 * (C_t - bg_cost) / C_t, signed, in percentage points.")
    add("  A new incumbent is the logged adoption of this draw's decompose-route")
    add("  candidate (iterations >= 1 only) or of its BG-AILS candidate.")
    add("  Repair gain = (dr_cost - bg_cost) / dr_cost. It is not an arm score.")
    add(f"  Cap-skipped launches with no BG-AILS candidate: {data['skipped']}. They are out.")
    add(f"  Rows: {quality.size}. Warm-up rows (iterations 0-9): {int(np.sum(iteration <= 9))}.")
    add("  The k test uses arms 2, 3, 4, 6, 8, 10. Dropped: 12, which is missing on")
    add("  X-n1654-k11, and every extension arm.")
    seen = sorted(set(data["k"].tolist()))
    dropped = [k for k in seen if k not in K_TEST]
    add(f"  Extension and other dropped k values present in the rows: {dropped}.")
    add("  Position control: within each instance, outcome ~ arm + centered iteration.")
    add("  The reported arm value is the fitted mean at that instance's mean iteration,")
    add("  then averaged across the 100 instances. Draws are weighted equally.")
    add("  The three seeds of an instance are pooled before the fit.")
    add("  Friedman treats instances as blocks and arms as treatments.")
    add("  The paradigm wheel has two arms, so that row is a Wilcoxon signed-rank")
    add("  test of the paired instance difference instead of Friedman.")
    add(f"  Bootstrap: {N_BOOT} resamples of instances, seed {BOOT_SEED}. The best and")
    add("  worst arms are re-selected inside each resample. Intervals are percentiles.")
    add(f"  The noise floor is the Section 4.1.4 per-instance seed standard deviation, {NOISE:.3f} pp.")
    add("")
    add("What a row looks like")
    add(f"  Mean quality: {f4(float(np.mean(quality)))} pp. Median: {f4(float(np.median(quality)))} pp.")
    add(
        f"  Quality percentiles 10 / 90: {f4(float(np.percentile(quality, 10)))} / "
        f"{f4(float(np.percentile(quality, 90)))}."
    )
    add(
        f"  Draws whose BG-AILS cost beats C_t: {int(np.sum(quality > 0))} "
        f"({float(np.mean(quality > 0)):.3f})."
    )
    dr_only = int(np.sum((data["dr_new"] == 1) & (data["bg_new"] == 0)))
    bg_only = int(np.sum((data["dr_new"] == 0) & (data["bg_new"] == 1)))
    both = int(np.sum((data["dr_new"] == 1) & (data["bg_new"] == 1)))
    neither = int(np.sum(new_inc == 0))
    add(
        f"  New incumbent: {int(np.sum(new_inc))} draws ({float(np.mean(new_inc)):.3f}). "
        f"Decompose-route only {dr_only}, BG-AILS only {bg_only}, both {both}, neither {neither}."
    )
    add("  Iteration 0 is included in those counts. Its decompose-route adoption is not.")
    add("")
    add("Repair gain, not an arm score")
    add("  (dr_cost - bg_cost) / dr_cost, expressed below as a percent of the decompose-route cost.")
    repair_pp = 100.0 * repair
    add(
        f"  All rows: mean {f4(float(np.mean(repair_pp)))}, median {f4(float(np.median(repair_pp)))}, "
        f"10th {f4(float(np.percentile(repair_pp, 10)))}, 90th {f4(float(np.percentile(repair_pp, 90)))}."
    )
    add(f"  Share with bg_cost < dr_cost: {float(np.mean(repair > 0)):.3f}.")
    later = iteration >= 1
    add(
        f"  Iterations >= 1: mean {f4(float(np.mean(repair_pp[later])))}, "
        f"median {f4(float(np.median(repair_pp[later])))}."
    )
    opening = iteration == 0
    add(
        f"  Iteration 0 only: mean {f4(float(np.mean(repair_pp[opening])))}, "
        f"median {f4(float(np.median(repair_pp[opening])))}."
    )
    add("")
    write_result_block(
        lines,
        "Primary. Quality in pp, adjusted for the iteration index.",
        blocks["quality"],
        "pp of incumbent cost",
        "Adjusted arm means, averaged across instances. Brackets are bootstrap 95% intervals.",
    )
    add("Same quality fit, one seed at a time. Instances that share a seed are pooled.")
    add("  This is the cost response to that seed's draw sequence, not a second test.")
    add("  Route is the better paradigm on every seed, and agglomerative_avg is the")
    add("  best route method on every seed. The pooled best arm of k, lambda, the")
    add("  vertex method, and the subsolver is not the best arm on every seed.")
    for line in per_seed_quality_lines(data):
        add(line)
    add("")
    write_result_block(
        lines,
        "Primary. New-incumbent rate, adjusted for the iteration index.",
        blocks["rate"],
        "share of draws (adjusted means can fall outside [0, 1])",
        "Adjusted arm means, averaged across instances. Brackets are bootstrap 95% intervals.",
    )
    write_result_block(
        lines,
        "Unadjusted quality. Same rows, no iteration covariate. Late draws pull this.",
        blocks["quality_raw"],
        "pp of incumbent cost",
        "Unadjusted arm means, averaged across instances. Brackets are bootstrap 95% intervals.",
    )
    add("Robustness. Thirds of each run by raw iteration index, raw means, no covariate.")
    add("  An instance enters a wheel only when every tested arm was drawn in that third.")
    add("  Dropping those instances is exactly the selection the covariate avoids;")
    add("  this block is a check, not the result.")
    for third, rows in enumerate(blocks["thirds_quality"]):
        add(f"  Third {third + 1}, quality. Instances in the k test: {rows[0]['n']}.")
        for row in rows:
            add(
                f"    {row['wheel']:<16} n={row['n']:3d}  {full_label(row['wheel'], row['best'])} minus "
                f"{full_label(row['wheel'], row['worst'])}  spread {f4(row['spread'])}  "
                f"[{f4(row['spread_lo'])}, {f4(row['spread_hi'])}]  "
                f"p {fp(row['p'])}  holm {fp(row['holm'])}  {noise_phrase(row)}"
            )
    for third, rows in enumerate(blocks["thirds_rate"]):
        add(f"  Third {third + 1}, new-incumbent rate.")
        for row in rows:
            add(
                f"    {row['wheel']:<16} n={row['n']:3d}  spread {f4(row['spread'])}  "
                f"[{f4(row['spread_lo'])}, {f4(row['spread_hi'])}]  "
                f"p {fp(row['p'])}  holm {fp(row['holm'])}"
            )
    add("")
    add("Warm-up, iterations 0-9. Same adjustment, same two outcomes.")
    add("  Logged selection probabilities at iteration 0 are uniform on every run.")
    add("  The realized draw is not an independent sample per instance. Within a seed")
    add("  the arm at a given iteration is the same on nearly every instance; the")
    add("  three seeds differ. Modal within-seed shares for the subsolver:")
    for line in schedule_share_lines(data):
        add(line)
    add("  The vertex-method wheel is also thin. Both facts qualify the warm-up test.")
    warm = iteration <= 9
    for wheel, _pretty, levels in WHEELS:
        counts = draw_counts(data, warm, wheel, levels)
        under = int(np.sum(counts < 5))
        add(
            f"  {wheel:<16} cells {counts.size:4d}  min {int(counts.min()):2d}  "
            f"median {float(np.median(counts)):.1f}  max {int(counts.max()):2d}  "
            f"under 5: {under} ({under / counts.size:.3f})"
        )
    add("")
    write_result_block(
        lines,
        "Warm-up quality, adjusted for the iteration index.",
        blocks["warm_quality"],
        "pp of incumbent cost",
        "Adjusted arm means, averaged across instances. Brackets are bootstrap 95% intervals.",
    )
    write_result_block(
        lines,
        "Warm-up new-incumbent rate, adjusted for the iteration index.",
        blocks["warm_rate"],
        "share of draws",
        "Adjusted arm means, averaged across instances. Brackets are bootstrap 95% intervals.",
    )
    add("Figure")
    add(f"  {FIG_PATH.relative_to(ROOT)}")
    add("  Six panels. Points are the primary adjusted quality means.")
    add("  The band on each panel is that panel's unweighted arm mean ± 0.037 pp.")
    add("  It is the seed-noise floor drawn around the arms, not a band around zero:")
    add("  the BG-AILS candidate is typically a fraction of a percent worse than C_t,")
    add("  so a band at ±0.037 around zero would sit off the estimates.")
    add("")
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _standalone(body: str) -> str:
    return (
        "\\documentclass{standalone}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\usepgfplotslibrary{groupplots,fillbetween}\n"
        "\\begin{document}\n"
        + body
        + "\\end{document}\n"
    )


def write_figure(rows: list[dict]) -> Path:
    panels = []
    lows = []
    highs = []
    for row in rows:
        center = float(np.mean(row["mean"]))
        lows.append(min(float(np.min(row["lo"])), center - NOISE))
        highs.append(max(float(np.max(row["hi"])), center + NOISE))
    pad = 0.01
    ymin = min(lows) - pad
    ymax = max(highs) + pad
    for row in rows:
        labels = [arm_label(row["wheel"], level) for level in row["levels"]]
        symbolic = ",".join(labels)
        center = float(np.mean(row["mean"]))
        band_lo = " ".join(f"({label}, {center - NOISE:.4f})" for label in labels)
        band_hi = " ".join(f"({label}, {center + NOISE:.4f})" for label in labels)
        coords = []
        for label, mean, lo, hi in zip(labels, row["mean"], row["lo"], row["hi"]):
            up = max(0.0, float(hi) - float(mean))
            down = max(0.0, float(mean) - float(lo))
            coords.append(f"({label}, {float(mean):.4f}) += (0, {up:.4f}) -= (0, {down:.4f})")
        title = {
            "k": "k",
            "lambda": r"$\lambda_q$",
            "paradigm": "paradigm",
            "vertex": "vertex method",
            "route": "route method",
            "solver": "subsolver",
        }[row["wheel"]]
        panels.append(
            "\\nextgroupplot[title={" + title + "}, symbolic x coords={" + symbolic + "}, xtick=data]\n"
            "\\addplot[name path=lo, draw=none, forget plot] coordinates {" + band_lo + "};\n"
            "\\addplot[name path=hi, draw=none, forget plot] coordinates {" + band_hi + "};\n"
            "\\addplot[forget plot, fill=black!12, draw=none] fill between[of=lo and hi];\n"
            "\\addplot[only marks, mark=*, mark size=1.5pt, blue!60!black,\n"
            "  error bars/.cd, y dir=both, y explicit, error bar style={black, thin},\n"
            "  /pgfplots/.cd] coordinates {" + " ".join(coords) + "};\n"
        )
    body = (
        "\\begin{tikzpicture}\n"
        "\\begin{groupplot}[\n"
        "  group style={group size=3 by 2, horizontal sep=1.5cm, vertical sep=2.1cm},\n"
        "  width=5.6cm, height=4.6cm,\n"
        f"  ymin={ymin:.4f}, ymax={ymax:.4f},\n"
        "  xlabel={arm},\n"
        "  ylabel={adjusted BG gap (pp)},\n"
        "  tick label style={font=\\small},\n"
        "  xticklabel style={rotate=35, anchor=east, font=\\scriptsize},\n"
        "  label style={font=\\small},\n"
        "  title style={font=\\small},\n"
        "  ymajorgrids,\n"
        "]\n"
        + "".join(panels)
        + "\\end{groupplot}\n"
        "\\end{tikzpicture}\n"
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_PATH.write_text(_standalone(body), encoding="utf-8")
    return FIG_PATH


def compile_figure(path: Path) -> None:
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", path.name],
        cwd=path.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-50:])
        raise SystemExit(f"pdflatex failed on {path.name}\n{tail}")
    for suffix in (".aux", ".log"):
        side = path.with_suffix(suffix)
        if side.exists():
            side.unlink()


def main() -> None:
    data = parse_runs()
    rng = np.random.default_rng(BOOT_SEED)
    all_rows = np.ones(data["quality"].size, dtype=bool)
    blocks = {
        "quality": analyze_block(data, all_rows, "quality", adjust=True, rng=rng),
        "rate": analyze_block(data, all_rows, "new_inc", adjust=True, rng=rng),
        "quality_raw": analyze_block(data, all_rows, "quality", adjust=False, rng=rng),
        "thirds_quality": [],
        "thirds_rate": [],
        "warm_quality": analyze_block(data, data["iteration"] <= 9, "quality", adjust=True, rng=rng),
        "warm_rate": analyze_block(data, data["iteration"] <= 9, "new_inc", adjust=True, rng=rng),
    }
    for third in (0, 1, 2):
        mask = data["third"] == third
        blocks["thirds_quality"].append(analyze_block(data, mask, "quality", adjust=False, rng=rng))
        blocks["thirds_rate"].append(analyze_block(data, mask, "new_inc", adjust=False, rng=rng))
    for key in ("quality", "rate", "warm_quality", "warm_rate"):
        if any(row["n"] != 100 for row in blocks[key]):
            missing = [(row["wheel"], row["n"]) for row in blocks[key] if row["n"] != 100]
            raise SystemExit(f"{key} dropped instances: {missing}")
    write_notes(data, blocks)
    figure = write_figure(blocks["quality"])
    compile_figure(figure)
    print(f"wrote {NOTES}")
    print(f"wrote {figure} and {figure.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
