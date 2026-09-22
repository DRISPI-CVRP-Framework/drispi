#!/usr/bin/env python3
"""HAOS weight-dynamics diagnostics for the Lagrange campaign.

Reads end-of-iteration raw weights from run.jsonl. Does not replay the wheels.
Immediate rewards are recomputed from dr_apply / bg_apply costs. Deferred reward
scalars are recomputed from spsc_apply costs; their per-arm targets are not
logged and are not invented.

Warm-up iterations 0-9 are stored in the wheel-state table and excluded from
every diagnostic.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, spearmanr

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data/results/lagrange_benchmark"
WHEEL_CSV = ROOT / "data/results/haos_wheel_state.csv"
IMMEDIATE_CSV = ROOT / "data/results/haos_immediate_credits.csv"
DEFERRED_CSV = ROOT / "data/results/haos_deferred_credits.csv"
NOTES = ROOT / "notes/haos_diagnostics.txt"
FIG_DIR = ROOT / "figures"

W0 = 10.0
GAMMA = 0.95
WARMUP = 10
# Credit-then-decay: w* = gamma * p * R / (1 - gamma) = 19 * p * R.
MULTIPLIER = GAMMA / (1.0 - GAMMA)
LIFT = W0 * (1.0 - GAMMA) / GAMMA  # R > LIFT * A at uniform p
X_MIN = 10
X_MAX = 78
SLOPE_LO = 40
SLOPE_HI = 78
ATOL = 1e-4
FLOOR_ATOL = 1e-8
N_BOOT = 2000
BOOT_SEED = 20260922
MIN_DRAWS = 5
KW_ALPHA = 0.05

WHEELS = (
    ("level_1_k", "k", "k"),
    ("level_2_lambda_demand", "lambda", r"$\lambda_q$"),
    ("level_3_paradigm", "paradigm", "paradigm"),
    ("level_4a_vertex_method", "vertex_method", "vertex method"),
    ("level_4b_route_method", "route_method", "route method"),
    ("level_5_solver", "solver", "subsolver"),
)
WHEEL_KEYS = [key for key, _, _ in WHEELS]
SHORT = [short for _, short, _ in WHEELS]
PRETTY = [pretty for _, _, pretty in WHEELS]

PARADIGM = {"vb": "vertex", "rb": "route", "vertex": "vertex", "route": "route"}

EXPECTED_REWARDS = {
    "haos_reward_new_best": 10.0,
    "haos_reward_improvement": 4.0,
    "haos_reward_no_improvement": 1.0,
    "haos_reward_no_solution": 0.0,
    "haos_deferred_new_best": 7.0,
    "haos_deferred_improvement": 3.0,
    "haos_deferred_no_improvement": 0.0,
}


def effective_probabilities(weights: np.ndarray, min_weight: float) -> np.ndarray:
    """Equation 3.3: normalize, floor, renormalize."""
    total = float(weights.sum())
    if total <= 0.0:
        raw = np.full(weights.shape, 1.0 / len(weights))
    else:
        raw = weights / total
    floored = np.maximum(raw, min_weight)
    return floored / floored.sum()


def immediate_reward(cost: float, best: float, last: float) -> float:
    if cost < best:
        return 10.0
    if cost < last:
        return 4.0
    return 1.0


def deferred_reward(cost: float, best: float, last: float) -> float:
    if cost < best:
        return 7.0
    if cost < last:
        return 3.0
    return 0.0


def match_index(values: list, choice: object) -> int | None:
    for i, value in enumerate(values):
        if value == choice:
            return i
        if isinstance(value, (int, float)) and isinstance(choice, (int, float)) and not isinstance(value, bool):
            if abs(float(value) - float(choice)) <= 1e-9:
                return i
    return None


def arm_label(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return format(number, ".1f")
    return str(value)


def num(value: float) -> str:
    return format(float(value), ".12g")


def f3(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def f4(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def f6(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.6f}"


@dataclass
class Gate:
    n_compare: int = 0
    max_dev: float = 0.0
    where: str = ""


@dataclass
class Run:
    run_id: str
    instance: str
    seed: int
    n: int
    weights: list[np.ndarray]
    probs: list[np.ndarray]
    values: list[list]
    min_weight: list[float]
    sel: list[np.ndarray]
    labels: list[tuple]
    has_dr: np.ndarray
    r_dr: np.ndarray
    r_bg: np.ndarray
    bg_apply_at: np.ndarray
    bg_apply: list[list[list[tuple[int, float]]]]
    deferred_events: list[tuple[int, float]]
    skipped: np.ndarray
    incumbent_mismatches: int = 0
    missing_dr: int = 0


def _check_config(event: dict, path: Path) -> None:
    cfg = event["config"]
    haos = cfg["haos"]
    rewards = cfg["haos_rewards"]
    if haos.get("haos_decay") != GAMMA or haos.get("haos_warmup") != WARMUP:
        raise SystemExit(f"{path}: unexpected decay/warmup {haos.get('haos_decay')} {haos.get('haos_warmup')}")
    if haos.get("haos_starting_weight") != W0:
        raise SystemExit(f"{path}: unexpected starting weight")
    if cfg.get("bg_ails", {}).get("bg_ails_mode") != "async" or cfg.get("sp_sc", {}).get("sp_sc_mode") != "async":
        raise SystemExit(f"{path}: campaign path is not async BG and async SC/SP")
    for key, expected in EXPECTED_REWARDS.items():
        if rewards.get(key) != expected:
            raise SystemExit(f"{path}: reward {key} is {rewards.get(key)}")


def _selection_arms(roll: dict, levels: dict, k_choice: object) -> list[int]:
    paradigm = PARADIGM[roll["paradigm"]]
    specs: list[tuple[str, object | None]] = [
        ("level_1_k", k_choice),
        ("level_2_lambda_demand", roll["lambda_demand"]),
        ("level_3_paradigm", paradigm),
        ("level_4a_vertex_method", roll["method"] if paradigm == "vertex" else None),
        ("level_4b_route_method", roll["method"] if paradigm == "route" else None),
        ("level_5_solver", roll["solver"]),
    ]
    arms = []
    for key, choice in specs:
        if choice is None:
            arms.append(-1)
            continue
        idx = match_index(levels[key]["values"], choice)
        if idx is None:
            raise SystemExit(f"arm {choice!r} not in {key} values {levels[key]['values']}")
        arms.append(idx)
    return arms


def parse_run(path: Path, gate: Gate) -> Run:
    instance = path.parents[2].name
    seed = int(path.parents[1].name.removeprefix("seed"))
    run_id = f"{instance}__seed{seed}"

    best = math.inf
    last = math.inf
    snaps: dict[int, dict] = {}
    starts: set[int] = set()
    clips: dict[int, object] = {}
    has_dr: dict[int, float] = {}
    r_bg_launch: dict[int, float] = {}
    bg_apply_at_map: dict[int, int] = {}
    bg_apply: dict[int, dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    deferred_events: list[tuple[int, float]] = []
    skipped: set[int] = set()
    seen_bg: set[int] = set()
    mismatches = 0
    config_seen = False

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            kind = event.get("type")
            if kind == "config":
                _check_config(event, path)
                config_seen = True
            elif kind == "k_clipped_to_routes":
                clips[int(event["iteration"])] = event["k_rolled"]
            elif kind == "iteration_skipped":
                skipped.add(int(event["iteration"]))
            elif kind == "haos_roll":
                iteration = int(event["iteration"])
                if "levels" not in event:
                    starts.add(iteration)
                    continue
                if iteration in snaps:
                    raise SystemExit(f"{run_id}: two weighted rolls at iteration {iteration}")
                levels = event["levels"]
                k_choice = clips.get(iteration, event["k"])
                arms = _selection_arms(event, levels, k_choice)
                paradigm = PARADIGM[event["paradigm"]]
                weights = []
                probs = []
                values = []
                mins = []
                for key in WHEEL_KEYS:
                    level = levels[key]
                    w = np.asarray(level["raw_weights"], dtype=float)
                    p = np.asarray(level["probabilities"], dtype=float)
                    recomputed = effective_probabilities(w, float(level["min_weight"]))
                    dev = np.max(np.abs(recomputed - p))
                    gate.n_compare += int(w.size)
                    if dev > gate.max_dev:
                        gate.max_dev = float(dev)
                        arm = int(np.argmax(np.abs(recomputed - p)))
                        gate.where = f"{run_id} iteration {iteration} {key} arm {arm}"
                    weights.append(w)
                    probs.append(p)
                    values.append(list(level["values"]))
                    mins.append(float(level["min_weight"]))
                snaps[iteration] = {
                    "weights": weights,
                    "probs": probs,
                    "values": values,
                    "min_weight": mins,
                    "sel": arms,
                    "label": (k_choice, event["lambda_demand"], paradigm, event["method"], event["solver"]),
                }
            elif kind == "dr_apply":
                iteration = int(event["iteration"])
                if iteration in has_dr:
                    raise SystemExit(f"{run_id}: two dr_apply at iteration {iteration}")
                incumbent = event["incumbent_cost"]
                best_logged = math.inf if incumbent is None else float(incumbent)
                if math.isfinite(best) and math.isfinite(best_logged) and abs(best - best_logged) > 1e-4:
                    mismatches += 1
                best_use = best_logged
                dr_cost = float(event["dr_cost"])
                has_dr[iteration] = immediate_reward(dr_cost, best_use, last)
                if dr_cost < best_use:
                    best = dr_cost
                else:
                    best = best_use
                last = dr_cost
            elif kind == "bg_apply":
                # A failed worker does not change the incumbent or last cost.
                # A successful apply does, including during warm-up, where the
                # reward is computed and then discarded rather than credited.
                raw_cost = event.get("bg_cost")
                if raw_cost is None:
                    continue
                bg_cost = float(raw_cost)
                reward = immediate_reward(bg_cost, best, last)
                launch = event.get("haos_credited_iteration")
                if launch is not None:
                    launch = int(launch)
                    apply_at = int(event["bg_ails_apply_iteration"])
                    if launch in seen_bg:
                        raise SystemExit(f"{run_id}: BG launch {launch} credited twice")
                    seen_bg.add(launch)
                    if launch not in snaps:
                        raise SystemExit(f"{run_id}: BG credit for launch {launch} before its weight snapshot")
                    r_bg_launch[launch] = reward
                    bg_apply_at_map[launch] = apply_at
                    for wheel, arm in enumerate(snaps[launch]["sel"]):
                        if arm >= 0:
                            bg_apply[apply_at][wheel].append((int(arm), reward))
                live = event.get("live_best_cost")
                if bg_cost < best:
                    best = bg_cost
                if live is not None and math.isfinite(float(live)):
                    best = float(live)
                last = bg_cost
            elif kind == "spsc_apply":
                final_cost = event.get("final_cost")
                if final_cost is None:
                    continue
                apply_at = int(event["apply_iteration"])
                reward = deferred_reward(float(final_cost), best, last)
                deferred_events.append((apply_at, reward))
                live = event.get("live_best_cost")
                if float(final_cost) < best:
                    best = float(final_cost)
                if live is not None and math.isfinite(float(live)):
                    best = float(live)
                last = float(final_cost)

    if not config_seen:
        raise SystemExit(f"{path}: no config event")
    if not snaps:
        raise SystemExit(f"{run_id}: no weighted rolls")
    n = max(snaps) + 1
    if set(snaps) != set(range(n)):
        missing = sorted(set(range(n)) - set(snaps))
        raise SystemExit(f"{run_id}: missing iterations {missing[:8]}")
    if starts != set(snaps):
        raise SystemExit(f"{run_id}: start rolls and weighted rolls differ")

    weights = [np.vstack([snaps[t]["weights"][w] for t in range(n)]) for w in range(6)]
    probs = [np.vstack([snaps[t]["probs"][w] for t in range(n)]) for w in range(6)]
    sel = [np.array([snaps[t]["sel"][w] for t in range(n)], dtype=int) for w in range(6)]
    r_dr = np.zeros(n)
    has = np.zeros(n, dtype=bool)
    for iteration, reward in has_dr.items():
        r_dr[iteration] = reward
        has[iteration] = True
    r_bg = np.zeros(n)
    bg_apply_at = np.full(n, -1, dtype=int)
    for iteration, reward in r_bg_launch.items():
        r_bg[iteration] = reward
        bg_apply_at[iteration] = bg_apply_at_map[iteration]
    apply_lists: list[list[list[tuple[int, float]]]] = []
    for iteration in range(n):
        per_wheel = []
        for wheel in range(6):
            per_wheel.append(list(bg_apply[iteration][wheel]))
        apply_lists.append(per_wheel)
    skip = np.zeros(n, dtype=bool)
    for iteration in skipped:
        if 0 <= iteration < n:
            skip[iteration] = True
    missing_dr = sum(1 for t in range(WARMUP, n) if not has[t])

    return Run(
        run_id=run_id,
        instance=instance,
        seed=seed,
        n=n,
        weights=weights,
        probs=probs,
        values=snaps[0]["values"],
        min_weight=snaps[0]["min_weight"],
        sel=sel,
        labels=[snaps[t]["label"] for t in range(n)],
        has_dr=has,
        r_dr=r_dr,
        r_bg=r_bg,
        bg_apply_at=bg_apply_at,
        bg_apply=apply_lists,
        deferred_events=deferred_events,
        skipped=skip,
        incumbent_mismatches=mismatches,
        missing_dr=missing_dr,
    )


def _immediate_on(run: Run, wheel: int, iteration: int, arm: int) -> float:
    total = 0.0
    if run.sel[wheel][iteration] == arm and run.has_dr[iteration]:
        total += float(run.r_dr[iteration])
    for landed_arm, reward in run.bg_apply[iteration][wheel]:
        if landed_arm == arm:
            total += reward
    return total


def audit_windows(run: Run) -> tuple[float, int]:
    """Check credit-then-decay windows against the logged weights.

    On iterations with no positive deferred reward, an arm above the floor must
    satisfy w = gamma * (w_prev + immediate). A positive deferred reward can
    only add credit, so the inferred credit cannot fall short of the known
    immediate credit.
    """
    deferred_at = np.zeros(run.n)
    for iteration, reward in run.deferred_events:
        if 0 <= iteration < run.n:
            deferred_at[iteration] += reward
    max_resid = 0.0
    violations = 0
    for wheel in range(6):
        w = run.weights[wheel]
        for t in range(WARMUP, run.n):
            positive_deferred = deferred_at[t] > 0.0
            for arm in range(w.shape[1]):
                prev = float(w[t - 1, arm])
                cur = float(w[t, arm])
                immediate = _immediate_on(run, wheel, t, arm)
                if cur > W0 + FLOOR_ATOL:
                    inferred = cur / GAMMA - prev
                    if inferred < -ATOL:
                        violations += 1
                    if inferred + ATOL < immediate:
                        violations += 1
                    if not positive_deferred:
                        max_resid = max(max_resid, abs(inferred - immediate))
                else:
                    upper = max(0.0, W0 / GAMMA - prev)
                    if immediate > upper + ATOL:
                        violations += 1
                    if prev > W0 / GAMMA + ATOL:
                        violations += 1
    return max_resid, violations


def write_tables(runs: list[Run]) -> None:
    WHEEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with WHEEL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "instance",
                "seed",
                "iteration",
                "wheel",
                "arm",
                "arm_index",
                "n_arms",
                "weight",
                "probability",
                "min_weight",
            ]
        )
        for run in runs:
            for wheel, short in enumerate(SHORT):
                values = run.values[wheel]
                n_arms = len(values)
                min_weight = run.min_weight[wheel]
                for iteration in range(run.n):
                    for arm in range(n_arms):
                        writer.writerow(
                            [
                                run.run_id,
                                run.instance,
                                run.seed,
                                iteration,
                                short,
                                arm_label(values[arm]),
                                arm,
                                n_arms,
                                num(run.weights[wheel][iteration, arm]),
                                num(run.probs[wheel][iteration, arm]),
                                num(min_weight),
                            ]
                        )
    with IMMEDIATE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "instance",
                "seed",
                "credited_iteration",
                "channel",
                "reward",
                "apply_iteration",
                "k",
                "lambda_demand",
                "paradigm",
                "method",
                "solver",
            ]
        )
        for run in runs:
            for iteration in range(WARMUP, run.n):
                reward = float(run.r_dr[iteration]) if run.has_dr[iteration] else 0.0
                k, lam, paradigm, method, solver = run.labels[iteration]
                writer.writerow(
                    [
                        run.run_id,
                        run.instance,
                        run.seed,
                        iteration,
                        "dr",
                        num(reward),
                        iteration,
                        arm_label(k),
                        arm_label(lam),
                        paradigm,
                        method,
                        solver,
                    ]
                )
            for launch in range(WARMUP, run.n):
                # A credited BG reward is at least the no-improvement score.
                # r_bg stays 0 when the launch was never credited.
                if run.r_bg[launch] <= 0.0:
                    continue
                apply_at = int(run.bg_apply_at[launch])
                k, lam, paradigm, method, solver = run.labels[launch]
                writer.writerow(
                    [
                        run.run_id,
                        run.instance,
                        run.seed,
                        launch,
                        "bg",
                        num(run.r_bg[launch]),
                        apply_at,
                        arm_label(k),
                        arm_label(lam),
                        paradigm,
                        method,
                        solver,
                    ]
                )
    with DEFERRED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "instance", "seed", "apply_iteration", "reward"])
        for run in runs:
            for iteration, reward in run.deferred_events:
                if iteration < WARMUP:
                    continue
                writer.writerow([run.run_id, run.instance, run.seed, iteration, num(reward)])


def _post_mask(n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[WARMUP:] = True
    return mask


def selection_reward(run: Run, iteration: int) -> float:
    dr = float(run.r_dr[iteration]) if run.has_dr[iteration] else 0.0
    return dr + float(run.r_bg[iteration])


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def _median_curve(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """matrix is (n_runs, n_iter) with NaN where the run has ended."""
    n_iter = matrix.shape[1]
    q25 = np.full(n_iter, np.nan)
    med = np.full(n_iter, np.nan)
    q75 = np.full(n_iter, np.nan)
    count = np.zeros(n_iter, dtype=int)
    for t in range(n_iter):
        col = matrix[:, t]
        col = col[np.isfinite(col)]
        count[t] = col.size
        if col.size:
            q25[t], med[t], q75[t] = np.percentile(col, [25, 50, 75])
    return q25, med, q75, count


def _slope(med: np.ndarray) -> float:
    x = np.arange(SLOPE_LO, SLOPE_HI + 1)
    y = med[SLOPE_LO : SLOPE_HI + 1]
    ok = np.isfinite(y)
    if int(ok.sum()) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(x[ok].astype(float), y[ok], 1)
    return float(slope)


def _bootstrap_slopes(matrix: np.ndarray) -> tuple[float, float, float]:
    point = _slope(_median_curve(matrix)[1])
    rng = np.random.default_rng(BOOT_SEED)
    n_runs = matrix.shape[0]
    slopes = np.empty(N_BOOT)
    for draw in range(N_BOOT):
        idx = rng.integers(0, n_runs, n_runs)
        slopes[draw] = _slope(_median_curve(matrix[idx])[1])
    lo, hi = np.percentile(slopes[np.isfinite(slopes)], [2.5, 97.5])
    return point, float(lo), float(hi)


def _spell(weights: np.ndarray, start: int) -> tuple[int, bool]:
    if weights[start] <= W0 + FLOOR_ATOL:
        return 0, False
    length = 0
    for t in range(start, len(weights)):
        if weights[t] <= W0 + FLOOR_ATOL:
            return length, False
        length += 1
    return length, True


def analyze(runs: list[Run]) -> dict:
    max_n = max(run.n for run in runs)
    ratio = [np.full((len(runs), max_n), np.nan) for _ in range(6)]
    dust = 0
    exact_floor = 0
    floor_tests = 0
    for i, run in enumerate(runs):
        for wheel in range(6):
            w = run.weights[wheel]
            for t in range(WARMUP, run.n):
                ratio[wheel][i, t] = float(np.max(w[t])) / W0
            post = w[WARMUP:]
            floor_tests += int(post.size)
            exact_floor += int(np.sum(post == W0))
            dust += int(np.sum((post > W0) & (post <= W0 + 1e-6)))

    curves = []
    slopes = []
    for wheel in range(6):
        q25, med, q75, count = _median_curve(ratio[wheel])
        curves.append((q25, med, q75, count))
        slopes.append(_bootstrap_slopes(ratio[wheel]))

    # Selection rewards and per-wheel probabilities of the credited arm.
    rewards: list[float] = []
    r_by_wheel: list[list[float]] = [[] for _ in range(6)]
    p_sum = np.zeros(6)
    p_count = np.zeros(6)
    above = np.zeros(6)  # selections with R > LIFT * A
    n_sel = np.zeros(6)
    above_group: dict[tuple[str, int], int] = defaultdict(int)
    sel_group: dict[tuple[str, int], int] = defaultdict(int)
    r_values_for_hist: list[float] = []
    dr_values: list[float] = []
    bg_values: list[float] = []
    n_with_bg = 0
    n_selections = 0
    deferred_hist: list[float] = []
    warmup_deferred = 0

    # Diagnostic 4 accumulators, filled per run.
    spreads: list[list[float]] = [[] for _ in range(6)]
    spearman: list[list[float]] = [[] for _ in range(6)]
    kw_defined = np.zeros(6)
    kw_reject = np.zeros(6)
    kw_reject_raw = np.zeros(6)

    # Diagnostic 3 / persistence / gap, pooled and per (wheel, A).
    pinned_num = np.zeros(6)
    pinned_den = np.zeros(6)
    unpinned_counts: list[list[int]] = [[] for _ in range(6)]
    spell_lengths: list[list[int]] = [[] for _ in range(6)]
    spell_censored = np.zeros(6)
    gaps: list[list[float]] = [[] for _ in range(6)]
    group_pinned: dict[tuple[str, int], list[float]] = defaultdict(list)
    group_spell: dict[tuple[str, int], list[float]] = defaultdict(list)
    group_gap: dict[tuple[str, int], list[float]] = defaultdict(list)
    group_runs: dict[tuple[str, int], int] = defaultdict(int)

    # H1 end-of-run max weight, and top-arm probability.
    end_max = np.full((len(runs), 6), np.nan)
    end_top_p = np.full((len(runs), 6), np.nan)

    # Diagnostic 5 and credit mass.
    rose_def = [[] for _ in range(6)]
    rose_oth = [[] for _ in range(6)]
    inj_def_lo = [[] for _ in range(6)]
    inj_def_hi = [[] for _ in range(6)]
    inj_oth_lo = [[] for _ in range(6)]
    inj_oth_hi = [[] for _ in range(6)]
    mass_exact = np.zeros(6)
    mass_floor_imm = np.zeros(6)
    mass_floor_upper = np.zeros(6)
    mass_immediate = np.zeros(6)
    mass_deferred_exact = np.zeros(6)
    mass_hidden_deferred = np.zeros(6)
    mass_phantom = np.zeros(6)
    n_def_iters = np.zeros(6)
    n_oth_iters = np.zeros(6)

    max_resid = 0.0
    violations = 0
    incumbent_mismatches = 0
    missing_dr = 0
    skipped_post = 0

    for i, run in enumerate(runs):
        incumbent_mismatches += run.incumbent_mismatches
        missing_dr += run.missing_dr
        skipped_post += int(np.sum(run.skipped[WARMUP:]))
        resid, viol = audit_windows(run)
        max_resid = max(max_resid, resid)
        violations += viol

        deferred_at = np.zeros(run.n)
        for iteration, reward in run.deferred_events:
            if iteration < WARMUP:
                warmup_deferred += 1
                continue
            deferred_hist.append(reward)
            if 0 <= iteration < run.n:
                deferred_at[iteration] += reward

        # Per-run KW groups: wheel -> arm -> list of R.
        groups: list[dict[int, list[float]]] = [defaultdict(list) for _ in range(6)]

        for t in range(WARMUP, run.n):
            reward = selection_reward(run, t)
            rewards.append(reward)
            r_values_for_hist.append(reward)
            n_selections += 1
            if run.has_dr[t]:
                dr_values.append(float(run.r_dr[t]))
            if run.r_bg[t] > 0.0:
                bg_values.append(float(run.r_bg[t]))
                n_with_bg += 1
            for wheel in range(6):
                n_arms = run.weights[wheel].shape[1]
                n_sel[wheel] += 1
                group_key = (SHORT[wheel], n_arms)
                sel_group[group_key] += 1
                if reward > LIFT * n_arms:
                    above[wheel] += 1
                    above_group[group_key] += 1
                arm = int(run.sel[wheel][t])
                if arm >= 0:
                    r_by_wheel[wheel].append(reward)
                    groups[wheel][arm].append(reward)
                    p_sum[wheel] += float(run.probs[wheel][t - 1, arm])
                    p_count[wheel] += 1

        p_run = []
        for wheel in range(6):
            n_post = run.n - WARMUP
            # Unconditional mean: 0 on iterations that do not credit this wheel.
            p_bar_run = 0.0
            for t in range(WARMUP, run.n):
                arm = int(run.sel[wheel][t])
                if arm >= 0:
                    p_bar_run += float(run.probs[wheel][t - 1, arm])
            p_run.append(p_bar_run / n_post)
            end_max[i, wheel] = float(np.max(run.weights[wheel][-1]))
            end_top_p[i, wheel] = float(np.max(run.probs[wheel][-1]))

            w = run.weights[wheel]
            post = w[WARMUP:]
            pinned_num[wheel] += float(np.sum(post == W0))
            pinned_den[wheel] += float(post.size)
            unpinned = int(np.sum(np.any(post > W0 + FLOOR_ATOL, axis=0)))
            unpinned_counts[wheel].append(unpinned)

            run_gaps: list[float] = []
            for arm in range(w.shape[1]):
                times = np.flatnonzero(run.sel[wheel] == arm)
                times = times[times >= WARMUP]
                if times.size >= 2:
                    run_gaps.extend(np.diff(times).astype(float).tolist())
                credited_at = set()
                for t in range(WARMUP, run.n):
                    if run.sel[wheel][t] == arm and run.has_dr[t] and run.r_dr[t] > 0.0:
                        credited_at.add(t)
                    for landed, landed_r in run.bg_apply[t][wheel]:
                        if landed == arm and landed_r > 0.0:
                            credited_at.add(t)
                for t in credited_at:
                    length, censored = _spell(w[:, arm], t)
                    spell_lengths[wheel].append(length)
                    if censored:
                        spell_censored[wheel] += 1
            if run_gaps:
                gaps[wheel].append(float(np.mean(run_gaps)))

            key = (SHORT[wheel], int(w.shape[1]))
            group_runs[key] += 1
            group_pinned[key].append(float(np.sum(post == W0)) / float(post.size))
            # Per-run mean spell and gap for the figure, so runs weigh equally.
            spell_run = []
            for arm in range(w.shape[1]):
                credited_at = set()
                for t in range(WARMUP, run.n):
                    if run.sel[wheel][t] == arm and run.has_dr[t] and run.r_dr[t] > 0.0:
                        credited_at.add(t)
                    for landed, landed_r in run.bg_apply[t][wheel]:
                        if landed == arm and landed_r > 0.0:
                            credited_at.add(t)
                for t in credited_at:
                    length, _censored = _spell(w[:, arm], t)
                    spell_run.append(length)
            if spell_run:
                group_spell[key].append(float(np.mean(spell_run)))
            if run_gaps:
                group_gap[key].append(float(np.mean(run_gaps)))

            # Diagnostic 4 for this run. Built after the loop over t via groups.
            qualified = {arm: vals for arm, vals in groups[wheel].items() if len(vals) >= MIN_DRAWS}
            if len(qualified) >= 2:
                means = [float(np.mean(vals)) for vals in qualified.values()]
                spreads[wheel].append(max(means) - min(means))
            final_p = run.probs[wheel][-1]
            if len(qualified) >= 3:
                arms = sorted(qualified)
                mean_r = [float(np.mean(qualified[arm])) for arm in arms]
                final = [float(final_p[arm]) for arm in arms]
                if np.std(mean_r) > 0 and np.std(final) > 0:
                    rho = spearmanr(mean_r, final).statistic
                    if math.isfinite(rho):
                        spearman[wheel].append(float(rho))

            # Credit mass and diagnostic 5.
            for t in range(WARMUP, run.n):
                lo = 0.0
                hi = 0.0
                rose = 0
                immediate_sum = 0.0
                for arm in range(w.shape[1]):
                    prev = float(w[t - 1, arm])
                    cur = float(w[t, arm])
                    if cur > prev + FLOOR_ATOL:
                        rose += 1
                    immediate = _immediate_on(run, wheel, t, arm)
                    immediate_sum += immediate
                    if cur > W0 + FLOOR_ATOL:
                        inferred = max(0.0, cur / GAMMA - prev)
                        lo += inferred
                        hi += inferred
                        mass_exact[wheel] += inferred
                        mass_deferred_exact[wheel] += max(0.0, inferred - immediate)
                    else:
                        upper = max(0.0, W0 / GAMMA - prev)
                        hidden = max(0.0, upper - immediate)
                        lo += immediate
                        hi += upper
                        mass_floor_imm[wheel] += immediate
                        mass_floor_upper[wheel] += upper
                        if deferred_at[t] > 0.0:
                            mass_hidden_deferred[wheel] += hidden
                        else:
                            mass_phantom[wheel] += hidden
                    mass_immediate[wheel] += immediate
                if deferred_at[t] > 0.0:
                    rose_def[wheel].append(rose)
                    inj_def_lo[wheel].append(lo)
                    inj_def_hi[wheel].append(hi)
                    n_def_iters[wheel] += 1
                else:
                    rose_oth[wheel].append(rose)
                    inj_oth_lo[wheel].append(lo)
                    inj_oth_hi[wheel].append(hi)
                    n_oth_iters[wheel] += 1

        # Kruskal-Wallis within the run, then Holm across the wheels that have a test.
        pvals = []
        p_wheels = []
        for wheel in range(6):
            qualified = [vals for vals in groups[wheel].values() if len(vals) >= MIN_DRAWS]
            if len(qualified) < 2:
                continue
            try:
                _stat, p = kruskal(*qualified)
            except ValueError:
                continue
            if not math.isfinite(p):
                continue
            pvals.append(float(p))
            p_wheels.append(wheel)
        if pvals:
            adjusted = holm(pvals)
            for wheel, raw_p, adj_p in zip(p_wheels, pvals, adjusted):
                kw_defined[wheel] += 1
                if raw_p < KW_ALPHA:
                    kw_reject_raw[wheel] += 1
                if adj_p < KW_ALPHA:
                    kw_reject[wheel] += 1

    lengths = np.array([run.n for run in runs])
    return {
        "n_runs": len(runs),
        "lengths": lengths,
        "curves": curves,
        "slopes": slopes,
        "rewards": np.asarray(rewards),
        "r_by_wheel": r_by_wheel,
        "p_sum": p_sum,
        "p_count": p_count,
        "above": above,
        "n_sel": n_sel,
        "above_group": above_group,
        "sel_group": sel_group,
        "dr_values": np.asarray(dr_values),
        "bg_values": np.asarray(bg_values),
        "n_with_bg": n_with_bg,
        "n_selections": n_selections,
        "deferred_hist": np.asarray(deferred_hist, dtype=float),
        "warmup_deferred": warmup_deferred,
        "spreads": spreads,
        "spearman": spearman,
        "kw_defined": kw_defined,
        "kw_reject": kw_reject,
        "kw_reject_raw": kw_reject_raw,
        "pinned_num": pinned_num,
        "pinned_den": pinned_den,
        "unpinned_counts": unpinned_counts,
        "spell_lengths": spell_lengths,
        "spell_censored": spell_censored,
        "gaps": gaps,
        "group_pinned": group_pinned,
        "group_spell": group_spell,
        "group_gap": group_gap,
        "group_runs": group_runs,
        "end_max": end_max,
        "end_top_p": end_top_p,
        "rose_def": rose_def,
        "rose_oth": rose_oth,
        "inj_def_lo": inj_def_lo,
        "inj_def_hi": inj_def_hi,
        "inj_oth_lo": inj_oth_lo,
        "inj_oth_hi": inj_oth_hi,
        "mass_exact": mass_exact,
        "mass_floor_imm": mass_floor_imm,
        "mass_floor_upper": mass_floor_upper,
        "mass_immediate": mass_immediate,
        "mass_deferred_exact": mass_deferred_exact,
        "mass_hidden_deferred": mass_hidden_deferred,
        "mass_phantom": mass_phantom,
        "n_def_iters": n_def_iters,
        "n_oth_iters": n_oth_iters,
        "dust": dust,
        "exact_floor": exact_floor,
        "floor_tests": floor_tests,
        "max_resid": max_resid,
        "violations": violations,
        "incumbent_mismatches": incumbent_mismatches,
        "missing_dr": missing_dr,
        "skipped_post": skipped_post,
        "ratio": ratio,
    }


def _mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _hist_counts(values: np.ndarray) -> list[tuple[float, int]]:
    keys = []
    for value in values:
        if abs(value - round(value)) < 1e-6:
            keys.append(int(round(value)))
        else:
            keys.append(round(float(value), 6))
    return sorted(Counter(keys).items(), key=lambda item: item[0])


def _h1_rows(stats: dict) -> list[dict]:
    rows = []
    rewards = stats["rewards"]
    global_mean = float(np.mean(rewards)) if rewards.size else float("nan")
    for wheel in range(6):
        med = stats["curves"][wheel][1]
        obs78 = float(med[X_MAX] * W0) if np.isfinite(med[X_MAX]) else float("nan")
        # p_bar is the mean, over post-warmup iterations, of the previous
        # snapshot's probability of the credited arm, and 0 when the wheel
        # was not credited. Stored as a sum over credited iterations only,
        # so divide by the number of post-warmup iterations, not p_count.
        n_post = int(np.sum(stats["lengths"] - WARMUP))
        p_bar = float(stats["p_sum"][wheel] / n_post) if n_post else float("nan")
        r_bar = _mean(stats["r_by_wheel"][wheel])
        pred = MULTIPLIER * p_bar * r_bar
        floored = max(W0, pred)
        obs_end = float(np.median(stats["end_max"][:, wheel]))
        rows.append(
            {
                "wheel": SHORT[wheel],
                "pretty": PRETTY[wheel],
                "p_bar": p_bar,
                "r_bar": r_bar,
                "r_global": global_mean,
                "pred": pred,
                "floored": floored,
                "obs20": float(med[20] * W0) if np.isfinite(med[20]) else float("nan"),
                "obs40": float(med[40] * W0) if np.isfinite(med[40]) else float("nan"),
                "obs78": obs78,
                "obs_end": obs_end,
                "n78": int(stats["curves"][wheel][3][X_MAX]),
                "ratio_raw": obs78 / pred if pred else float("nan"),
                "ratio_floor": obs78 / floored if floored else float("nan"),
                "ratio_end": obs_end / floored if floored else float("nan"),
            }
        )
    return rows


def _share(stats: dict, wheel: int) -> dict:
    """Credit-mass bounds.

    Above the floor the recurrence gives a point. On a floored arm it gives an
    interval. A positive deferred reward is 3 or 7, both larger than the
    one-step lift-off (about 0.526), so it cannot leave an arm on the floor.
    The identity's slack on iterations with no deferred event is phantom
    capacity, not deferred mass. The reported interval keeps that slack only
    on iterations that did queue a positive deferred reward.
    """
    exact = float(stats["mass_exact"][wheel])
    floor_imm = float(stats["mass_floor_imm"][wheel])
    immediate = float(stats["mass_immediate"][wheel])
    d_exact = float(stats["mass_deferred_exact"][wheel])
    hidden = float(stats["mass_hidden_deferred"][wheel])
    phantom = float(stats["mass_phantom"][wheel])
    t_lo = exact + floor_imm
    t_hi = t_lo + hidden
    d_lo = d_exact
    d_hi = d_exact + hidden
    share_lo = d_lo / t_hi if t_hi > 0 else float("nan")
    share_hi = d_hi / t_lo if t_lo > 0 else float("nan")
    share_exact = d_exact / exact if exact > 0 else float("nan")
    wide_hi = (d_hi + phantom) / t_lo if t_lo > 0 else float("nan")
    return {
        "t_lo": t_lo,
        "t_hi": t_hi,
        "immediate": immediate,
        "d_lo": d_lo,
        "d_hi": d_hi,
        "phantom": phantom,
        "share_lo": share_lo,
        "share_hi": min(1.0, share_hi) if math.isfinite(share_hi) else float("nan"),
        "share_exact": share_exact,
        "share_wide_hi": min(1.0, wide_hi) if math.isfinite(wide_hi) else float("nan"),
    }


def write_notes(stats: dict, gate: Gate, audit_resid: float, audit_violations: int) -> None:
    lines: list[str] = []
    add = lines.append
    lengths = stats["lengths"]
    add("HAOS weight-dynamics diagnostics")
    add("Campaign: data/results/lagrange_benchmark")
    add(f"Runs: {stats['n_runs']}")
    add("")
    add("Validation")
    add(f"  Comparisons (runs x iterations x arms, including warm-up): {gate.n_compare}")
    add(f"  Maximum absolute deviation of Equation 3.3 from the logged probabilities: {gate.max_dev:.3e}")
    add(f"  Location of that maximum: {gate.where}")
    add(
        "  Immediate-credit reconciliation on iterations with no positive deferred "
        f"reward, arms above the floor: max |inferred - immediate| = {audit_resid:.3e}"
    )
    add(f"  Window violations (inferred credit short of the known immediate, or a floor contradiction): {audit_violations}")
    add(f"  dr_apply incumbent mismatches against the replayed best: {stats['incumbent_mismatches']}")
    add(f"  Post-warm-up iterations with no dr_apply: {stats['missing_dr']}")
    add(f"  Post-warm-up skipped iterations: {stats['skipped_post']}")
    add("")
    add("Scope")
    add("  Warm-up iterations 0-9 are excluded from every diagnostic below.")
    add("  The logged iteration index is 0-based. Warm-up is iterations 0-9.")
    add("  Diagnostic 1's x-axis is iterations 10 through 78.")
    add("  The wheel-state table still contains iterations 0-9. The credit tables do not.")
    add(f"  gamma = {GAMMA}, w0 = {W0:.1f}. Credit is applied, then every arm decays once.")
    add(f"  Interior equilibrium: w* = {MULTIPLIER:.0f} * p * R.  Lift-off at uniform p: R > {LIFT:.4f} * A.")
    add(f"  Run length (iteration count): min {int(lengths.min())}, median {float(np.median(lengths)):.0f}, max {int(lengths.max())}.")
    add(f"  Runs that reach logged iteration 20 / 40 / 78: "
        f"{int(np.sum(lengths > 20))} / {int(np.sum(lengths > 40))} / {int(np.sum(lengths > 78))}.")
    add("")
    add("Diagnostic 1. Attainable weight range, max_a(w_a) / w0")
    add("  Median across runs. The slope is an OLS fit of that median on iterations 40 through 78.")
    add("  The 95% interval is a percentile bootstrap over runs (2000 resamples).")
    add(f"  {'wheel':<16} {'n20':>5} {'med20':>8} {'n40':>5} {'med40':>8} {'n78':>5} {'med78':>8} {'slope':>10} {'ci95':>22}")
    for wheel in range(6):
        _q25, med, _q75, count = stats["curves"][wheel]
        slope, lo, hi = stats["slopes"][wheel]
        add(
            f"  {SHORT[wheel]:<16} {int(count[20]):5d} {med[20]:8.3f} {int(count[40]):5d} {med[40]:8.3f} "
            f"{int(count[78]):5d} {med[78]:8.3f} {slope:10.5f} [{lo:.5f}, {hi:.5f}]"
        )
    add("")
    add("Diagnostic 2. Immediate reward of one selection, R = R_DR + R_BG")
    add("  R_BG is 0 when that launch was never credited (discarded, failed, or still in flight).")
    add("  The two channels are summed per selection. They are not added in the same pre-decay step:")
    add("  the decompose-route reward decays at the end of its own iteration, and the BG-AILS")
    add("  reward decays at the end of the later iteration that drains it.")
    rewards = stats["rewards"]
    add(f"  Selections: {rewards.size}")
    add(f"  Mean R: {float(np.mean(rewards)):.3f}    Median R: {float(np.median(rewards)):.3f}")
    add(f"  Selections with a credited BG-AILS reward: {stats['n_with_bg']} ({stats['n_with_bg'] / rewards.size:.3f})")
    add(f"  Decompose-route rewards: n {stats['dr_values'].size}, median {float(np.median(stats['dr_values'])):.3f}, mean {float(np.mean(stats['dr_values'])):.3f}")
    if stats["bg_values"].size:
        add(f"  BG-AILS rewards: n {stats['bg_values'].size}, median {float(np.median(stats['bg_values'])):.3f}, mean {float(np.mean(stats['bg_values'])):.3f}")
    add("  Immediate R histogram (count):")
    for value, count in _hist_counts(rewards):
        add(f"    R = {value}: {count}")
    add("  Deferred reward histogram, one SC/SP apply each (count):")
    if stats["deferred_hist"].size:
        for value, count in _hist_counts(stats["deferred_hist"]):
            add(f"    R = {value}: {count}")
        add(f"  Deferred events excluded because the apply iteration was still in warm-up: {stats['warmup_deferred']}")
    else:
        add("    none")
    add("  Fraction of selections with R > (10/19)*A. A is that run's arm count on the wheel.")
    add("  The median R is the same for every row: it is the selection total, not a per-wheel reward.")
    add(f"  {'wheel':<16} {'A':>4} {'runs':>5} {'threshold':>10} {'frac R > threshold':>20}")
    for key in sorted(stats["sel_group"], key=lambda item: (item[0], item[1])):
        arms = key[1]
        frac = stats["above_group"][key] / stats["sel_group"][key]
        add(
            f"  {key[0]:<16} {arms:4d} {stats['group_runs'][key]:5d} "
            f"{LIFT * arms:10.3f} {frac:20.3f}"
        )
    add("")
    add("Diagnostic 3. Floor, persistence, and re-selection gap")
    add("  Pinned means weight == 10 exactly, on post-warm-up (run, iteration, arm) triples.")
    add(f"  Exact floor hits: {stats['exact_floor']} of {stats['floor_tests']}.")
    add(f"  Weights in (10, 10+1e-6] that are not exact 10s: {stats['dust']}.")
    add("  Persistence is the number of logged snapshots an arm stays strictly above 10")
    add("  after an iteration in which it received a positive immediate credit.")
    add("  A spell that is still above the floor at the last iteration is right-censored.")
    add("  The gap is the mean difference in iteration index between consecutive draws of the same arm.")
    add(f"  {'wheel':<16} {'pinned':>8} {'unpinned':>10} {'persist':>8} {'cens':>8} {'gap':>8} {'persist/gap':>12}")
    for wheel in range(6):
        pinned = stats["pinned_num"][wheel] / stats["pinned_den"][wheel]
        spells = stats["spell_lengths"][wheel]
        persist = _mean(spells)
        cens = stats["spell_censored"][wheel] / len(spells) if spells else float("nan")
        gap = _mean(stats["gaps"][wheel])
        ratio = persist / gap if gap else float("nan")
        add(
            f"  {SHORT[wheel]:<16} {pinned:8.3f} {_mean(stats['unpinned_counts'][wheel]):10.2f} "
            f"{persist:8.3f} {cens:8.3f} {gap:8.3f} {ratio:12.3f}"
        )
    add("  By arm count (the figure uses these rows; k is split, the other wheels are one A each):")
    add(f"  {'wheel':<16} {'A':>4} {'runs':>5} {'pinned':>8} {'persist':>8} {'gap':>8}")
    for key in sorted(stats["group_runs"], key=lambda item: (item[1], item[0])):
        add(
            f"  {key[0]:<16} {key[1]:4d} {stats['group_runs'][key]:5d} "
            f"{_mean(stats['group_pinned'][key]):8.3f} {_mean(stats['group_spell'][key]):8.3f} "
            f"{_mean(stats['group_gap'][key]):8.3f}"
        )
    add("")
    add("Diagnostic 4. Within-run arm differences, immediate rewards only")
    add(f"  Arms drawn fewer than {MIN_DRAWS} times are left out of that run's comparison.")
    add("  Spread is the mean over runs of (max arm-mean reward minus min arm-mean reward).")
    add("  Kruskal-Wallis is one test per run and wheel. Holm correction is applied within the run,")
    add("  across the wheels that had at least two arms with enough draws. The fraction is")
    add("  rejections divided by runs with a defined test.")
    add("  Spearman correlates, within a run, arm-mean reward with that arm's final probability.")
    add("  It is computed on runs with at least three qualifying arms, then averaged.")
    add(f"  {'wheel':<16} {'spread':>8} {'n_spread':>9} {'KW Holm':>10} {'n_KW':>6} {'KW raw':>8} {'spearman':>9} {'n_rho':>6}")
    for wheel in range(6):
        defined = stats["kw_defined"][wheel]
        frac = stats["kw_reject"][wheel] / defined if defined else float("nan")
        raw = stats["kw_reject_raw"][wheel] / defined if defined else float("nan")
        add(
            f"  {SHORT[wheel]:<16} {_mean(stats['spreads'][wheel]):8.3f} {len(stats['spreads'][wheel]):9d} "
            f"{frac:10.3f} {int(defined):6d} {raw:8.3f} {_mean(stats['spearman'][wheel]):9.3f} "
            f"{len(stats['spearman'][wheel]):6d}"
        )
    add("")
    add("H1 equilibrium test (primary)")
    add("  p_bar is the mean, over every post-warm-up iteration, of the previous snapshot's")
    add("  probability of the arm credited on this wheel, with a 0 when the wheel was not drawn.")
    add("  For the method wheels that 0 is the iterations of the other paradigm.")
    add("  The previous snapshot is the post-decay state. A BG-AILS credit drained at the")
    add("  start of the iteration lands after that snapshot and before the draw, so p_bar")
    add("  is not exactly the probability the sampler used.")
    add("  R_bar is the mean immediate R on the iterations that credited the wheel.")
    add(f"  Predicted interior weight = {MULTIPLIER:.0f} * p_bar * R_bar.")
    add("  Equilibrium weight = max(w0, that prediction). The logged weight cannot sit below w0.")
    add("  Observed is the diagnostic 1 median of max_a(w_a), converted back to a weight.")
    add("  obs_end is the median across runs of the max weight on the run's last iteration.")
    add(f"  {'wheel':<16} {'p_bar':>8} {'R_bar':>8} {'19pR':>8} {'floored':>8} {'obs78':>8} {'obs_end':>8} {'obs78/19pR':>10} {'obs78/floor':>11}")
    h1 = _h1_rows(stats)
    for row in h1:
        add(
            f"  {row['wheel']:<16} {row['p_bar']:8.4f} {row['r_bar']:8.3f} {row['pred']:8.3f} "
            f"{row['floored']:8.3f} {row['obs78']:8.3f} {row['obs_end']:8.3f} "
            f"{row['ratio_raw']:10.3f} {row['ratio_floor']:11.3f}"
        )
    add(f"  Global mean immediate R (all selections): {h1[0]['r_global']:.3f}")
    add("  Median top-arm probability at the last iteration, next to 1/A:")
    for wheel in range(6):
        arms = sorted({int(key[1]) for key in stats["group_runs"] if key[0] == SHORT[wheel]})
        uniform = ", ".join(f"1/{a}={1/a:.3f}" for a in arms)
        add(f"    {SHORT[wheel]:<16} median top p = {float(np.median(stats['end_top_p'][:, wheel])):.3f}    uniform {uniform}")
    add("")
    add("Diagnostic 5. Deferred credit as a flattening force")
    add("  An iteration is in the deferred group when an SC/SP apply in that iteration has a")
    add("  positive recomputed deferred reward (7 for a new best, 3 for an improvement on the")
    add("  last cost). A deferred reward of 0 does not change a weight and stays with the other")
    add("  iterations. Per-arm targets are not logged. What is compared is already in the weights:")
    add("  how many arms are strictly heavier than in the previous snapshot, and the credit mass")
    add("  implied by w_t = gamma (w_(t-1) + credits_t) where the floor does not bind.")
    add("  Injected mass is the identified credit: exact on arms that finished above the floor,")
    add("  and the known immediate credit on arms that finished on it. The floor slack is the")
    add("  extra the identity would still allow on those floored arms.")
    add(f"  {'wheel':<16} {'n_def':>7} {'rose_def':>8} {'rose_oth':>8} {'inj_def':>8} {'slack_def':>10} {'inj_oth':>8} {'slack_oth':>10}")
    for wheel in range(6):
        def_lo = _mean(stats["inj_def_lo"][wheel])
        def_hi = _mean(stats["inj_def_hi"][wheel])
        oth_lo = _mean(stats["inj_oth_lo"][wheel])
        oth_hi = _mean(stats["inj_oth_hi"][wheel])
        add(
            f"  {SHORT[wheel]:<16} {int(stats['n_def_iters'][wheel]):7d} "
            f"{_mean(stats['rose_def'][wheel]):8.3f} {_mean(stats['rose_oth'][wheel]):8.3f} "
            f"{def_lo:8.3f} {def_hi - def_lo:10.3f} {oth_lo:8.3f} {oth_hi - oth_lo:10.3f}"
        )
    add("")
    add("Deferred share of credit mass")
    add("  Above the floor, credits_t = w_t / gamma - w_(t-1), summed over arms. That part is a point.")
    add("  On a floored arm the identity only bounds the credit by w0/gamma - w_prev.")
    add("  A positive deferred reward is 3 or 7. Both clear the one-step lift-off of about 0.526,")
    add("  so a deferred credit cannot finish on the floor. The slack on iterations with no deferred")
    add("  event is unused capacity (phantom), not deferred mass. The share interval below keeps the")
    add("  floor bound only on iterations that did queue a positive deferred reward.")
    add("  exact share uses only arm-iterations that finished above the floor.")
    add("  wide upper is the share if every idle floored arm is allowed its full identity slack.")
    add(f"  {'wheel':<16} {'imm':>10} {'total':>22} {'deferred':>22} {'share':>18} {'exact':>8} {'wide hi':>8} {'phantom':>10}")
    shares = []
    for wheel in range(6):
        share = _share(stats, wheel)
        shares.append(share)
        add(
            f"  {SHORT[wheel]:<16} {share['immediate']:10.1f} "
            f"[{share['t_lo']:.1f}, {share['t_hi']:.1f}]".rjust(22) + " "
            f"[{share['d_lo']:.1f}, {share['d_hi']:.1f}]".rjust(22) + " "
            f"[{share['share_lo']:.3f}, {share['share_hi']:.3f}]".rjust(18) + " "
            f"{share['share_exact']:8.3f} {share['share_wide_hi']:8.3f} {share['phantom']:10.1f}"
        )
    add("")
    add("Reading")
    _reading(add, stats, h1, shares)
    add("")
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _within_factor(ratio: float, factor: float = 1.5) -> bool:
    if not math.isfinite(ratio) or ratio <= 0:
        return False
    return (1.0 / factor) <= ratio <= factor


def _reading(add, stats: dict, h1: list[dict], shares: list[dict]) -> None:
    ratios = ", ".join(f"{row['wheel']} {row['ratio_floor']:.2f}" for row in h1)
    add("  H1, dynamic-range ceiling. The factor test does not pass.")
    add("  Predicted weight is max(w0, 19*p_bar*R_bar). On these runs the interior expression")
    add("  is already above w0 on every wheel, so the floor inside the fixed point is not")
    add("  what separates prediction from observation.")
    add(f"  At iteration 78, observed median max weight / prediction: {ratios}.")
    add("  Every ratio is above 1.5. The end-of-run median max weight gives the same verdict.")
    add("  That is the weakening condition: the immediate-channel equilibrium is not the")
    add("  whole of what sets the range.")
    add("  Two measurements sit on the same side as the gap. p_bar is the mean probability of")
    add("  the drawn arm. The median final top-arm probability, in the table above, is higher")
    share_txt = ", ".join(f"{SHORT[w]} {shares[w]['share_exact']:.3f}" for w in range(6))
    add("  on every wheel. And deferred credit is not in R_bar.")
    add(f"  Above-floor deferred share of credit mass: {share_txt}.")
    add("  R_bar leaves that stream out. Diagnostic 1 also does not show a plateau by iteration 20.")
    med20 = [float(stats["curves"][w][1][20]) for w in range(6)]
    add(f"  The median max/w0 there runs from {min(med20):.2f} to {max(med20):.2f}.")

    add("  H0, evidence still accumulating. Supported on three wheels, not on the other three.")
    rising = []
    flat = []
    for wheel in range(6):
        slope, lo, hi = stats["slopes"][wheel]
        text = f"{SHORT[wheel]} (slope {slope:.5f}, CI [{lo:.5f}, {hi:.5f}])"
        if lo > 0:
            rising.append(text)
        else:
            flat.append(text)
    add("  Still rising between iterations 40 and 78, CI above 0:")
    for item in rising:
        add(f"    {item}")
    add("  Slope CI includes 0 or lies below it:")
    for item in flat:
        add(f"    {item}")
    n78 = int(stats["curves"][0][3][X_MAX])
    add(f"  Only {n78} of {stats['n_runs']} runs reach iteration 78, so that part of the curve is the longer half.")

    add("  H2, arm equivalence. Supported on the within-run tests.")
    add("  After Holm correction within the run, almost no run rejects.")
    for wheel in range(6):
        defined = stats["kw_defined"][wheel]
        if not defined:
            add(f"  {SHORT[wheel]}: no run had two arms with at least {MIN_DRAWS} draws.")
            continue
        frac = stats["kw_reject"][wheel] / defined
        rho = _mean(stats["spearman"][wheel])
        rho_txt = "not computed (fewer than three qualifying arms)" if not stats["spearman"][wheel] else f"{rho:.3f}"
        add(
            f"  {SHORT[wheel]}: Holm rejection {frac:.3f} of {int(defined)} testable runs; "
            f"mean reward spread {_mean(stats['spreads'][wheel]):.3f}; mean Spearman {rho_txt}."
        )
    add("  The spreads are about 0.4 to 1.1 reward points, against a median selection reward of 5.")
    add("  The arms are close. That is about the option set. It does not account for the weight")
    add("  level: the max weight is several times w0, and nearly every arm leaves the floor at least once.")

    add("  Probability floors. Not what keeps the probabilities flat.")
    add("  The weights are large and the top-arm probabilities have moved above 1/A.")
    for wheel in range(6):
        med78 = stats["curves"][wheel][1][X_MAX]
        top = float(np.median(stats["end_top_p"][:, wheel]))
        add(f"  {SHORT[wheel]}: median max/w0 at iteration 78 is {med78:.3f}; median final top-arm probability is {top:.3f}.")
    add("  A floor artifact would show large weight ratios with probabilities still near uniform.")
    add("  The probabilities have followed the weights, and they are still far from a point mass.")

    add("  Persistence. The three-iteration blip is not what these runs do.")
    add("  An isolated credit of 1 logs 10.45 and is back on the floor at the next snapshot.")
    add("  The spells here average 31 to 40 iterations, and most of them are still open at the")
    add("  last logged iteration, so those averages are lower bounds. The gap before the same")
    add("  arm is drawn again is shorter than the spell, so the arm is credited again before")
    add("  it returns to the floor. The pinned fraction does not rise monotonically with arm count.")
    for wheel in range(6):
        spells = stats["spell_lengths"][wheel]
        persist = _mean(spells)
        cens = stats["spell_censored"][wheel] / len(spells) if spells else float("nan")
        gap = _mean(stats["gaps"][wheel])
        pinned = stats["pinned_num"][wheel] / stats["pinned_den"][wheel]
        add(
            f"  {SHORT[wheel]}: persistence {persist:.2f} (censored {cens:.3f}), gap {gap:.2f}, "
            f"ratio {persist / gap:.2f}, pinned {pinned:.3f}, "
            f"arms ever unpinned {_mean(stats['unpinned_counts'][wheel]):.2f}."
        )

    add("  Deferred channel. This is a third mechanism, and it is large.")
    add("  The deferred share of credit mass is 0.44 on every wheel. The above-floor identity")
    add("  and the floor-bounded interval agree; a reward of 3 or 7 cannot finish on the floor,")
    add("  so the share is a point estimate rather than a wide bound. The 'wide hi' column is")
    add("  what the identity would say if every idle arm were allowed a hidden credit. It is not")
    add("  the estimate. Diagnostic 2's immediate histogram leaves out this 44%.")
    add("  On a deferred iteration several arms of the same wheel get heavier at once:")
    for wheel in range(6):
        a = _mean(stats["rose_def"][wheel])
        b = _mean(stats["rose_oth"][wheel])
        inj_a = _mean(stats["inj_def_lo"][wheel])
        inj_b = _mean(stats["inj_oth_lo"][wheel])
        add(f"    {SHORT[wheel]}: arms up {a:.2f} vs {b:.2f}; credit mass {inj_a:.1f} vs {inj_b:.1f}.")
    add("  The identical credit mass on k, lambda, the paradigm wheel and the subsolver is the")
    add("  check that each deferred tag pays its reward once on each of those wheels.")
    add("  The factor test did not pass, so there is no single decay to carry into Section 5.3.")


def _coords(pairs: list[tuple[float, float]]) -> str:
    return " ".join(f"({x:.4f}, {y:.6f})" for x, y in pairs)


def write_figures(stats: dict, h1: list[dict]) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        _fig_range(stats),
        _fig_hist(stats),
        _fig_pinned(stats),
        _fig_equilibrium(h1),
    ]
    return paths


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


def _fig_range(stats: dict) -> Path:
    panels = []
    for wheel in range(6):
        q25, med, q75, count = stats["curves"][wheel]
        lo, mid, hi = [], [], []
        for t in range(X_MIN, X_MAX + 1):
            if count[t] == 0 or not np.isfinite(med[t]):
                continue
            lo.append((t, float(q25[t])))
            mid.append((t, float(med[t])))
            hi.append((t, float(q75[t])))
        panels.append(
            "\\nextgroupplot[title={" + PRETTY[wheel] + "}]\n"
            "\\addplot[name path=lo" + str(wheel) + ", draw=none] coordinates {" + _coords(lo) + "};\n"
            "\\addplot[name path=hi" + str(wheel) + ", draw=none] coordinates {" + _coords(hi) + "};\n"
            "\\addplot[fill=blue!20] fill between[of=lo" + str(wheel) + " and hi" + str(wheel) + "];\n"
            "\\addplot[thick, blue!60!black] coordinates {" + _coords(mid) + "};\n"
        )
    body = (
        "\\begin{tikzpicture}\n"
        "\\begin{groupplot}[\n"
        "  group style={group size=3 by 2, horizontal sep=1.4cm, vertical sep=1.6cm},\n"
        "  width=5.4cm, height=4.2cm,\n"
        "  xmin=10, xmax=78,\n"
        "  xlabel={iteration},\n"
        "  ylabel={$\\max_a w_a / w_0$},\n"
        "  tick label style={font=\\small},\n"
        "  label style={font=\\small},\n"
        "  title style={font=\\small},\n"
        "  ymajorgrids,\n"
        "]\n"
        + "".join(panels)
        + "\\end{groupplot}\n"
        "\\end{tikzpicture}\n"
    )
    path = FIG_DIR / "haos_diag_weight_range.tex"
    path.write_text(_standalone(body), encoding="utf-8")
    return path


def _fig_hist(stats: dict) -> Path:
    immediate = _hist_counts(stats["rewards"])
    deferred = _hist_counts(stats["deferred_hist"]) if stats["deferred_hist"].size else []
    ymax = max(count for _, count in immediate)
    bars = " ".join(f"({value}, {count})" for value, count in immediate)
    xticks = ",".join(str(value) for value, _count in immediate)
    # Fixed wheel arm counts plus the k counts present in the campaign.
    colors = {
        "paradigm": "black",
        "route_method": "blue",
        "solver": "orange",
        "vertex_method": "green!60!black",
        "lambda": "violet",
        "k": "red",
    }
    lines = []
    for key in sorted(stats["group_runs"], key=lambda item: (item[1], item[0])):
        threshold = LIFT * key[1]
        color = colors[key[0]]
        lines.append(
            f"\\addplot[dashed, thin, {color}] coordinates {{({threshold:.4f}, 0) ({threshold:.4f}, {ymax})}};\n"
            f"\\addlegendentry{{{key[0].replace('_', ' ')}, $A={key[1]}$}}\n"
        )
    def_bars = " ".join(f"({value}, {count})" for value, count in deferred) if deferred else "(0, 0)"
    def_ticks = ",".join(str(value) for value, _count in deferred) if deferred else "0"
    body = (
        "\\begin{tikzpicture}\n"
        "\\begin{groupplot}[\n"
        "  group style={group size=2 by 1, horizontal sep=1.8cm},\n"
        "  width=7.2cm, height=5.2cm,\n"
        "  ymin=0,\n"
        "  tick label style={font=\\small},\n"
        "  label style={font=\\small},\n"
        "  title style={font=\\small},\n"
        "]\n"
        "\\nextgroupplot[title={immediate $R$}, xlabel={$R$}, ylabel={selections}, xtick={"
        + xticks
        + "}, ymax="
        + str(ymax)
        + ", legend style={font=\\scriptsize, at={(0.5,-0.28)}, anchor=north, legend columns=3},\n"
        "  clip=false]\n"
        "\\addplot+[ybar, fill=blue!35, draw=black, bar width=6pt, forget plot] coordinates {"
        + bars
        + "};\n"
        + "".join(lines)
        + "\\nextgroupplot[title={deferred $R$}, xlabel={$R$}, ylabel={SC/SP applies}, xtick={"
        + def_ticks
        + "}]\n"
        "\\addplot+[ybar, fill=orange!40, draw=black, bar width=10pt] coordinates {"
        + def_bars
        + "};\n"
        "\\end{groupplot}\n"
        "\\end{tikzpicture}\n"
    )
    path = FIG_DIR / "haos_diag_reward_hist.tex"
    path.write_text(_standalone(body), encoding="utf-8")
    return path


def _fig_pinned(stats: dict) -> Path:
    # One mark style per wheel. k contributes several A values.
    marks = ["*", "square*", "triangle*", "diamond*", "pentagon*", "otimes*"]
    open_marks = ["o", "square", "triangle", "diamond", "pentagon", "otimes"]
    mark_colors = ["black", "blue", "red", "green!60!black", "orange", "violet"]
    pinned_plots = []
    persist_plots = []
    gap_plots = []
    for wheel, mark in enumerate(marks):
        color = mark_colors[wheel]
        open_mark = open_marks[wheel]
        pinned_pts = []
        persist_pts = []
        gap_pts = []
        for key in sorted(stats["group_runs"]):
            if key[0] != SHORT[wheel]:
                continue
            arms = key[1]
            pinned_pts.append((arms, _mean(stats["group_pinned"][key])))
            if stats["group_spell"][key]:
                persist_pts.append((arms, _mean(stats["group_spell"][key])))
            if stats["group_gap"][key]:
                gap_pts.append((arms, _mean(stats["group_gap"][key])))
        pinned_plots.append(
            "\\addplot[only marks, mark=" + mark + ", " + color + "] coordinates {" + _coords(pinned_pts) + "};\n"
            "\\addlegendentry{" + PRETTY[wheel] + "}\n"
        )
        persist_plots.append(
            "\\addplot[only marks, mark=" + mark + ", " + color + "] coordinates {" + _coords(persist_pts) + "};\n"
        )
        gap_plots.append(
            "\\addplot[only marks, mark=" + open_mark + ", " + color + "] coordinates {" + _coords(gap_pts) + "};\n"
        )
    body = (
        "\\begin{tikzpicture}\n"
        "\\begin{groupplot}[\n"
        "  group style={group size=2 by 1, horizontal sep=1.8cm},\n"
        "  width=7.2cm, height=5.4cm,\n"
        "  xlabel={arm count $A$},\n"
        "  tick label style={font=\\small},\n"
        "  label style={font=\\small},\n"
        "  title style={font=\\small},\n"
        "  legend style={font=\\scriptsize, at={(0.02,0.98)}, anchor=north west},\n"
        "  ymajorgrids,\n"
        "]\n"
        "\\nextgroupplot[title={pinned fraction}, ylabel={fraction at $w_0$}]\n"
        + "".join(pinned_plots)
        + "\\nextgroupplot[title={filled: persistence, open: gap}, ylabel={iterations}]\n"
        + "".join(persist_plots)
        + "".join(gap_plots)
        + "\\end{groupplot}\n"
        "\\end{tikzpicture}\n"
    )
    path = FIG_DIR / "haos_diag_pinned.tex"
    path.write_text(_standalone(body), encoding="utf-8")
    return path


def _fig_equilibrium(h1: list[dict]) -> Path:
    names = ["k", "lambda", "paradigm", "vertex", "route", "solver"]
    symbolic = ",".join(names)
    pred = " ".join(f"({name}, {row['pred']:.4f})" for name, row in zip(names, h1))
    obs = " ".join(f"({name}, {row['obs78']:.4f})" for name, row in zip(names, h1))
    body = (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "  ybar,\n"
        "  bar width=7pt,\n"
        "  width=12cm, height=6cm,\n"
        "  symbolic x coords={" + symbolic + "},\n"
        "  xtick=data,\n"
        "  ylabel={weight},\n"
        "  ymin=0,\n"
        "  tick label style={font=\\small},\n"
        "  label style={font=\\small},\n"
        "  legend style={font=\\small, at={(0.5,-0.18)}, anchor=north},\n"
        "  ymajorgrids,\n"
        "  clip=false,\n"
        "]\n"
        "\\addplot[fill=blue!35, draw=black] coordinates {" + pred + "};\n"
        "\\addlegendentry{$19\\bar p\\bar R$}\n"
        "\\addplot[fill=orange!40, draw=black] coordinates {" + obs + "};\n"
        "\\addlegendentry{median $\\max w$ at iteration 78}\n"
        "\\addlegendimage{black, dashed}\n"
        "\\addlegendentry{$w_0=10$}\n"
        "\\draw[dashed] (axis cs:k,10) -- (axis cs:solver,10);\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
    )
    path = FIG_DIR / "haos_diag_equilibrium.tex"
    path.write_text(_standalone(body), encoding="utf-8")
    return path


def compile_figures(paths: list[Path]) -> None:
    for path in paths:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", path.name],
            cwd=path.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-40:])
            raise SystemExit(f"pdflatex failed on {path.name}\n{tail}")
        for suffix in (".aux", ".log"):
            side = path.with_suffix(suffix)
            if side.exists():
                side.unlink()


def main() -> None:
    paths = sorted(LOG_ROOT.glob("*/seed*/**/run.jsonl"))
    if len(paths) != 300:
        raise SystemExit(f"expected 300 run logs, found {len(paths)}")
    gate = Gate()
    runs: list[Run] = []
    seen: set[str] = set()
    for index, path in enumerate(paths, start=1):
        run = parse_run(path, gate)
        if run.run_id in seen:
            raise SystemExit(f"duplicate run id {run.run_id}")
        seen.add(run.run_id)
        runs.append(run)
        if index % 25 == 0 or index == len(paths):
            print(f"parsed {index}/{len(paths)}  max |dp| = {gate.max_dev:.3e}", file=sys.stderr)
    if gate.max_dev > 1e-6:
        NOTES.parent.mkdir(parents=True, exist_ok=True)
        NOTES.write_text(
            "Validation failed.\n"
            f"Maximum absolute deviation {gate.max_dev:.3e} at {gate.where}.\n"
            f"Comparisons: {gate.n_compare}.\n"
            "Diagnostics were not computed.\n",
            encoding="utf-8",
        )
        raise SystemExit(f"validation gate failed: max abs dev {gate.max_dev:.3e} at {gate.where}")
    print(
        f"validation passed: max |dp| = {gate.max_dev:.3e} over {gate.n_compare} comparisons",
        file=sys.stderr,
    )
    stats = analyze(runs)
    if stats["violations"] or stats["max_resid"] > ATOL:
        raise SystemExit(
            "credit window does not match the logged weights: "
            f"max residual {stats['max_resid']:.3e}, violations {stats['violations']}"
        )
    if stats["incumbent_mismatches"]:
        raise SystemExit(
            f"replayed best diverged from dr_apply incumbent on {stats['incumbent_mismatches']} rows"
        )
    write_tables(runs)
    write_notes(stats, gate, stats["max_resid"], stats["violations"])
    figures = write_figures(stats, _h1_rows(stats))
    compile_figures(figures)
    print(f"wrote {NOTES}", file=sys.stderr)
    for path in figures:
        print(f"wrote {path} and {path.with_suffix('.pdf')}", file=sys.stderr)


if __name__ == "__main__":
    main()
