#!/usr/bin/env python3
"""Compute the Section 5.2.1 comparison statistics for DRISPI vs. the eight
Queiroga et al. (2026) monolithic solvers, all against the current official
BKS.

Reads:
  - data/results/xl_solver_comparison.json (per-instance costs, built by
    scripts/build_solver_comparison_data.py)
  - data/results/finalBenchmarkResults.csv (DRISPI's per-seed costs, for the
    best-of-k subsampling progression and the seed standard deviation)

Writes data/results/xl_comparison_stats.json with:
  - summary_table: per-solver mean gap to BKS, for the mean-of-N and
    best-of-N metrics, each split into all / n<3400 / n>3400 (the same
    customer-count split Queiroga et al. use for their own Table 2).
  - best_of_k: DRISPI's best-of-1 (= mean-of-3), best-of-2 (average over the
    three possible 2-of-3 seed subsets), and best-of-3 (actual) gap to BKS,
    plus the marginal gain between each step and the mean per-instance
    across-seed standard deviation.
  - mean_gain: per published solver, the gap reduction from its mean-of-60
    to its best-of-60 (the same quantity DRISPI's best_of_k progression is
    compared against).
  - paired_tests: Wilcoxon signed-rank tests, DRISPI vs. AILS-II / FILO2 /
    FILO / SISRs / HGS-CVRP, for both the best and mean metric, with Holm
    correction across the resulting family of ten tests.
  - distribution: mean, median, IQR, 90th percentile, and max of the
    per-instance gap distribution, for DRISPI (best and mean) and the four
    baselines used in the paired tests.
  - highlights: the specific per-instance results called out in the text
    (beats the initial BKS, matches BKS exactly, beats AILS-II's 60-run
    mean).

Usage:
    python scripts/analyze_xl_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
COMPARISON_JSON = ROOT / "data/results/xl_solver_comparison_lagrange.json"
CSV_PATH = ROOT / "data/results/finalBenchmarkResults_lagrange.csv"
OUTPUT = ROOT / "data/results/xl_comparison_stats_lagrange.json"

SIZE_SPLIT = 3400  # customers; matches Queiroga et al. (2026) Table 2's own split

# All nine solvers shown in the compact summary table (Table 5.1).
SUMMARY_SOLVERS = [
    "AILS-II",
    "FILO",
    "FILO2",
    "KGLSXXL",
    "HGS-CVRP",
    "SISRs",
    "LKH-3",
    "OR-Tools",
    "DRISPI",
]

# Baselines with per-instance data for the paired tests. (The ECDF figure only
# plots the five solvers with a mean-of-N gap below 1%, a different subset.)
BASELINES = ["AILS-II", "FILO2", "FILO", "SISRs", "HGS-CVRP"]


def _gap_pct(cost: float, bks: float) -> float:
    return (cost - bks) / bks * 100.0


def _load_comparison(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def build_summary_table(data: dict[str, dict]) -> dict[str, dict]:
    def collect(metric: str, subset: str | None) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {s: [] for s in SUMMARY_SOLVERS}
        for entry in data.values():
            n = entry["n"]
            if subset == "lo" and not (n < SIZE_SPLIT):
                continue
            if subset == "hi" and not (n > SIZE_SPLIT):
                continue
            bks = entry["bks"]
            for solver in SUMMARY_SOLVERS:
                cost = entry["solvers"].get(solver, {}).get(metric)
                if cost is not None:
                    out[solver].append(_gap_pct(cost, bks))
        return out

    table: dict[str, dict] = {}
    for solver in SUMMARY_SOLVERS:
        row: dict[str, object] = {}
        for metric in ("mean", "best"):
            for subset_name, subset in (("all", None), ("lo", "lo"), ("hi", "hi")):
                vals = collect(metric, subset)[solver]
                row[f"{metric}_{subset_name}"] = round(mean(vals), 3) if vals else None
                row[f"n_{metric}_{subset_name}"] = len(vals)
        table[solver] = row
    return table


def _seed_cost_columns(rows: list[dict]) -> list[str]:
    if not rows:
        raise RuntimeError("Benchmark CSV has no rows")
    cols = [name for name in rows[0] if name.startswith("cost_seed")]
    cols.sort(key=lambda name: int(name.removeprefix("cost_seed")))
    if len(cols) != 3:
        raise RuntimeError(f"Expected 3 per-seed cost columns, found {cols}")
    return cols


def build_best_of_k(rows: list[dict]) -> dict[str, object]:
    seed_cols = _seed_cost_columns(rows)
    best1, best2, best3 = [], [], []
    for row in rows:
        bks = float(row["bks_current"])
        costs = [float(row[c]) for c in seed_cols]
        seed_gaps = [_gap_pct(c, bks) for c in costs]
        best1.append(mean(seed_gaps))  # expected best-of-1 == mean-of-3
        pair_mins = [min(costs[i], costs[j]) for i, j in combinations(range(3), 2)]
        best2.append(mean(_gap_pct(c, bks) for c in pair_mins))
        best3.append(_gap_pct(min(costs), bks))

    seed_sd = mean(float(row["seed_sd_pp"]) for row in rows)

    return {
        "best_of_1": round(mean(best1), 3),
        "best_of_2": round(mean(best2), 3),
        "best_of_3": round(mean(best3), 3),
        "gain_1_to_2": round(mean(best1) - mean(best2), 3),
        "gain_2_to_3": round(mean(best2) - mean(best3), 3),
        "mean_seed_sd_pp": round(seed_sd, 3),
    }


def build_mean_gain(summary_table: dict[str, dict]) -> dict[str, float]:
    gains = {}
    for solver in ("AILS-II", "FILO2", "FILO", "SISRs", "KGLSXXL", "HGS-CVRP"):
        row = summary_table[solver]
        gains[solver] = round(row["mean_all"] - row["best_all"], 3)
    return gains


def build_paired_tests(data: dict[str, dict]) -> list[dict]:
    instances = sorted(data.keys())
    raw_results = []
    for baseline in BASELINES:
        for metric in ("best", "mean"):
            drispi_vals, base_vals, bks_vals = [], [], []
            for inst in instances:
                entry = data[inst]
                b = entry["solvers"].get(baseline, {}).get(metric)
                if b is None:
                    continue
                drispi_vals.append(entry["solvers"]["DRISPI"][metric])
                base_vals.append(b)
                bks_vals.append(entry["bks"])
            n = len(drispi_vals)
            wins = sum(1 for d, b in zip(drispi_vals, base_vals) if d < b)
            pct_diffs = [(d - b) / bks * 100 for d, b, bks in zip(drispi_vals, base_vals, bks_vals)]
            stat, p = wilcoxon(drispi_vals, base_vals)
            raw_results.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "n": n,
                    "drispi_wins": wins,
                    "mean_diff_pp": round(mean(pct_diffs), 3),
                    "wilcoxon_p": p,
                }
            )

    # Holm-Bonferroni step-down correction across the family of 8 tests.
    ps = [r["wilcoxon_p"] for r in raw_results]
    m = len(ps)
    order = np.argsort(ps)
    holm_p = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * ps[idx]
        running_max = max(running_max, adj)
        holm_p[idx] = min(running_max, 1.0)
    for r, hp in zip(raw_results, holm_p):
        r["holm_p"] = hp
        r["significant_holm_0.05"] = bool(hp < 0.05)

    return raw_results


def build_distribution(data: dict[str, dict]) -> dict[str, dict]:
    def series(solver: str, metric: str) -> np.ndarray:
        vals = []
        for entry in data.values():
            c = entry["solvers"].get(solver, {}).get(metric)
            if c is not None:
                vals.append(_gap_pct(c, entry["bks"]))
        return np.array(vals)

    targets = [
        ("DRISPI", "best"),
        ("DRISPI", "mean"),
        ("AILS-II", "mean"),
        ("FILO2", "mean"),
        ("FILO", "mean"),
        ("SISRs", "mean"),
    ]
    out: dict[str, dict] = {}
    for solver, metric in targets:
        vals = series(solver, metric)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        out[f"{solver}_{metric}"] = {
            "mean": round(float(vals.mean()), 3),
            "median": round(float(med), 3),
            "iqr_low": round(float(q1), 3),
            "iqr_high": round(float(q3), 3),
            "p90": round(float(np.percentile(vals, 90)), 3),
            "max": round(float(vals.max()), 3),
            "n": len(vals),
        }
    return out


def build_highlights(data: dict[str, dict], rows: list[dict]) -> dict[str, object]:
    csv_rows = {row["instance"]: row for row in rows}

    def cost_row(inst: str) -> dict:
        r = csv_rows[inst]
        return {
            "instance": inst,
            "cost_best3": int(round(float(r["cost_best3"]))),
            "bks_initial": int(round(float(r["bks_initial"]))),
            "bks_current": int(round(float(r["bks_current"]))),
        }

    beats_initial = []
    exact_match = []
    for inst in sorted(csv_rows):
        record = cost_row(inst)
        if record["cost_best3"] < record["bks_initial"]:
            beats_initial.append(record)
        if record["cost_best3"] == record["bks_initial"] == record["bks_current"]:
            exact_match.append(record)

    beats_ails_mean = []
    for inst, entry in data.items():
        ails_mean = entry["solvers"].get("AILS-II", {}).get("mean")
        if ails_mean is None:
            continue
        if entry["solvers"]["DRISPI"]["best"] <= ails_mean:
            beats_ails_mean.append(inst)

    return {
        "beats_initial_bks": beats_initial,
        "exact_bks_match": exact_match,
        "n_beats_ails2_mean_with_best3": len(beats_ails_mean),
        "instances_beating_ails2_mean": sorted(beats_ails_mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", type=Path, default=COMPARISON_JSON)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    data = _load_comparison(args.comparison_json)
    rows = _load_csv_rows(args.csv)

    summary_table = build_summary_table(data)
    result = {
        "size_split_customers": SIZE_SPLIT,
        "summary_table": summary_table,
        "best_of_k": build_best_of_k(rows),
        "mean_gain": build_mean_gain(summary_table),
        "paired_tests": build_paired_tests(data),
        "distribution": build_distribution(data),
        "highlights": build_highlights(data, rows),
    }

    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
