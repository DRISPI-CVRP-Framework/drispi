#!/usr/bin/env python3
"""Paired one-sided Wilcoxon tests of DRISPI's mean-of-3 gap against five baselines.

The earlier comparison ranked raw cost differences (wilcoxon of the two cost
samples, two-sided, approximate, no continuity correction). This script tests
the percentage-gap differences the section reports, with method='approx' and
correction=False, and Holm-corrects each one-sided family of five separately.

Writes data/results/baseline_gaps.csv, data/results/paired_tests_lagrange.json,
and thesis/tables/paired_tests_body.tex.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "data/results/xl_solver_comparison_lagrange.json"
CAMPAIGN = ROOT / "data/results/finalBenchmarkResults_lagrange.csv"
BASELINE_CSV = ROOT / "data/results/baseline_gaps.csv"
OUT_JSON = ROOT / "data/results/paired_tests_lagrange.json"
OUT_TEX = ROOT / "thesis/tables/paired_tests_body.tex"

BASELINES = ["AILS-II", "FILO2", "FILO", "SISRs", "HGS-CVRP"]


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    keys = list(pvalues)
    raw = np.array([pvalues[key] for key in keys])
    order = np.argsort(raw)
    adjusted = np.empty(len(raw))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(raw) - rank) * raw[index])
        adjusted[index] = min(running, 1.0)
    return dict(zip(keys, adjusted))


def hodges_lehmann(diffs: np.ndarray) -> float:
    walsh = (diffs[:, None] + diffs[None, :])[np.triu_indices(len(diffs))] / 2.0
    return float(np.median(walsh))


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 4000, seed: int = 11) -> list[float]:
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    n = len(diffs)
    for i in range(n_boot):
        stats[i] = hodges_lehmann(diffs[rng.integers(0, n, n)])
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def format_p(value: float) -> str:
    if value >= 0.995:
        return "$1.00$"
    if value >= 0.01:
        return f"${value:.3f}$"
    mantissa, exponent = f"{value:.2e}".split("e")
    exp = int(exponent)
    return f"${float(mantissa):.1f} \\times 10^{{{exp}}}$"


def main() -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    campaign = {}
    with CAMPAIGN.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            campaign[row["instance"]] = row

    instances = sorted(comparison)
    if len(instances) != 100:
        raise SystemExit(f"expected 100 instances, found {len(instances)}")

    with BASELINE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "bks_current", "drispi_mean", *BASELINES])
        for name in instances:
            entry = comparison[name]
            writer.writerow(
                [
                    name,
                    entry["bks"],
                    entry["solvers"]["DRISPI"]["mean"],
                    *[entry["solvers"][solver]["mean"] for solver in BASELINES],
                ]
            )

    rows = []
    better: dict[str, float] = {}
    worse: dict[str, float] = {}
    for solver in BASELINES:
        diffs = []
        wins = 0
        for name in instances:
            entry = comparison[name]
            bks = entry["bks"]
            drispi = float(campaign[name]["cost_mean3"])
            base = entry["solvers"][solver]["mean"]
            diffs.append((drispi - base) / bks * 100.0)
            if drispi < base:
                wins += 1
        diffs_arr = np.array(diffs)
        better[solver] = float(
            wilcoxon(diffs_arr, alternative="less", method="approx", correction=False).pvalue
        )
        worse[solver] = float(
            wilcoxon(diffs_arr, alternative="greater", method="approx", correction=False).pvalue
        )
        ci = bootstrap_ci(diffs_arr)
        rows.append(
            {
                "solver": solver,
                "mean_diff_pp": float(diffs_arr.mean()),
                "ahead": wins,
                "hodges_lehmann_pp": hodges_lehmann(diffs_arr),
                "hl_ci95": ci,
                "raw_p_better": better[solver],
                "raw_p_worse": worse[solver],
            }
        )

    holm_better = holm(better)
    holm_worse = holm(worse)
    lines = []
    for row in rows:
        solver = row["solver"]
        pb = holm_better[solver]
        pw = holm_worse[solver]
        if pw < 0.05:
            verdict = "worse"
        elif pb < 0.05:
            verdict = "better"
        else:
            verdict = "indistinguishable"
        row["holm_p_better"] = pb
        row["holm_p_worse"] = pw
        row["verdict"] = verdict
        sign = "+" if row["mean_diff_pp"] >= 0 else "-"
        lines.append(
            f"{solver} & ${sign}{abs(row['mean_diff_pp']):.3f}$ & {row['ahead']} & "
            f"{format_p(pb)} & {format_p(pw)} & {verdict} \\\\"
        )

    # \bottomrule has to live in this file. A \\ at the end of an \input,
    # followed by \bottomrule in the parent, is tokenized as an open cell.
    lines.append("\\bottomrule")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {BASELINE_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TEX}")
    for row in rows:
        print(
            f"{row['solver']:10} delta {row['mean_diff_pp']:+.4f} ahead {row['ahead']:3d} "
            f"HL {row['hodges_lehmann_pp']:+.4f} CI {row['hl_ci95']} "
            f"holm better {row['holm_p_better']:.4g} worse {row['holm_p_worse']:.4g} {row['verdict']}"
        )


if __name__ == "__main__":
    main()
