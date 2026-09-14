"""Paired Wilcoxon / Hodges–Lehmann analysis for the BG-AILS ablation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from drispi.ablation import config as ac
from drispi.haos.k_domain import parse_n_kmin


def _collect_performance(root: Path | None = None) -> list[dict[str, Any]]:
    base = root or ac.PERF_DIR
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("*/*/performance_summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def instance_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Average percentage improvement over checkpoints, per instance and arm."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        inst = row["instance"]
        for arm, rec in row["arms"].items():
            buckets[inst][arm].append(float(rec["pct_improvement"]))
    return {
        inst: {arm: float(np.mean(vals)) for arm, vals in arms.items()}
        for inst, arms in buckets.items()
    }


def paired_vectors(
    means: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    names = sorted(means)
    a = np.array([means[n]["A"] for n in names], dtype=np.float64)
    b = np.array([means[n]["B"] for n in names], dtype=np.float64)
    c = np.array([means[n]["C"] for n in names], dtype=np.float64)
    return a, b, c, names


def hodges_lehmann(diff: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Hodges–Lehmann paired median with a Walsh-average 95% CI."""
    n = len(diff)
    walsh = np.array(
        [(diff[i] + diff[j]) / 2.0 for i in range(n) for j in range(i, n)],
        dtype=np.float64,
    )
    est = float(np.median(walsh))
    # Approximate CI via the Wilcoxon signed-rank critical value (normal).
    z = 1.959963984540054
    se = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    # Map the rank-sum CI onto Walsh averages by order statistics.
    lo_k = max(0, int(np.floor((len(walsh) - 1) / 2 - z * se / 2)))
    hi_k = min(len(walsh) - 1, int(np.ceil((len(walsh) - 1) / 2 + z * se / 2)))
    ordered = np.sort(walsh)
    return est, (float(ordered[lo_k]), float(ordered[hi_k]))


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvalues[idx] * (m - rank))
        running = max(running, val)
        adj[idx] = running
    return adj


def _pair_block(diff: np.ndarray) -> dict[str, Any]:
    if len(diff) < 2:
        hl, ci = (float("nan"), (float("nan"), float("nan")))
        if len(diff) == 1:
            hl, ci = hodges_lehmann(diff)
        return {
            "hodges_lehmann": hl,
            "ci95": list(ci),
            "wilcoxon_p": None,
            "holm_p": None,
            "wins": int(np.sum(diff > 0)),
        }
    w = stats.wilcoxon(diff, alternative="greater", zero_method="pratt")
    hl, ci = hodges_lehmann(diff)
    return {
        "hodges_lehmann": hl,
        "ci95": list(ci),
        "wilcoxon_p": float(w.pvalue),
        "holm_p": None,
        "wins": int(np.sum(diff > 0)),
    }


def analyse_performance(root: Path | None = None) -> dict[str, Any]:
    rows = _collect_performance(root)
    means = {
        inst: arms
        for inst, arms in instance_means(rows).items()
        if set(arms) >= {"A", "B", "C"}
    }
    a, b, c, names = paired_vectors(means) if means else (
        np.array([]),
        np.array([]),
        np.array([]),
        [],
    )
    d_ca = c - a
    d_cb = c - b
    block_ca = _pair_block(d_ca)
    block_cb = _pair_block(d_cb)
    raw = [block_ca["wilcoxon_p"], block_cb["wilcoxon_p"]]
    if None not in raw:
        adj = holm([float(p) for p in raw])
        block_ca["holm_p"] = adj[0]
        block_cb["holm_p"] = adj[1]
    t_b = [float(r["t_perturb_B_s"]) / float(r["T"]) for r in rows if r["T"] > 0]
    t_c = [float(r["t_perturb_C_s"]) / float(r["T"]) for r in rows if r["T"] > 0]
    t_m = [float(r["t_matrix_s"]) / float(r["T"]) for r in rows if r["T"] > 0]
    r64 = [
        n
        for n in names
        if (parsed := parse_n_kmin(n)) is not None and parsed[1] >= 64
    ]
    r64_out: dict[str, Any] | None = None
    if r64:
        d_ca_r = np.array([means[n]["C"] - means[n]["A"] for n in r64])
        d_cb_r = np.array([means[n]["C"] - means[n]["B"] for n in r64])
        hl_ca_r, ci_ca_r = hodges_lehmann(d_ca_r)
        hl_cb_r, ci_cb_r = hodges_lehmann(d_cb_r)
        r64_out = {
            "n_instances": len(r64),
            "note": "estimation only; not a confirmatory test",
            "C_minus_A": {"hodges_lehmann": hl_ca_r, "ci95": list(ci_ca_r)},
            "C_minus_B": {"hodges_lehmann": hl_cb_r, "ci95": list(ci_cb_r)},
            "instances": r64,
        }
    return {
        "n_instances": len(names),
        "n_cells": len(rows),
        "C_minus_A": block_ca,
        "C_minus_B": block_cb,
        "family_size": 2,
        "t_perturb_over_T_median": {
            "B": float(np.median(t_b)) if t_b else None,
            "C": float(np.median(t_c)) if t_c else None,
        },
        "t_matrix_over_T_median": float(np.median(t_m)) if t_m else None,
        "instances": names,
        "r_ge_64": r64_out,
    }


def share_above_tau(
    ranks_hat: np.ndarray,
    touches: np.ndarray,
    tau: float = ac.PINNED_TAU,
) -> float:
    total = float(np.sum(touches))
    if total <= 0:
        return float("nan")
    return float(np.sum(touches[ranks_hat > tau]) / total)
