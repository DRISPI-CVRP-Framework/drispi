#!/usr/bin/env python3
"""Analyse performance runs and emit figure JSON contracts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from drispi.ablation import config as ac  # noqa: E402
from drispi.ablation.checkpoint import load_checkpoint  # noqa: E402
from drispi.ablation.stats import analyse_performance, share_above_tau  # noqa: E402
from drispi.core.instance import CVRPInstance  # noqa: E402


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_figure3(meas_root: Path, dest: Path) -> dict:
    """Normalize within checkpoint, average checkpoints per instance, then instances."""
    inst_deciles: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"A": [], "B": [], "C": []}
    )
    inst_shares: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"A": [], "B": [], "C": []}
    )
    inst_tau: dict[str, list[float]] = defaultdict(list)
    for summary_path in sorted(meas_root.glob("*/*/measurement_summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        inst = row["instance"]
        pert = json.loads((summary_path.parent / "perturb.json").read_text(encoding="utf-8"))
        ranks = np.array(pert["ranks_hat"], dtype=np.float64)
        order = np.argsort(ranks)
        n = len(ranks)
        if n < 10:
            continue
        inst_tau[inst].append(float(np.mean(ranks <= ac.PINNED_TAU) * 10.0))
        for arm in ("A", "B", "C"):
            arm_json = summary_path.parent / f"arm{arm}_measurement.json"
            rec = json.loads(arm_json.read_text(encoding="utf-8"))
            touches = np.array(rec["eval"], dtype=np.float64)
            total = touches.sum()
            if total <= 0:
                continue
            inst_shares[inst][arm].append(share_above_tau(ranks, touches))
            share = np.zeros(10, dtype=np.float64)
            ranked_t = touches[order]
            edges = np.linspace(0, n, 11, dtype=int)
            for d in range(10):
                share[d] = ranked_t[edges[d] : edges[d + 1]].sum() / total
            inst_deciles[inst][arm].append(share)
    per_arm: dict[str, list[np.ndarray]] = {"A": [], "B": [], "C": []}
    shares = {"A": [], "B": [], "C": []}
    tau_quantiles: list[float] = []
    for inst, arms in inst_deciles.items():
        tau_quantiles.append(float(np.mean(inst_tau[inst])))
        for arm in ("A", "B", "C"):
            if arms[arm]:
                per_arm[arm].append(np.mean(np.vstack(arms[arm]), axis=0))
            if inst_shares[inst][arm]:
                shares[arm].append(float(np.mean(inst_shares[inst][arm])))
    rng = np.random.default_rng(0)
    out = {
        "mean_share": {},
        "ci_lo": {},
        "ci_hi": {},
        "share_above_tau": {},
        "tau_decile_position": float(np.mean(tau_quantiles)) if tau_quantiles else 5.5,
        "aggregation": "normalize within instance, then average across instances",
        "n_instances": len(inst_deciles),
    }
    for arm in ("A", "B", "C"):
        mat = np.vstack(per_arm[arm]) if per_arm[arm] else np.zeros((1, 10))
        out["mean_share"][arm] = mat.mean(axis=0).tolist()
        lo, hi = [], []
        for d in range(10):
            a, b = _bootstrap_ci(mat[:, d], rng)
            lo.append(a)
            hi.append(b)
        out["ci_lo"][arm] = lo
        out["ci_hi"][arm] = hi
        out["share_above_tau"][arm] = float(np.nanmean(shares[arm])) if shares[arm] else None
    dest.write_text(json.dumps(out), encoding="utf-8")
    return out


def write_figures_1_2(dest_dir: Path, instance: str, seed: int) -> None:
    meas = ac.MEAS_DIR / instance / f"seed{seed}"
    ckpt = load_checkpoint(ac.CHECKPOINT_DIR / instance / f"seed{seed}.json")
    pert = json.loads((meas / "perturb.json").read_text(encoding="utf-8"))
    inst = CVRPInstance.from_vrplib(instance)
    labels = {}
    for k, group in enumerate(ckpt["partition"]):
        for c in group:
            labels[c] = k
    coords = {str(c): list(inst.coordinates[c]) for c in inst.customers}
    rec_c = json.loads((meas / "armC_measurement.json").read_text(encoding="utf-8"))
    rec_a = json.loads((meas / "armA_measurement.json").read_text(encoding="utf-8"))
    from drispi.utils.io import read_sol

    pert_c, _ = read_sol(Path(pert["init_sol"]["C"]))
    result_seqs, _ = read_sol(Path(rec_c["out_sol"]))
    fig1 = {
        "instance": instance,
        "generator_seed": seed,
        "T": ckpt["bg_ails_budget_seconds"],
        "depot": list(inst.depot),
        "customers": list(inst.customers),
        "coordinates": coords,
        "labels": {str(k): v for k, v in labels.items()},
        "input_seqs": ckpt["combined_seqs"],
        "perturbed_seqs": pert_c,
        "guided_trace": pert["guided_trace"],
        "ranks_hat": pert["ranks_hat"],
        "eval_touches": rec_c["eval"],
        "result_seqs": result_seqs,
        "jar_sha256_measurement": rec_c["jar_sha256"],
    }
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "figure1_mechanism.json").write_text(json.dumps(fig1), encoding="utf-8")
    fig2 = {
        "instance": instance,
        "generator_seed": seed,
        "T": ckpt["bg_ails_budget_seconds"],
        "depot": list(inst.depot),
        "customers": list(inst.customers),
        "coordinates": coords,
        "labels": {str(k): v for k, v in labels.items()},
        "eval_A": rec_a["eval"],
        "eval_C": rec_c["eval"],
        "jar_sha256_measurement": rec_c["jar_sha256"],
        "counter": "eval",
        "normalization": "shared LogNorm across panels",
    }
    (dest_dir / "figure2_touchmaps.json").write_text(json.dumps(fig2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--figure-instance", default="XL-n1281-k29")
    p.add_argument("--figure-seed", type=int, default=101)
    args = p.parse_args()
    ac.FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats = analyse_performance()
    (ac.FIGURE_DATA_DIR / "performance_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    if ac.MEAS_DIR.exists():
        fig3 = write_figure3(ac.MEAS_DIR, ac.FIGURE_DATA_DIR / "figure3_concentration.json")
        print("figure3 share_above_tau", fig3["share_above_tau"])
        fig_dir = ac.MEAS_DIR / args.figure_instance / f"seed{args.figure_seed}"
        if fig_dir.is_dir():
            write_figures_1_2(ac.FIGURE_DATA_DIR, args.figure_instance, args.figure_seed)
            print("wrote figure1/2 JSON")


if __name__ == "__main__":
    main()
