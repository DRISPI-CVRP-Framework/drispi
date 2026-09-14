"""Three-arm runner: identical -limit=T, perturb-once, stock vs instrumented jars."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from drispi.ablation import config as ac
from drispi.ablation.checkpoint import load_checkpoint
from drispi.ablation.sanity import assert_touch_covers_modifications, modified_customers
from drispi.ablation.touch import customer_touch_maps, parse_touch_tsv
from drispi.clustering.dissimilarity import compute_dissimilarity_matrix
from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route
from drispi.improvement.bg_ails import (
    run_blind_perturb,
    run_guided_perturb_traced,
)
from drispi.solvers.ails2 import Ails2Solver
from drispi.utils.io import write_sol

Arm = Literal["A", "B", "C"]
Campaign = Literal["performance", "measurement"]


def _seqs_to_routes(instance: CVRPInstance, seqs: list[list[int]]) -> list[Route]:
    return [Route(customers=list(s), cost=instance.route_cost(s)) for s in seqs]


def _jar_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_dir(base: Path, instance: str, seed: int) -> Path:
    return base / instance / f"seed{seed}"


def perturb_and_persist(
    checkpoint: dict[str, Any],
    instance: CVRPInstance,
    dest: Path,
) -> dict[str, Any]:
    """Perturb B and C once; persist .sol files, traces, and timings."""
    dest.mkdir(parents=True, exist_ok=True)
    seqs = [list(s) for s in checkpoint["combined_seqs"]]
    partition = [list(g) for g in checkpoint["partition"]]
    seed = int(checkpoint["generator_seed"])
    routes = _seqs_to_routes(instance, seqs)

    t0 = time.perf_counter()
    dissim = compute_dissimilarity_matrix(
        instance, float(checkpoint["lambda_q"]), float(checkpoint["phi"])
    )
    t_matrix = time.perf_counter() - t0

    t1 = time.perf_counter()
    blind, blind_idx, blind_trace = run_blind_perturb(
        instance,
        routes,
        partition,
        seed=seed,
        n_chains_mode=ac.PINNED_N_CHAINS_MODE,  # type: ignore[arg-type]
    )
    t_perturb_b = time.perf_counter() - t1

    t2 = time.perf_counter()
    guided, guided_idx, guided_trace, ranks = run_guided_perturb_traced(
        instance,
        routes,
        dissim,
        partition,
        boundary_threshold=ac.PINNED_TAU,
        small_cluster_cap=ac.PINNED_SMALL_CLUSTER_CAP,
        small_cluster_alpha=ac.PINNED_SMALL_CLUSTER_ALPHA,
        seed=seed,
        pair_selection=ac.PINNED_PAIR_SELECTION,  # type: ignore[arg-type]
        n_chains_mode=ac.PINNED_N_CHAINS_MODE,  # type: ignore[arg-type]
    )
    t_perturb_c = time.perf_counter() - t2

    a_path = dest / "armA_init.sol"
    b_path = dest / "armB_init.sol"
    c_path = dest / "armC_init.sol"
    cost_a = float(checkpoint["cost"])
    cost_b = float(sum(r.cost for r in blind))
    cost_c = float(sum(r.cost for r in guided))
    write_sol(seqs, cost_a, a_path)
    write_sol([r.customers for r in blind], cost_b, b_path)
    write_sol([r.customers for r in guided], cost_c, c_path)

    meta = {
        "t_matrix_s": t_matrix,
        "t_perturb_B_s": t_perturb_b,
        "t_perturb_C_s": t_perturb_c,
        "blind_indices": blind_idx,
        "guided_indices": guided_idx,
        "blind_trace": blind_trace,
        "guided_trace": guided_trace,
        "ranks_hat": ranks.tolist(),
        "customers": list(instance.customers),
        "cost_input": cost_a,
        "cost_after_perturb_B": cost_b,
        "cost_after_perturb_C": cost_c,
        "init_sol": {"A": str(a_path), "B": str(b_path), "C": str(c_path)},
    }
    (dest / "perturb.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def run_arm(
    checkpoint: dict[str, Any],
    instance: CVRPInstance,
    arm: Arm,
    *,
    campaign: Campaign,
    work_dir: Path,
    perturb_meta: dict[str, Any],
    time_limit: float | None = None,
) -> dict[str, Any]:
    """One AILS-II call. Performance uses the stock jar; measurement the touch jar."""
    t_limit = float(time_limit if time_limit is not None else checkpoint["bg_ails_budget_seconds"])
    init_sol = Path(perturb_meta["init_sol"][arm])
    work_dir.mkdir(parents=True, exist_ok=True)
    out_sol = work_dir / f"arm{arm}_{campaign}.sol"
    jar = ac.STOCK_JAR if campaign == "performance" else ac.TOUCH_JAR
    omega = None if arm == "A" else ac.PINNED_OMEGA_BC
    solver = Ails2Solver(binary_path=jar, active_processor_count=1)
    touch_path = work_dir / f"arm{arm}_{campaign}.touch.tsv" if campaign == "measurement" else None

    t0 = time.perf_counter()
    routes = solver.run_improvement(
        instance,
        t_limit,
        init_sol,
        initial_omega=omega,
        seed=int(checkpoint["generator_seed"]),
        touch_dump_path=touch_path,
    )
    wall = time.perf_counter() - t0
    out_seqs = [list(r.customers) for r in routes]
    out_cost = float(sum(instance.route_cost(s) for s in out_seqs))
    write_sol(out_seqs, out_cost, out_sol)

    from drispi.utils.io import read_sol

    in_seqs, _ = read_sol(init_sol)
    changed = modified_customers(in_seqs, out_seqs)
    result: dict[str, Any] = {
        "arm": arm,
        "campaign": campaign,
        "instance": checkpoint["instance"],
        "generator_seed": checkpoint["generator_seed"],
        "time_limit": t_limit,
        "initial_omega": omega,
        "jar": str(jar),
        "jar_sha256": _jar_sha256(jar),
        "cost_in": float(sum(instance.route_cost(s) for s in in_seqs)),
        "cost_out": out_cost,
        "pct_improvement": 100.0
        * (float(sum(instance.route_cost(s) for s in in_seqs)) - out_cost)
        / float(sum(instance.route_cost(s) for s in in_seqs)),
        "jvm_wall_s": wall,
        "out_sol": str(out_sol),
        "n_changed_customers": len(changed),
    }
    if campaign == "measurement":
        assert touch_path is not None
        dump = parse_touch_tsv(touch_path)
        maps = customer_touch_maps(dump, instance)
        result["touch_path"] = str(touch_path)
        result["ils_iterator"] = dump["header"].get("iterator")
        result["touch_elapsed_s"] = dump["header"].get("elapsed_s")
        result["eval_on"] = dump["header"].get("eval_on")
        assert_touch_covers_modifications(changed, maps["accept"], maps["perturb"])
        result["sanity_ok"] = True
        # Persist compact per-customer arrays aligned to instance.customers
        customers = list(instance.customers)
        result["eval"] = [maps["eval"].get(c, 0) for c in customers]
        result["accept"] = [maps["accept"].get(c, 0) for c in customers]
        result["perturb"] = [maps["perturb"].get(c, 0) for c in customers]
    (work_dir / f"arm{arm}_{campaign}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    return result


def _arm_json(dest: Path, arm: Arm, campaign: Campaign) -> Path:
    return dest / f"arm{arm}_{campaign}.json"


def cell_is_complete(
    dest: Path, campaign: Campaign, arms: tuple[Arm, ...]
) -> bool:
    summary = dest / f"{campaign}_summary.json"
    if not summary.is_file():
        return False
    return all(_arm_json(dest, arm, campaign).is_file() for arm in arms)


def run_cell(
    checkpoint_path: Path,
    *,
    campaign: Campaign,
    out_root: Path | None = None,
    time_limit: float | None = None,
    arms: tuple[Arm, ...] = ("A", "B", "C"),
    force: bool = False,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    os.chdir(ac.ROOT)
    instance = CVRPInstance.from_vrplib(checkpoint["instance"])
    base = out_root or (ac.PERF_DIR if campaign == "performance" else ac.MEAS_DIR)
    dest = cell_dir(base, checkpoint["instance"], int(checkpoint["generator_seed"]))
    dest.mkdir(parents=True, exist_ok=True)
    if not force and cell_is_complete(dest, campaign, arms):
        return json.loads((dest / f"{campaign}_summary.json").read_text(encoding="utf-8"))
    perturb_path = dest / "perturb.json"
    if perturb_path.is_file() and not force:
        perturb_meta = json.loads(perturb_path.read_text(encoding="utf-8"))
    else:
        perturb_meta = perturb_and_persist(checkpoint, instance, dest)
    arm_results: dict[str, Any] = {}
    for arm in arms:
        existing = _arm_json(dest, arm, campaign)
        if existing.is_file() and not force:
            arm_results[arm] = json.loads(existing.read_text(encoding="utf-8"))
            continue
        arm_results[arm] = run_arm(
            checkpoint,
            instance,
            arm,
            campaign=campaign,
            work_dir=dest,
            perturb_meta=perturb_meta,
            time_limit=time_limit,
        )
    summary = {
        "instance": checkpoint["instance"],
        "generator_seed": checkpoint["generator_seed"],
        "campaign": campaign,
        "T": checkpoint["bg_ails_budget_seconds"],
        "t_perturb_B_s": perturb_meta["t_perturb_B_s"],
        "t_perturb_C_s": perturb_meta["t_perturb_C_s"],
        "t_matrix_s": perturb_meta["t_matrix_s"],
        "arms": arm_results,
    }
    (dest / f"{campaign}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def cell_job(
    args: tuple[str, str, float | None, tuple[str, ...], bool],
) -> str:
    """Picklable worker for ProcessPoolExecutor (spawn-safe)."""
    os.chdir(ac.ROOT)
    ckpt, campaign, time_limit, arms, force = args
    run_cell(
        Path(ckpt),
        campaign=campaign,  # type: ignore[arg-type]
        time_limit=time_limit,
        arms=tuple(arms),  # type: ignore[arg-type]
        force=force,
    )
    return ckpt
