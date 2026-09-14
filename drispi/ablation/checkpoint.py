"""Checkpoint generation: one pinned decompose-route, HAOS and SC/SP off."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from drispi.ablation import config as ac
from drispi.clustering.dissimilarity import compute_dissimilarity_matrix
from drispi.clustering.vertex.agglomerative import cluster as agglomerative_cluster
from drispi.clustering.vertex.kmedoids import cluster as kmedoids_cluster
from drispi.core.instance import CVRPInstance
from drispi.haos.k_domain import k_domain, parse_n_kmin
from drispi.improvement.bg_ails_budget import bg_ails_budget_seconds, predicted_dr_wall_seconds
from drispi.pipeline.subproblem import solve_subclusters_parallel, subcluster_wall_timeout


def instance_vrp_path(instance: CVRPInstance, *aliases: str) -> Path:
    stems = [instance.name, *aliases]
    for directory in CVRPInstance.INSTANCE_SEARCH_DIRS:
        for stem in stems:
            if not stem:
                continue
            for name in (stem, f"{stem}.vrp"):
                candidate = directory / name
                if candidate.is_file():
                    return candidate.resolve()
    raise FileNotFoundError(f"no .vrp for {instance.name} aliases={aliases}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def median_k(instance: CVRPInstance) -> int:
    parsed = parse_n_kmin(instance.name)
    if parsed is None:
        raise ValueError(f"cannot parse n/k_min from {instance.name}")
    n, k_min = parsed
    domain = k_domain(
        n,
        k_min,
        min_routes_per_cluster=8,
        min_arm_spacing=2,
        base_arms=[2, 3, 4, 6, 8, 10, 12],
        max_arms=10,
        ext_per_1000=4,
    )
    return domain[len(domain) // 2]


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ac.ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def checkpoint_path(instance_name: str, seed: int, dest: Path | None = None) -> Path:
    root = dest or ac.CHECKPOINT_DIR
    return root / instance_name / f"seed{seed}.json"


def generate_checkpoint(
    instance_name: str,
    seed: int,
    *,
    dest: Path | None = None,
    n_workers: int = ac.GRANTED_DRI_WORKERS,
) -> dict[str, Any]:
    """Run pinned decompose+route and persist a self-contained checkpoint JSON."""
    os.chdir(ac.ROOT)
    instance = CVRPInstance.from_vrplib(instance_name)
    vrp = instance_vrp_path(instance, instance_name)
    rng = np.random.default_rng(int(seed))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    k = median_k(instance)

    t_matrix0 = time.perf_counter()
    dissim = compute_dissimilarity_matrix(instance, ac.PINNED_LAMBDA_Q, phi)
    t_matrix = time.perf_counter() - t_matrix0

    method_used = ac.PINNED_METHOD
    try:
        partition = kmedoids_cluster(instance, k, dissim, seed=int(seed))
    except Exception:
        # Geometry-aligned fallback: same dissimilarity matrix, average linkage.
        partition = agglomerative_cluster(
            instance, k, dissim, seed=int(seed), linkage="average"
        )
        method_used = "agglomerative_avg"
    cluster_sizes = [len(g) for g in partition]
    t = bg_ails_budget_seconds(
        cluster_sizes,
        n_workers=n_workers,
        rate=ac.SUBCLUSTER_RATE,
        sub_floor=ac.SUBCLUSTER_FLOOR_S,
        floor=ac.BG_BUDGET_FLOOR_S,
        margin=ac.BG_BUDGET_MARGIN,
    )

    predicted_wall = predicted_dr_wall_seconds(
        cluster_sizes,
        n_workers=n_workers,
        rate=ac.SUBCLUSTER_RATE,
        sub_floor=ac.SUBCLUSTER_FLOOR_S,
    )
    wall_timeout = max(
        subcluster_wall_timeout(predicted_wall),
        ac.CHECKPOINT_WALL_FLOOR_S,
    )

    cluster_routes, _n_rounds = solve_subclusters_parallel(
        instance,
        partition,
        ac.PINNED_SOLVER,
        ac.SUBCLUSTER_RATE,
        n_workers,
        int(seed),
        floor_s=ac.SUBCLUSTER_FLOOR_S,
        wall_timeout_s=wall_timeout,
    )
    combined_seqs = [list(r) for cluster in cluster_routes for r in cluster]
    cost = float(sum(instance.route_cost(s) for s in combined_seqs))

    payload: dict[str, Any] = {
        "instance": instance_name,
        "vrplib_name": instance.name,
        "vrp_sha256": sha256_file(vrp),
        "combined_seqs": combined_seqs,
        "cost": cost,
        "partition": [list(g) for g in partition],
        "lambda_q": ac.PINNED_LAMBDA_Q,
        "phi": phi,
        "k": k,
        "cluster_sizes": cluster_sizes,
        "granted_workers": n_workers,
        "rate_s_per_customer": ac.SUBCLUSTER_RATE,
        "floor_s": ac.SUBCLUSTER_FLOOR_S,
        "bg_ails_budget_seconds": t,
        "t_matrix_s": t_matrix,
        "generator_seed": int(seed),
        "pinned": {
            **ac.pinned_config(),
            "method": method_used,
        },
        "config_hash": ac.config_hash(),
        "git_commit": git_commit(),
        "n_customers": instance.n_customers,
    }
    path = checkpoint_path(instance_name, int(seed), dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    payload["_path"] = str(path)
    del dissim
    return payload


def load_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_xl_instances() -> list[str]:
    xl = ac.ROOT / "data/instances/xl"
    names = sorted(p.stem for p in xl.glob("XL-n*.vrp"))
    if not names:
        names = sorted(p.stem for p in xl.glob("X-n*.vrp"))
    return names
