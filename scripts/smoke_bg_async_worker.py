#!/usr/bin/env python3
"""Smoke test: real AsyncBgAilsController round trip (perturb + AILS-II JVM)."""

from __future__ import annotations

import logging
import time

from drispi.core.instance import CVRPInstance
from drispi.haos.tag import HAOSTag
from drispi.pipeline.bg_ails_async import AsyncBgAilsController

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_instance(n: int = 20) -> CVRPInstance:
    customers = list(range(2, 2 + n))
    coordinates = {1: (0.0, 0.0)}
    demands = {1: 0}
    for idx, cid in enumerate(customers):
        coordinates[cid] = (float(idx % 5) * 3.0, float(idx // 5) * 3.0)
        demands[cid] = 10
    return CVRPInstance(
        name="smoke20",
        n_customers=n,
        capacity=50,
        depot=(0.0, 0.0),
        customers=customers,
        coordinates=coordinates,
        demands=demands,
    )


def main() -> None:
    instance = build_instance()
    customers = list(instance.customers)
    partition = [customers[:10], customers[10:]]
    combined_seqs = [customers[i : i + 4] for i in range(0, len(customers), 4)]

    ctrl = AsyncBgAilsController(bg_cpus=None, xmx="1g")
    ctrl.start(instance)
    tag = HAOSTag(
        k=2, lambda_demand=0.0, paradigm="vertex", method="kmeans",
        solver="ails2", iteration=0,
    )
    job_id = ctrl.launch(
        launch_iteration=0,
        combined_seqs=combined_seqs,
        partition=partition,
        lambda_demand=0.4,
        angular_offset=1.0,
        initial_omega=0.8,
        time_limit=5.0,
        seed=42,
        boundary_threshold=0.5,
        small_cluster_cap=20,
        small_cluster_alpha=0.5,
        pair_selection="greedy",
        n_chains_mode="k",
        op_tag=tag,
    )
    print(f"launched job_id={job_id}, blocking on result...")
    t0 = time.perf_counter()
    result = ctrl.wait_result()
    wall = time.perf_counter() - t0
    assert result is not None
    print(
        f"result after {wall:.1f}s: reason={result.reason} "
        f"cost_before={result.cost_before:.1f} bg_cost={result.bg_cost} "
        f"n_routes={len(result.bg_seqs) if result.bg_seqs else None} "
        f"perturb={result.perturb_wall_s:.2f}s improve={result.improve_wall_s:.2f}s"
    )
    # consume-once: second drain must be empty
    assert ctrl.poll_result() is None
    ctrl.shutdown()
    assert result.reason == "ok", result.error
    print("smoke OK")


if __name__ == "__main__":
    main()
