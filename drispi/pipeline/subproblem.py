"""Sub-instance construction and parallel per-cluster solving."""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.core.types import Route
from drispi.improvement.bg_ails_budget import lockstep_wave_sum, subcluster_budget_seconds
from drispi.pipeline.cores import affinity_logging_enabled

LOGGER = logging.getLogger(__name__)


class SubclusterSolveError(RuntimeError):
    """Raised when parallel subcluster solving fails (wall timeout or worker error)."""


class SubclusterWallTimeoutError(SubclusterSolveError):
    """Raised when parallel subcluster workers exceed the wall-clock timeout."""


def batch_rounds(n_clusters: int, n_workers: int) -> int:
    """Derived wave count: ``ceil(k / n_workers)`` (observational; scheduler stays implicit)."""
    workers = max(1, n_workers)
    return max(1, int(math.ceil(n_clusters / workers)))


def subcluster_wall_timeout(wave_sum: float) -> float:
    """Hang detector: wall-clock limit over the lockstep wave sum.

    The slack (2x or +120 s, whichever is larger) is applied once over the
    whole predicted lockstep wall, not per wave — the old per-wave scaling
    grew uselessly loose at 10+ waves.
    """
    return max(2.0 * wave_sum, wave_sum + 120.0)


def _dri_worker_initializer(dri_cpus: list[int] | None) -> None:
    """Set OMP_NUM_THREADS=1 and optional affinity (quiet unless DRISPI_LOG_AFFINITY)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if dri_cpus:
        os.sched_setaffinity(0, set(dri_cpus))
    if affinity_logging_enabled():
        try:
            aff = sorted(os.sched_getaffinity(0))
        except AttributeError:
            aff = []
        # print: worker logging may not be configured
        print(f"[dri-worker pid={os.getpid()}] sched_getaffinity(0)={aff}", flush=True)


def make_subinstance(instance: CVRPInstance, cluster: list[int]) -> CVRPInstance:
    """
    Build a sub-CVRP with contiguous internal node IDs 2..len(cluster)+1.

    Customer coordinates and demands are copied from ``cluster`` order; depot
    is node 1. Routes returned by workers are remapped to parent IDs via
    :func:`remap_routes_from_subcluster`.
    """
    if not cluster:
        raise ValueError("cluster must be non-empty")
    coordinates: dict[int, tuple[float, float]] = {1: instance.coordinates[1]}
    demands: dict[int, int] = {1: instance.demands[1]}
    for new_id, old_id in enumerate(cluster, start=2):
        coordinates[new_id] = instance.coordinates[old_id]
        demands[new_id] = instance.demands[old_id]
    customers = list(range(2, len(cluster) + 2))
    return CVRPInstance(
        name=f"{instance.name}_sub_{len(cluster)}",
        n_customers=len(cluster),
        capacity=instance.capacity,
        depot=instance.depot,
        customers=customers,
        coordinates=coordinates,
        demands=demands,
    )


def remap_routes_from_subcluster(routes: list[list[int]], cluster: list[int]) -> list[list[int]]:
    """Map subproblem customer IDs back to parent instance node IDs."""
    old_by_sub = {i + 2: cluster[i] for i in range(len(cluster))}
    return [[old_by_sub[c] for c in route] for route in routes]


def _solve_one_cluster_worker(
    pickled_parent: CVRPInstance,
    cluster: list[int],
    solver_name: str,
    time_limit: float,
    seed: int,
) -> list[list[int]]:
    """Executed in child process: build subinstance, solve, remap routes to parent IDs."""
    sub = make_subinstance(pickled_parent, cluster)
    _ = sub.distance_matrix
    name = solver_name.lower()
    if name == "pyvrp":
        from drispi.solvers.pyvrp_solver import PyVRPSolver

        solver = PyVRPSolver()
    elif name == "filo":
        from drispi.solvers.filo import FiloSolver

        solver = FiloSolver()
    elif name == "filo2":
        from drispi.solvers.filo2 import Filo2Solver

        solver = Filo2Solver()
    elif name == "ails2":
        from drispi.solvers.ails2 import Ails2Solver

        solver = Ails2Solver()
    else:
        raise ValueError(f"Unknown solver_name: {solver_name!r}")

    out: list[SolutionRoute] = solver.solve(sub, time_limit, seed)
    sub_routes = [list(r.customers) for r in out]
    return remap_routes_from_subcluster(sub_routes, cluster)


def solve_subclusters_parallel(
    instance: CVRPInstance,
    partition: list[list[int]],
    solver_name: str,
    rate_s_per_customer: float,
    n_workers: int,
    seed: int,
    *,
    floor_s: float = 5.0,
    dri_cpus: list[int] | None = None,
) -> tuple[list[list[Route]], int]:
    """
    Solve each cluster in parallel; return (routes per cluster, batch_rounds).

    Waves remain implicit in ``ProcessPoolExecutor``; ``batch_rounds`` is derived
    as ``ceil(k / n_workers)`` for instrumentation. The wall timeout is anchored
    to the lockstep wave sum of per-cluster budgets (submit order).
    """
    # Warm the parent matrix in-process for route_cost; CVRPInstance.__getstate__
    # strips it from worker pickles, so this never crosses a process boundary.
    _ = instance.distance_matrix
    if not partition:
        return [], 1

    def budget(cluster: list[int]) -> float:
        return subcluster_budget_seconds(
            len(cluster), rate=rate_s_per_customer, floor=floor_s
        )

    max_workers = max(1, n_workers)
    n_rounds = batch_rounds(len(partition), max_workers)
    budgets = [budget(c) for c in partition]
    max_budget = max(budgets, default=0.0)
    wall_timeout = subcluster_wall_timeout(lockstep_wave_sum(budgets, max_workers))

    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_dri_worker_initializer,
        initargs=(list(dri_cpus) if dri_cpus is not None else None,),
    )
    clean_shutdown = True
    try:
        futures = [
            executor.submit(
                _solve_one_cluster_worker,
                instance,
                cluster,
                solver_name,
                budget(cluster),
                seed + idx,
            )
            for idx, cluster in enumerate(partition)
        ]
        _done, pending = wait(futures, timeout=wall_timeout, return_when=ALL_COMPLETED)
        if pending:
            clean_shutdown = False
            executor.shutdown(wait=False, cancel_futures=True)
            sizes = [len(c) for c in partition]
            raise SubclusterWallTimeoutError(
                f"Subcluster parallel solve exceeded wall timeout {wall_timeout:.0f}s "
                f"(solver={solver_name!r}, cluster_sizes={sizes}, max_budget={max_budget:.1f}s, "
                f"batch_rounds={n_rounds}). "
                "A worker may be hung or ignoring its time limit."
            )
        try:
            return [f.result() for f in futures], n_rounds
        except Exception as exc:
            sizes = [len(c) for c in partition]
            raise SubclusterSolveError(
                f"Subcluster worker failed "
                f"(solver={solver_name!r}, cluster_sizes={sizes}, "
                f"max_budget={max_budget:.1f}s): {exc}"
            ) from exc
    finally:
        if clean_shutdown:
            executor.shutdown(wait=True, cancel_futures=False)
