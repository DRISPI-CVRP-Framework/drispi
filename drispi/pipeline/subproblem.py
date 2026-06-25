"""Sub-instance construction and parallel per-cluster solving."""

from __future__ import annotations

from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.core.types import Route


class SubclusterWallTimeoutError(RuntimeError):
    """Raised when parallel subcluster workers exceed the wall-clock timeout."""


def subcluster_wall_timeout(max_budget: float) -> float:
    """Wall-clock limit for waiting on all subcluster workers (>= per-cluster budgets)."""
    return max(180.0, 2.0 * max_budget)


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
    time_per_customer: float,
    n_workers: int,
    seed: int,
) -> list[list[Route]]:
    """
    Solve each cluster in parallel; return one list of routes per cluster
    (partition order), using parent node IDs.
    """
    _ = instance.distance_matrix
    if not partition:
        return []

    def budget(cluster: list[int]) -> float:
        return max(10.0, float(len(cluster)) * time_per_customer)

    max_workers = max(1, n_workers)
    max_budget = max((budget(c) for c in partition), default=0.0)
    # Parallel wall clock should be ~max_budget; allow model-build / pickle slack.
    wall_timeout = subcluster_wall_timeout(max_budget)

    executor = ProcessPoolExecutor(max_workers=max_workers)
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
                f"(solver={solver_name!r}, cluster_sizes={sizes}, max_budget={max_budget:.1f}s). "
                "A worker may be hung or ignoring its time limit."
            )
        return [f.result() for f in futures]
    finally:
        if clean_shutdown:
            executor.shutdown(wait=True, cancel_futures=False)
