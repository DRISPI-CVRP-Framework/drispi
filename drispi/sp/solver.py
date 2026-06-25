"""Orchestrate one SP/SC phase for a pipeline iteration."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.types import Route
from drispi.route_pool.manager import RoutePoolManager
from drispi.route_pool.pool import RoutePool
from drispi.sp.deduplicate import remove_duplicates
from drispi.sp.model import build_and_solve
from drispi.sp.policy import should_run_sp_sc, should_use_sp


def run_sp_sc(
    pool: RoutePool,
    instance: CVRPInstance,
    manager: RoutePoolManager,
    iteration: int,
    best_solution: list[Route],
    time_limit: float,
    mip_gap: float,
    min_coverage: int,
    warmup_iterations: int,
    sp_interval: int,
) -> tuple[list[Route] | None, bool]:
    """
    Run set partitioning or set covering for one iteration when policy allows.

    LP weights always update pool scores. On MIP time limit with no feasible integer
    solution (``SolCount == 0``), the returned solution falls back to ``best_solution``.
    The route pool is never reset here.

    Returns:
        (None, False) if this iteration skips SP/SC.
        Otherwise ``(solution, use_sp)`` where ``solution`` is ready for improvement
        (SC routes are deduplicated; SP routes are already a partition).
    """
    if not should_run_sp_sc(iteration, warmup_iterations, sp_interval):
        return None, False

    use_sp = should_use_sp(pool, instance, min_coverage)
    lp_weights, raw_solution, timed_out, sol_count = build_and_solve(
        pool,
        instance,
        use_sp,
        time_limit=time_limit,
        mip_gap=mip_gap,
    )

    manager.update_scores_after_solve(pool, lp_weights)

    if timed_out and sol_count == 0:
        base_solution = [list(r) for r in best_solution]
    else:
        base_solution = [list(r) for r in raw_solution]

    if use_sp:
        solution: list[Route] = base_solution
    else:
        solution = remove_duplicates(base_solution, instance)

    return solution, use_sp
