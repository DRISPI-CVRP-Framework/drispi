"""When to run SP/SC and whether the pool supports set partitioning."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.route_pool.coverage import coverage_satisfied
from drispi.route_pool.pool import RoutePool


def should_run_sp_sc(
    iteration: int,
    warmup_iterations: int = 10,
    sp_interval: int = 3,
) -> bool:
    """
    Returns True if SP/SC should run this iteration.

    True when:
      iteration >= warmup_iterations
      AND (iteration - warmup_iterations) % sp_interval == 0
    """
    if iteration < warmup_iterations:
        return False
    return (iteration - warmup_iterations) % sp_interval == 0


def should_use_sp(
    pool: RoutePool,
    instance: CVRPInstance,
    min_coverage: int = 1,
) -> bool:
    """
    Returns True if every customer is covered at least ``min_coverage`` times in the pool.

    When False, fall back to set covering. Delegates to ``coverage_satisfied``.
    """
    return coverage_satisfied(pool, instance, min_coverage)
