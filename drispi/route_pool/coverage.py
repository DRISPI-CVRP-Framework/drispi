"""Coverage counting and feasibility helpers for route pools."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.route_pool.pool import RoutePool


def coverage_counts(pool: RoutePool, instance: CVRPInstance) -> dict[int, int]:
    """Map each customer node ID to how many routes in ``pool`` visit it."""
    # TODO: aggregate from pool.covers or direct iteration
    raise NotImplementedError


def uncovered_customers(pool: RoutePool, instance: CVRPInstance, min_coverage: int) -> list[int]:
    """Customers with coverage count strictly below ``min_coverage``."""
    # TODO: filter coverage_counts
    raise NotImplementedError


def coverage_satisfied(pool: RoutePool, instance: CVRPInstance, min_coverage: int) -> bool:
    """Return whether every customer meets the minimum coverage requirement."""
    # TODO: all(c >= min_coverage for c in coverage_counts.values())
    raise NotImplementedError
