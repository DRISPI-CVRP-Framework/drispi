"""Coverage counting and feasibility helpers for route pools."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.types import Route
from drispi.route_pool.pool import RoutePool


def coverage_counts(pool: RoutePool, instance: CVRPInstance) -> dict[int, int]:
    """
    Return customer coverage counts for all customer node IDs.

    Customers with zero coverage are included with value ``0``.
    """
    counts = {customer: 0 for customer in instance.customers}
    for entry in pool:
        for customer in entry.customer_set:
            if customer in counts:
                counts[customer] += 1
    return counts


def uncovered_customers(pool: RoutePool, instance: CVRPInstance) -> list[int]:
    """Return customer node IDs not covered by any route in the pool."""
    counts = coverage_counts(pool, instance)
    return [customer for customer, count in counts.items() if count == 0]


def coverage_satisfied(
    pool: RoutePool,
    instance: CVRPInstance,
    min_coverage: int = 1,
) -> bool:
    """Return whether every customer meets the minimum coverage requirement."""
    counts = coverage_counts(pool, instance)
    return all(count >= min_coverage for count in counts.values())


def routes_covering(pool: RoutePool, customer: int) -> list[Route]:
    """Return all routes in the pool that include ``customer``."""
    return [list(entry.route) for entry in pool if customer in entry.customer_set]
