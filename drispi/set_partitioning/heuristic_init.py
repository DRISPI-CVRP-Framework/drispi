"""Heuristic initial columns and warm-start for set partitioning."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.route_pool.pool import RoutePool


def greedy_initial_columns(instance: CVRPInstance, pool: RoutePool) -> None:
    """Populate ``pool`` with simple feasible routes as starting columns."""
    # TODO: nearest-neighbor or savings columns
    raise NotImplementedError
