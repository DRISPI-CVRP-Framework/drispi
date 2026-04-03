"""PyVRP-based HGS solver stub."""

from __future__ import annotations

from typing import Any

from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool
from drispi.solvers.base import BaseSolver, register_solver


def _to_pyvrp_model(instance: CVRPInstance, customers: list[int]) -> Any:
    """Build a PyVRP ``Model`` (or equivalent) restricted to ``customers``."""
    # TODO: map coordinates, demands, capacity to pyvrp.Model
    raise NotImplementedError


def _from_pyvrp_result(result: Any) -> RoutePool:
    """Convert PyVRP solution object to internal ``RoutePool``."""
    # TODO: extract routes and costs from result
    raise NotImplementedError


@register_solver("hgs")
class HgsSolver(BaseSolver):
    """Hybrid Genetic Search via PyVRP."""

    def solve(self, instance: CVRPInstance, customers: list[int], time_limit: float) -> RoutePool:
        # TODO: _to_pyvrp_model, run solver with time_limit, _from_pyvrp_result
        raise NotImplementedError
