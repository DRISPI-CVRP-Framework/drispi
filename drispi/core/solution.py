"""Route and full solution representations with feasibility stubs."""

from __future__ import annotations

from dataclasses import dataclass

from drispi.core.instance import CVRPInstance


@dataclass
class Route:
    """A single vehicle route as an ordered list of customer visits."""

    customers: list[int]
    cost: float

    def is_feasible(self, instance: CVRPInstance) -> bool:
        """Return whether the route respects capacity and visit rules."""
        # TODO: check capacity vs demands, node membership, depot handling
        raise NotImplementedError

    def total_demand(self, instance: CVRPInstance) -> int:
        """Sum demands of customers on this route."""
        # TODO: sum instance.demands[c] for c in self.customers
        raise NotImplementedError


@dataclass
class Solution:
    """Complete routing solution for one instance."""

    routes: list[Route]
    total_cost: float
    instance_name: str

    def is_feasible(self, instance: CVRPInstance) -> bool:
        """Return whether all routes are feasible and customers are covered correctly."""
        # TODO: aggregate route checks + coverage constraints
        raise NotImplementedError

    def n_routes(self) -> int:
        """Number of routes (vehicles) used."""
        # TODO: len(self.routes)
        raise NotImplementedError
