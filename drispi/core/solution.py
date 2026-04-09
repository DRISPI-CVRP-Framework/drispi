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
        """Return whether the route respects capacity and customer membership."""
        if any(customer not in instance.customers for customer in self.customers):
            return False
        return self.total_demand(instance) <= instance.capacity

    def total_demand(self, instance: CVRPInstance) -> int:
        """Sum demands of customers on this route."""
        return sum(instance.demands[c] for c in self.customers)


@dataclass
class Solution:
    """Complete routing solution for one instance."""

    routes: list[Route]
    total_cost: float
    instance_name: str

    def is_feasible(self, instance: CVRPInstance) -> bool:
        """Return whether all routes are feasible and customers are covered correctly."""
        if any(not route.is_feasible(instance) for route in self.routes):
            return False

        visited: list[int] = [customer for route in self.routes for customer in route.customers]
        expected = set(instance.customers)
        visited_set = set(visited)
        if visited_set != expected:
            return False
        if len(visited) != len(visited_set):
            return False
        return True

    def n_routes(self) -> int:
        """Number of routes (vehicles) used."""
        return len(self.routes)

    def recompute_cost(self, instance: CVRPInstance) -> Solution:
        """Return a new solution with route costs recomputed from instance geometry."""
        _ = instance.distance_matrix
        new_routes: list[Route] = []
        for route in self.routes:
            new_cost = instance.route_cost(route.customers)
            new_routes.append(Route(customers=route.customers, cost=new_cost))
        return Solution(
            routes=new_routes,
            total_cost=sum(route.cost for route in new_routes),
            instance_name=self.instance_name,
        )
