"""PyVRP-based solver (in-process, no subprocess)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drispi.core.solution import Route
from drispi.solvers.base import BaseSolver

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance
    from drispi.core.types import RoutePool


class PyVRPSolver(BaseSolver):
    """CVRP solver using the ``pyvrp`` Python package."""

    name = "pyvrp"

    def __init__(self) -> None:
        """
        Validate that ``pyvrp`` is importable.

        Raises:
            ImportError: If ``pyvrp`` is not installed, with install hints.
        """
        try:
            import pyvrp  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "PyVRPSolver requires the 'pyvrp' package. Install with: "
                "`pip install pyvrp` or `uv sync` from the project root."
            ) from exc

    def solve(
        self,
        instance: CVRPInstance,
        time_limit: float,
        seed: int = 42,
    ) -> RoutePool:
        """
        Build a :class:`pyvrp.Model` from ``instance``, solve, and return routes.

        The model uses the depot and client coordinates from ``instance``,
        demands from :py:attr:`~drispi.core.instance.CVRPInstance.demands`,
        ``service_duration=0``, vehicle type with ``num_available`` equal to
        ``n_customers`` and capacity from the instance. Edge weights are taken
        from :py:attr:`~drispi.core.instance.CVRPInstance.distance_matrix`,
        rounded to integers.

        Raises:
            RuntimeError: If the best solution is not feasible.
        """
        from pyvrp import Model
        from pyvrp.stop import MaxRuntime

        model = Model()
        dx, dy = instance.depot
        depot = model.add_depot(int(round(dx)), int(round(dy)))

        clients: list[object] = []
        for node_id in instance.customers:
            x, y = instance.coordinates[node_id]
            demand = instance.demands[node_id]
            clients.append(
                model.add_client(
                    int(round(x)),
                    int(round(y)),
                    delivery=int(demand),
                    service_duration=0,
                )
            )

        locations = [depot, *clients]
        node_ids = [1, *instance.customers]
        dm = instance.distance_matrix
        n = instance.n_customers + 1
        for i in range(n):
            for j in range(n):
                ni, nj = node_ids[i], node_ids[j]
                dist = int(round(float(dm[ni, nj])))
                model.add_edge(locations[i], locations[j], distance=dist)

        model.add_vehicle_type(
            num_available=instance.n_customers,
            capacity=int(instance.capacity),
        )

        result = model.solve(
            MaxRuntime(float(time_limit)),
            seed=int(seed),
            display=False,
            collect_stats=False,
        )

        if not result.is_feasible():
            raise RuntimeError("PyVRP did not find a feasible solution.")

        best = result.best
        pool: RoutePool = []
        for route in best.routes():
            visits = list(route.visits())
            internal_customers = [instance.customers[v - 1] for v in visits]
            pool.append(
                Route(
                    customers=internal_customers,
                    cost=instance.route_cost(internal_customers),
                )
            )
        return pool
