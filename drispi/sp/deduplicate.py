"""Savings-based duplicate removal after set covering."""

from __future__ import annotations

import copy

import numpy as np
import numpy.typing as npt

from drispi.core.instance import CVRPInstance
from drispi.core.types import Route


def _removal_savings(
    customer: int,
    route: Route,
    distance_matrix: npt.NDArray[np.floating],
) -> float:
    """Cost reduction from removing ``customer`` from this ordered ``route``."""
    idx = route.index(customer)
    pred = route[idx - 1] if idx > 0 else 1
    succ = route[idx + 1] if idx + 1 < len(route) else 1
    return float(
        distance_matrix[pred, customer]
        + distance_matrix[customer, succ]
        - distance_matrix[pred, succ]
    )


def remove_duplicates(
    solution: list[Route],
    instance: CVRPInstance,
) -> list[Route]:
    """
    Remove customers appearing in more than one route using a savings heuristic.

    Keeps each duplicated customer on the route where removal savings are lowest
    (removing it there saves the least cost); removes it from all other routes.
    Empty routes are dropped. Customer order within each route is preserved aside
    from removals.
    """
    routes = copy.deepcopy(solution)
    dm = instance.distance_matrix

    def duplicate_customers() -> list[int]:
        counts: dict[int, int] = {}
        for route in routes:
            for c in route:
                counts[c] = counts.get(c, 0) + 1
        return sorted(c for c, n in counts.items() if n > 1)

    while True:
        dups = duplicate_customers()
        if not dups:
            break

        for customer in dups:
            containing = [r for r in routes if customer in r]
            if len(containing) <= 1:
                continue

            savings_by_route: list[tuple[float, Route]] = []
            for r in containing:
                savings_by_route.append((_removal_savings(customer, r, dm), r))

            keep_route = min(savings_by_route, key=lambda t: t[0])[1]

            for r in containing:
                if r is keep_route:
                    continue
                if customer in r:
                    r.remove(customer)

        routes = [r for r in routes if len(r) > 0]

    return routes
