"""Gurobi SP/SC formulation builders and LP relaxation stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from drispi.core.instance import CVRPInstance
from drispi.route_pool.pool import RoutePool


@dataclass
class LPResult:
    """Outcome of solving the LP relaxation of the route pool model."""

    lp_values: dict[int, float]
    reduced_costs: dict[int, float]
    duals: dict[int, float]


def build_model(
    pool: RoutePool,
    instance: CVRPInstance,
    mode: Literal["SP", "SC"],
) -> tuple[Any, Any]:
    """Build a Gurobi model and variable container for SP or SC over ``pool``.

    Returns:
        Tuple of ``(model, vars)`` where ``vars`` maps column indices to Gurobi variables.

    Raises:
        NotImplementedError: Stub only.
    """
    # TODO: add binary vars per route column, coverage constraints, objective
    raise NotImplementedError


def solve_lp_relaxation(pool: RoutePool, instance: CVRPInstance) -> LPResult | None:
    """Solve LP relaxation; return duals and reduced costs, or ``None`` if failed."""
    # TODO: relax integrality, optimize, extract pi and rc
    raise NotImplementedError
