"""Benchmark and solution quality metrics."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Solution


def primal_dual_gap(solution: Solution, lp_bound: float) -> float:
    """Relative gap between ``solution.total_cost`` and ``lp_bound``."""
    # TODO: (cost - bound) / bound or project convention
    raise NotImplementedError


def route_lengths(instance: CVRPInstance, solution: Solution) -> list[float]:
    """Per-route Euclidean length under ``instance`` coordinates."""
    # TODO: sum distances along each route
    raise NotImplementedError
