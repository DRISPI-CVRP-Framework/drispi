"""MIP solver wrapper for set partitioning / covering."""

from __future__ import annotations

from typing import Any

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Solution
from drispi.route_pool.pool import RoutePool


def solve_mip(
    pool: RoutePool,
    instance: CVRPInstance,
    time_limit: float,
    mode: str,
) -> Solution | None:
    """Solve integer SP/SC on ``pool`` within ``time_limit``."""
    # TODO: build_model, set params, optimize, construct Solution
    raise NotImplementedError
