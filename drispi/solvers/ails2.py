"""AILS2 external solver stub (subprocess or API TBD)."""

from __future__ import annotations

from pathlib import Path

from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool
from drispi.solvers.base import BaseSolver, register_solver


def _resolve_ails2_executable() -> Path:
    """Locate AILS2 binary under ``vendor/ails2``."""
    # TODO: walk vendor/ails2
    raise NotImplementedError


@register_solver("ails2")
class Ails2Solver(BaseSolver):
    """AILS2 integration placeholder."""

    def solve(self, instance: CVRPInstance, customers: list[int], time_limit: float) -> RoutePool:
        # TODO: mirror filo-style subprocess or native bindings
        raise NotImplementedError
