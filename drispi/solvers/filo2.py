"""FILO2 subprocess-based solver stub."""

from __future__ import annotations

from pathlib import Path

from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool
from drispi.solvers.base import BaseSolver, register_solver


def _resolve_executable(variant: str) -> Path:
    """Locate compiled FILO2 binary under ``vendor/`` for the given variant name."""
    # TODO: walk vendor/filo2 for executable
    raise NotImplementedError


def _write_subinstance_vrp(
    instance: CVRPInstance,
    customers: list[int],
    path: Path,
) -> tuple[dict[int, int], dict[int, int]]:
    """Write a temporary ``.vrp`` for the sub-instance; return global/local ID maps."""
    # TODO: same as filo module, possibly different format flags
    raise NotImplementedError


def _parse_routes_from_text(text: str) -> list[list[int]]:
    """Parse ``Route #N: ...`` lines from solver stdout."""
    # TODO: align with FILO2 output format
    raise NotImplementedError


def _run_executable(exe: Path, vrp_path: Path, extra_args: list[str]) -> tuple[str, float]:
    """Run subprocess; return stdout text and runtime seconds."""
    # TODO: subprocess.run wrapper
    raise NotImplementedError


@register_solver("filo2")
class Filo2Solver(BaseSolver):
    """FILO2 external solver integration."""

    def solve(self, instance: CVRPInstance, customers: list[int], time_limit: float) -> RoutePool:
        # TODO: wire helpers for filo2 binary and args
        raise NotImplementedError
