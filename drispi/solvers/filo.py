"""FILO subprocess-based solver stub."""

from __future__ import annotations

from pathlib import Path

from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool
from drispi.solvers.base import BaseSolver, register_solver


def _resolve_executable(variant: str) -> Path:
    """Locate compiled FILO binary under ``vendor/`` for the given variant name."""
    # TODO: walk vendor/ (e.g. vendor/filo) for executable matching variant
    raise NotImplementedError


def _write_subinstance_vrp(
    instance: CVRPInstance,
    customers: list[int],
    path: Path,
) -> tuple[dict[int, int], dict[int, int]]:
    """Write a temporary ``.vrp`` for the sub-instance; return global/local ID maps.

    Returns:
        global_to_local: original node ID -> local index used in file.
        local_to_global: local index -> original node ID.
    """
    # TODO: renumber nodes, emit VRPLIB, return bidirectional maps
    raise NotImplementedError


def _parse_routes_from_text(text: str) -> list[list[int]]:
    """Parse ``Route #N: ...`` lines from solver stdout into global customer sequences."""
    # TODO: regex or line scanner for route blocks
    raise NotImplementedError


def _run_executable(exe: Path, vrp_path: Path, extra_args: list[str]) -> tuple[str, float]:
    """Run ``exe`` on ``vrp_path``; return captured stdout and wall-clock runtime in seconds."""
    # TODO: subprocess.run with timeout, capture_output
    raise NotImplementedError


@register_solver("filo")
class FiloSolver(BaseSolver):
    """FILO external solver integration."""

    def solve(self, instance: CVRPInstance, customers: list[int], time_limit: float) -> RoutePool:
        # TODO: _resolve_executable("filo"), write vrp, _run_executable, parse to Route objects
        raise NotImplementedError
