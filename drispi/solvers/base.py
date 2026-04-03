"""Abstract solver interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool

SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {}

S = TypeVar("S", bound="BaseSolver")


def register_solver(name: str) -> Callable[[type[S]], type[S]]:
    """Decorator to register a solver implementation under ``name``."""

    def _decorator(cls: type[S]) -> type[S]:
        SOLVER_REGISTRY[name] = cls
        return cls

    return _decorator


class BaseSolver(ABC):
    """Base class for routing solvers on a customer subset."""

    @abstractmethod
    def solve(self, instance: CVRPInstance, customers: list[int], time_limit: float) -> RoutePool:
        """Return a pool of routes visiting ``customers`` within ``time_limit`` seconds."""
        raise NotImplementedError
