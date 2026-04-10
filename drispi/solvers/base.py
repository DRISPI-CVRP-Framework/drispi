"""Abstract solver interface for full-instance routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance
    from drispi.core.types import RoutePool


class BaseSolver(ABC):
    """Abstract base class for CVRP solvers operating on a full instance."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this solver (e.g. ``\"pyvrp\"``)."""
        ...

    @abstractmethod
    def solve(
        self,
        instance: CVRPInstance,
        time_limit: float,
        seed: int = 42,
    ) -> RoutePool:
        """
        Solve ``instance`` within ``time_limit`` seconds.

        Returns:
            ``RoutePool`` of :class:`~drispi.core.solution.Route` objects (depot
            excluded; customer IDs use internal 1-based VRPLIB node indexing).

        Raises:
            RuntimeError: If the solver fails to produce a valid solution.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
