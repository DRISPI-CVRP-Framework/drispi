"""Mutable collection of candidate routes."""

from __future__ import annotations

from drispi.core.solution import Route


class RoutePool:
    """Stores heterogeneous routes discovered during search."""

    routes: list[Route]

    def __init__(self, routes: list[Route] | None = None) -> None:
        self.routes = list(routes) if routes is not None else []

    def add(self, routes: list[Route]) -> None:
        """Append routes to the pool."""
        # TODO: extend self.routes, optional deduplication
        raise NotImplementedError

    def remove(self, indices: list[int]) -> None:
        """Remove routes at given indices (stable semantics TBD)."""
        # TODO: delete by sorted indices descending to preserve positions
        raise NotImplementedError

    def size(self) -> int:
        """Number of routes currently stored."""
        # TODO: return len(self.routes)
        raise NotImplementedError

    def covers(self, customer: int) -> list[int]:
        """Indices of routes that visit ``customer``."""
        # TODO: scan routes for membership
        raise NotImplementedError
