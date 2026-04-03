"""High-level route pool lifecycle (eviction, limits, statistics)."""

from __future__ import annotations

from drispi.route_pool.pool import RoutePool


class RoutePoolManager:
    """Applies policies for pool size, quality floor, and periodic eviction."""

    def __init__(self, pool: RoutePool, max_size: int) -> None:
        self.pool = pool
        self.max_size = max_size

    def maybe_evict(self) -> None:
        """Evict low-quality or redundant routes if over capacity."""
        # TODO: sort by cost contribution, call pool.remove
        raise NotImplementedError
