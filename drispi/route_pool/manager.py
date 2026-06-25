"""High-level route pool lifecycle (eviction, resets, score updates)."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.types import Route
from drispi.route_pool.fitness import select_for_eviction
from drispi.route_pool.pool import RoutePool


class RoutePoolManager:
    """Orchestrates route-pool eviction, resets, and score updates."""

    def __init__(
        self,
        max_pool_size: int,
        min_coverage: int,
        diversity_weight: float,
        warmup_iterations: int,
        sp_interval: int,
    ) -> None:
        self.max_pool_size = max_pool_size
        self.min_coverage = min_coverage
        self.diversity_weight = diversity_weight
        self.warmup_iterations = warmup_iterations
        self.sp_interval = sp_interval
        self.last_lp_fractionality: float | None = None

    def maybe_evict(self, pool: RoutePool) -> int:
        """Evict routes if pool exceeds configured max size."""
        keys_to_remove = select_for_eviction(
            entries=pool.routes(),
            current_size=pool.size(),
            max_size=self.max_pool_size,
            diversity_weight=self.diversity_weight,
        )
        for key in keys_to_remove:
            pool.remove(list(key))
        return len(keys_to_remove)

    def update_scores_after_solve(
        self,
        pool: RoutePool,
        lp_weights: dict[frozenset[int], float],
    ) -> None:
        """Update quality scores first, then diversity scores."""
        fractional = [v for v in lp_weights.values() if 0.0 < v < 1.0]
        self.last_lp_fractionality = (
            sum(fractional) / len(fractional) if fractional else 0.0
        )
        pool.update_quality_scores(lp_weights)
        pool.update_diversity_scores()

    def reset_pool_to_best(
        self,
        pool: RoutePool,
        best_solution: list[Route],
        costs: list[float],
        instance: CVRPInstance,
    ) -> None:
        """Reset pool to current best solution routes."""
        del instance
        pool.reset_to(best_solution, costs)
