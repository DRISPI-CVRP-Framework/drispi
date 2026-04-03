"""Diversity metrics and maintenance for route pools."""

from __future__ import annotations

from drispi.route_pool.pool import RoutePool


def hamming_diversity(pool: RoutePool) -> float:
    """Aggregate dissimilarity score between routes in ``pool``."""
    # TODO: pairwise set distance on customer sets
    raise NotImplementedError


def prune_similar(pool: RoutePool, threshold: float) -> None:
    """Remove routes that are too similar to others (in-place)."""
    # TODO: clustering or thresholded pairwise comparison
    raise NotImplementedError
