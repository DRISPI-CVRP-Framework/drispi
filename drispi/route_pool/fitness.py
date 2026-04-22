"""Fitness ranking and eviction selection for route entries."""

from __future__ import annotations

from drispi.route_pool._entry import RouteEntry


def _stable_key(entry: RouteEntry) -> tuple[int, ...]:
    return tuple(sorted(entry.customer_set))


def compute_fitness_ranks(
    entries: list[RouteEntry],
    diversity_weight: float = 1.0,
) -> dict[frozenset[int], float]:
    """
    Compute aggregate fitness score for each non-elite route entry.

    Higher aggregate score means a stronger eviction candidate.
    """
    candidates = [entry for entry in entries if not entry.is_elite]
    if not candidates:
        return {}

    quality_sorted = sorted(
        candidates,
        key=lambda entry: (
            len(entry.quality_scores) == 0,
            entry.quality_rank_score,
            _stable_key(entry),
        ),
    )
    diversity_sorted = sorted(
        candidates,
        key=lambda entry: (
            len(entry.diversity_scores) == 0,
            -entry.diversity_rank_score,
            _stable_key(entry),
        ),
    )

    quality_rank = {
        entry.customer_set: rank for rank, entry in enumerate(quality_sorted, start=1)
    }
    diversity_rank = {
        entry.customer_set: rank for rank, entry in enumerate(diversity_sorted, start=1)
    }

    return {
        entry.customer_set: quality_rank[entry.customer_set]
        + diversity_weight * diversity_rank[entry.customer_set]
        for entry in candidates
    }


def select_for_eviction(
    entries: list[RouteEntry],
    current_size: int,
    max_size: int,
    diversity_weight: float = 1.0,
) -> list[frozenset[int]]:
    """Select customer-set keys to evict in order to satisfy max pool size."""
    overflow = current_size - max_size
    if overflow <= 0:
        return []

    fitness = compute_fitness_ranks(entries, diversity_weight=diversity_weight)
    sorted_keys = sorted(fitness, key=lambda key: fitness[key], reverse=True)
    return sorted_keys[:overflow]
