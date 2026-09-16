"""Fitness ranking and eviction selection for route entries."""

from __future__ import annotations

from drispi.route_pool._entry import RouteEntry


def _stable_key(entry: RouteEntry) -> tuple[int, ...]:
    return tuple(sorted(entry.customer_set))


def _is_scored(entry: RouteEntry) -> bool:
    """True if the entry has survived at least one diversity score update."""
    return len(entry.diversity_scores) > 0


def compute_fitness_ranks(
    entries: list[RouteEntry],
    diversity_weight: float = 1.0,
) -> dict[frozenset[int], float]:
    """
    Compute aggregate fitness score for each non-elite route entry.

    Both component rankings are ascending in the raw score: rank one is the
    weakest LP weight / most redundant route, and the highest rank is the
    strongest / most unique. Higher aggregate fitness is better; eviction
    takes the lowest fitness first.
    """
    candidates = [entry for entry in entries if not entry.is_elite]
    if not candidates:
        return {}

    quality_sorted = sorted(
        candidates,
        key=lambda entry: (
            entry.quality_rank_score,
            _stable_key(entry),
        ),
    )
    diversity_sorted = sorted(
        candidates,
        key=lambda entry: (
            entry.diversity_rank_score,
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


def _eviction_order(
    entries: list[RouteEntry],
    diversity_weight: float,
) -> list[frozenset[int]]:
    """Lowest-fitness-first keys among ``entries`` (assumed non-elite)."""
    if not entries:
        return []
    fitness = compute_fitness_ranks(entries, diversity_weight=diversity_weight)
    return sorted(fitness, key=lambda key: fitness[key])


def select_for_eviction(
    entries: list[RouteEntry],
    current_size: int,
    max_size: int,
    diversity_weight: float = 1.0,
) -> list[frozenset[int]]:
    """
    Select customer-set keys to evict in order to satisfy max pool size.

    Soft-protects unscored routes (empty diversity history): scored non-elite
    routes are preferred for eviction. Unscored routes are only taken if overflow
    cannot be covered by scored non-elite alone. Elite routes are never selected.
    """
    overflow = current_size - max_size
    if overflow <= 0:
        return []

    non_elite = [entry for entry in entries if not entry.is_elite]
    scored = [entry for entry in non_elite if _is_scored(entry)]
    unscored = [entry for entry in non_elite if not _is_scored(entry)]

    selected: list[frozenset[int]] = []
    scored_order = _eviction_order(scored, diversity_weight)
    take_scored = min(overflow, len(scored_order))
    selected.extend(scored_order[:take_scored])

    remaining = overflow - len(selected)
    if remaining > 0:
        unscored_order = _eviction_order(unscored, diversity_weight)
        selected.extend(unscored_order[:remaining])

    return selected
