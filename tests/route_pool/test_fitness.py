"""Tests for route fitness ranking and eviction selection."""

from __future__ import annotations

from drispi.route_pool._entry import RouteEntry
from drispi.route_pool.fitness import compute_fitness_ranks, select_for_eviction


def _entry(
    route: list[int],
    *,
    elite: bool = False,
    quality: list[float] | None = None,
    diversity: list[float] | None = None,
) -> RouteEntry:
    entry = RouteEntry(route=route, cost=10.0, customer_set=frozenset(route), is_elite=elite)
    for value in quality or []:
        entry.quality_scores.append(value)
    for value in diversity or []:
        entry.diversity_scores.append(value)
    return entry


def test_compute_fitness_ranks_excludes_elite_routes() -> None:
    elite = _entry([2, 3], elite=True, quality=[0.2], diversity=[0.8])
    non_elite = _entry([4, 5], quality=[0.5], diversity=[0.5])
    result = compute_fitness_ranks([elite, non_elite])
    assert frozenset([2, 3]) not in result
    assert frozenset([4, 5]) in result


def test_compute_fitness_ranks_higher_quality_and_diversity_score_higher() -> None:
    strong = _entry([2, 3], quality=[0.8], diversity=[0.9])
    weak = _entry([4, 5], quality=[0.2], diversity=[0.1])
    result = compute_fitness_ranks([strong, weak], diversity_weight=1.0)
    assert result[strong.customer_set] > result[weak.customer_set]


def test_select_for_eviction_empty_when_within_limit() -> None:
    entries = [_entry([2, 3], quality=[0.2], diversity=[0.7])]
    selected = select_for_eviction(entries, current_size=1, max_size=2)
    assert selected == []


def test_select_for_eviction_never_selects_elite() -> None:
    elite = _entry([2, 3], elite=True, quality=[0.9], diversity=[0.1])
    non_elite_a = _entry([4, 5], quality=[0.8], diversity=[0.2])
    non_elite_b = _entry([6, 7], quality=[0.7], diversity=[0.3])

    selected = select_for_eviction(
        [elite, non_elite_a, non_elite_b],
        current_size=3,
        max_size=1,
    )
    assert elite.customer_set not in selected


def test_select_for_eviction_returns_required_number_of_keys() -> None:
    entries = [
        _entry([2, 3], quality=[0.2], diversity=[0.9]),
        _entry([4, 5], quality=[0.8], diversity=[0.2]),
        _entry([6, 7], quality=[0.7], diversity=[0.3]),
        _entry([8, 9], quality=[0.6], diversity=[0.4]),
    ]
    selected = select_for_eviction(entries, current_size=4, max_size=2)
    assert len(selected) == 2


def test_select_for_eviction_takes_lowest_fitness_first() -> None:
    """Weak and redundant routes have the lowest fitness and are evicted first."""
    strong_unique = _entry([2, 3], quality=[0.9], diversity=[0.9])
    weak_redundant = _entry([4, 5], quality=[0.1], diversity=[0.1])
    strong_redundant = _entry([6, 7], quality=[0.9], diversity=[0.1])
    weak_unique = _entry([8, 9], quality=[0.1], diversity=[0.9])

    selected = select_for_eviction(
        [strong_unique, weak_redundant, strong_redundant, weak_unique],
        current_size=4,
        max_size=3,
        diversity_weight=1.0,
    )
    assert selected == [weak_redundant.customer_set]


def test_select_for_eviction_prefers_scored_over_unscored() -> None:
    """Soft-protect: do not evict unscored when scored can cover overflow."""
    scored_a = _entry([2, 3], quality=[0.9], diversity=[0.1])
    scored_b = _entry([4, 5], quality=[0.1], diversity=[0.9])
    unscored_a = _entry([6, 7])
    unscored_b = _entry([8, 9])

    selected = select_for_eviction(
        [scored_a, scored_b, unscored_a, unscored_b],
        current_size=4,
        max_size=2,
        diversity_weight=1.0,
    )
    assert len(selected) == 2
    assert scored_a.customer_set in selected
    assert scored_b.customer_set in selected
    assert unscored_a.customer_set not in selected
    assert unscored_b.customer_set not in selected


def test_select_for_eviction_falls_back_to_unscored_when_needed() -> None:
    """If scored non-elite are insufficient, take remaining from unscored."""
    scored = _entry([2, 3], quality=[0.5], diversity=[0.5])
    unscored_a = _entry([4, 5])
    unscored_b = _entry([6, 7])
    elite = _entry([8, 9], elite=True, quality=[0.1], diversity=[0.9])

    selected = select_for_eviction(
        [scored, unscored_a, unscored_b, elite],
        current_size=4,
        max_size=1,
        diversity_weight=1.0,
    )
    assert len(selected) == 3
    assert scored.customer_set in selected
    assert elite.customer_set not in selected
    assert {unscored_a.customer_set, unscored_b.customer_set} <= set(selected)
