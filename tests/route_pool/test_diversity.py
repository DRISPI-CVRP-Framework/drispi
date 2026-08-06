"""Tests for route-pool diversity metrics."""

from __future__ import annotations

from drispi.route_pool._entry import RouteEntry
from drispi.route_pool.diversity import (
    jaccard_distance,
    mean_jaccard_diversities,
    mean_jaccard_diversity,
)
from drispi.route_pool.pool import RoutePool


def _entry(route: list[int]) -> RouteEntry:
    return RouteEntry(route=route, cost=1.0, customer_set=frozenset(route))


def test_jaccard_distance_disjoint_and_identical() -> None:
    a = _entry([2, 3])
    b = _entry([4, 5])
    c = _entry([2, 3])
    assert jaccard_distance(a, b) == 1.0
    assert jaccard_distance(a, c) == 0.0


def test_mean_jaccard_diversities_matches_reference() -> None:
    entries = [
        _entry([2, 3, 4]),
        _entry([3, 4, 5]),
        _entry([6, 7]),
        _entry([2, 8]),
    ]
    batch = mean_jaccard_diversities(entries)
    reference = [mean_jaccard_diversity(entry, entries) for entry in entries]
    assert len(batch) == len(reference)
    for got, expected in zip(batch, reference, strict=True):
        assert got == expected


def test_mean_jaccard_diversities_single_and_empty() -> None:
    assert mean_jaccard_diversities([]) == []
    assert mean_jaccard_diversities([_entry([2, 3])]) == [1.0]


def test_pool_update_diversity_scores_uses_batch_metric() -> None:
    pool = RoutePool()
    routes = [[2, 3, 4], [3, 4, 5], [6, 7]]
    for route in routes:
        pool.add(route, 10.0)
    entries_before = list(pool.routes())
    reference = [mean_jaccard_diversity(e, entries_before) for e in entries_before]
    pool.update_diversity_scores()
    for entry, expected in zip(pool.routes(), reference, strict=True):
        assert list(entry.diversity_scores) == [expected]
