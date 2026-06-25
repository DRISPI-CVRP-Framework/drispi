"""Tests for route pool manager behavior."""

from __future__ import annotations

from drispi.route_pool.manager import RoutePoolManager
from drispi.route_pool.pool import RoutePool


def test_maybe_evict_reduces_pool_to_max_size() -> None:
    pool = RoutePool()
    pool.add([2, 3], 10.0)
    pool.add([4, 5], 11.0)
    pool.add([6, 7], 12.0)

    for idx, entry in enumerate(pool.routes(), start=1):
        entry.quality_scores.append(float(idx))
        entry.diversity_scores.append(float(4 - idx))

    manager = RoutePoolManager(
        max_pool_size=2, min_coverage=1, diversity_weight=1.0, warmup_iterations=10, sp_interval=3
    )
    evicted = manager.maybe_evict(pool)

    assert evicted == 1
    assert pool.size() == 2


def test_update_scores_after_solve_appends_quality_and_diversity() -> None:
    pool = RoutePool()
    route_a = [2, 3]
    route_b = [4, 5]
    pool.add(route_a, 10.0)
    pool.add(route_b, 12.0)
    manager = RoutePoolManager(
        max_pool_size=10, min_coverage=1, diversity_weight=1.0, warmup_iterations=10, sp_interval=3
    )

    lp_weights = {frozenset(route_a): 0.7}
    manager.update_scores_after_solve(pool, lp_weights)

    entries = {entry.customer_set: entry for entry in pool.routes()}
    assert list(entries[frozenset(route_a)].quality_scores) == [0.7]
    assert list(entries[frozenset(route_b)].quality_scores) == []
    assert len(entries[frozenset(route_a)].diversity_scores) == 1
    assert len(entries[frozenset(route_b)].diversity_scores) == 1


def test_reset_pool_to_best_clears_and_marks_all_elite(grid_instance) -> None:
    pool = RoutePool()
    pool.add([2, 3], 10.0)

    best_solution = [[4, 5], [6, 7]]
    costs = [11.0, 12.0]
    manager = RoutePoolManager(
        max_pool_size=10, min_coverage=1, diversity_weight=1.0, warmup_iterations=10, sp_interval=3
    )
    manager.reset_pool_to_best(pool, best_solution, costs, grid_instance)

    assert pool.size() == 2
    assert len(pool.routes(elite_only=True)) == 2
