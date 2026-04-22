"""Tests for route pool coverage utilities."""

from __future__ import annotations

from drispi.route_pool.coverage import (
    coverage_counts,
    coverage_satisfied,
    routes_covering,
    uncovered_customers,
)
from drispi.route_pool.pool import RoutePool


def test_coverage_counts_known_pool(grid_instance) -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    pool.add([4, 5, 6], 11.0)
    pool.add([6, 7], 8.0)

    counts = coverage_counts(pool, grid_instance)
    assert counts[2] == 1
    assert counts[4] == 2
    assert counts[6] == 2
    assert counts[11] == 0


def test_uncovered_customers_returns_missing_nodes(grid_instance) -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    missing = uncovered_customers(pool, grid_instance)
    assert set(missing) == {5, 6, 7, 8, 9, 10, 11}


def test_coverage_satisfied_true_for_min_one(grid_instance) -> None:
    pool = RoutePool()
    pool.add([2, 3, 4, 5, 6], 10.0)
    pool.add([7, 8, 9, 10, 11], 10.0)
    assert coverage_satisfied(pool, grid_instance, min_coverage=1) is True


def test_coverage_satisfied_false_for_min_two(grid_instance) -> None:
    pool = RoutePool()
    pool.add([2, 3, 4, 5, 6], 10.0)
    pool.add([7, 8, 9, 10, 11], 10.0)
    assert coverage_satisfied(pool, grid_instance, min_coverage=2) is False


def test_routes_covering_returns_expected_routes() -> None:
    pool = RoutePool()
    route_a = [2, 3, 4]
    route_b = [4, 5, 6]
    route_c = [7, 8]
    pool.add(route_a, 10.0)
    pool.add(route_b, 11.0)
    pool.add(route_c, 12.0)

    routes = routes_covering(pool, 4)
    assert {tuple(route) for route in routes} == {tuple(route_a), tuple(route_b)}
