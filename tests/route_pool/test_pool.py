"""Tests for route pool core behavior."""

from __future__ import annotations

import pytest

from drispi.route_pool.pool import RoutePool


def test_add_new_route_increases_size() -> None:
    pool = RoutePool()
    added = pool.add([2, 3, 4], 10.0)
    assert added is True
    assert pool.size() == 1


def test_add_duplicate_higher_cost_is_discarded() -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    added = pool.add([4, 3, 2], 11.0)
    assert added is False
    assert pool.size() == 1
    assert pool.routes()[0].cost == 10.0


def test_add_duplicate_lower_cost_replaces_entry() -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    added = pool.add([4, 3, 2], 9.0)
    assert added is False
    assert pool.size() == 1
    assert pool.routes()[0].cost == 9.0


def test_remove_present_route_decreases_size() -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    removed = pool.remove([2, 3, 4])
    assert removed is True
    assert pool.size() == 0


def test_remove_absent_route_returns_false() -> None:
    pool = RoutePool()
    removed = pool.remove([2, 3, 4])
    assert removed is False


def test_set_elite_marks_exact_routes() -> None:
    pool = RoutePool()
    route_a = [2, 3, 4]
    route_b = [5, 6, 7]
    route_c = [8, 9, 10]
    pool.add(route_a, 10.0)
    pool.add(route_b, 11.0)
    pool.add(route_c, 12.0)

    pool.set_elite([route_a, route_c])
    elite_sets = {entry.customer_set for entry in pool.routes(elite_only=True)}
    assert elite_sets == {frozenset(route_a), frozenset(route_c)}

    non_elite_sets = {entry.customer_set for entry in pool.routes() if not entry.is_elite}
    assert non_elite_sets == {frozenset(route_b)}


def test_set_elite_raises_for_missing_route() -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    with pytest.raises(KeyError):
        pool.set_elite([[2, 3, 4], [8, 9, 10]])


def test_as_route_pool_returns_plain_route_list() -> None:
    pool = RoutePool()
    route = [2, 3, 4]
    pool.add(route, 10.0)
    exported = pool.as_route_pool()
    assert exported == [route]
    assert isinstance(exported, list)
    assert isinstance(exported[0], list)


def test_reset_to_readds_routes_as_elite() -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    reset_routes = [[5, 6, 7], [8, 9, 10]]
    reset_costs = [11.0, 12.0]

    pool.reset_to(reset_routes, reset_costs)

    assert pool.size() == 2
    assert len(pool.routes(elite_only=True)) == 2
    elite_sets = {entry.customer_set for entry in pool.routes(elite_only=True)}
    assert elite_sets == {frozenset(r) for r in reset_routes}


def test_contains_accepts_permuted_customer_order() -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    assert [4, 2, 3] in pool
