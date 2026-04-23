"""Tests for savings-based duplicate removal."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.sp.deduplicate import remove_duplicates


def test_remove_duplicates_no_change_when_partition(sp_instance: CVRPInstance) -> None:
    routes = [[2, 3], [4, 5], [6, 7]]
    out = remove_duplicates(routes, sp_instance)
    assert out == routes


def test_duplicate_kept_on_minimum_removal_savings_route(sp_instance: CVRPInstance) -> None:
    """Removal savings lower on ``[3, 4]`` → keep ``3`` there; ``[2, 3]`` becomes ``[2]``."""
    routes = [[2, 3], [3, 4]]
    out = remove_duplicates(routes, sp_instance)
    assert out == [[2], [3, 4]]


def test_empty_routes_dropped(sp_instance: CVRPInstance) -> None:
    routes = [[3], [2, 3]]
    out = remove_duplicates(routes, sp_instance)
    assert [] not in out
    assert all(len(r) > 0 for r in out)


def test_each_customer_at_most_once(sp_instance: CVRPInstance) -> None:
    routes = [[2, 3], [3, 4], [5, 6, 7]]
    out = remove_duplicates(routes, sp_instance)
    seen: set[int] = set()
    for route in out:
        for c in route:
            assert c not in seen
            seen.add(c)


def test_order_preserved_where_no_duplicate(sp_instance: CVRPInstance) -> None:
    routes = [[5, 7, 6], [2, 3], [3, 4]]
    out = remove_duplicates(routes, sp_instance)
    untouched = next(r for r in out if 5 in r and 7 in r)
    assert untouched == [5, 7, 6]
