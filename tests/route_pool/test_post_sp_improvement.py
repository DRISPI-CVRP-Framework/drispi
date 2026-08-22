"""Tests for post-SP/SC standard AILS pool tagging helper."""

from __future__ import annotations

import random

from drispi.core.instance import CVRPInstance
from drispi.haos.config import HAOSConfig
from drispi.haos.haos import HAOS
from drispi.haos.tag import HAOSTag
from drispi.route_pool.pool import RoutePool
from drispi.route_pool.post_sp_improvement import add_post_standard_improvement_routes_to_pool


def _tiny_instance() -> CVRPInstance:
    customers = [2, 3, 4, 5]
    coordinates = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0), 4: (3.0, 0.0), 5: (4.0, 0.0)}
    demands = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}
    return CVRPInstance(
        name="post_sp_tiny",
        n_customers=4,
        capacity=10,
        depot=(0.0, 0.0),
        customers=customers,
        coordinates=coordinates,
        demands=demands,
    )


def test_always_adds_even_when_not_new_best() -> None:
    instance = _tiny_instance()
    pool = RoutePool()
    haos = HAOS(config=HAOSConfig(k_min_routes_per_cluster=0, k_min_arm_spacing=1), instance=instance)
    selection = haos.select(iteration=0, rng=random.Random(0))
    sp = [[2, 3], [4, 5]]
    ails = [[2, 3], [4, 5]]
    assert (
        add_post_standard_improvement_routes_to_pool(
            pool,
            instance,
            sp,
            ails,
            iteration=1,
            selection=selection,
        )
        == 2
    )
    assert pool.size() == 2


def test_changed_route_gets_improvement_tag() -> None:
    instance = _tiny_instance()
    pool = RoutePool()
    haos = HAOS(config=HAOSConfig(k_min_routes_per_cluster=0, k_min_arm_spacing=1), instance=instance)
    selection = haos.select(iteration=0, rng=random.Random(1))
    sp = [[2, 3], [4, 5]]
    ails = [[2, 3, 4], [5]]  # merged first route -> new customer set vs SP
    assert (
        add_post_standard_improvement_routes_to_pool(
            pool,
            instance,
            sp,
            ails,
            iteration=2,
            selection=selection,
        )
        == 2
    )
    by_key = {frozenset(e.route): e for e in pool.routes()}
    assert frozenset({2, 3, 4}) in by_key
    assert by_key[frozenset({2, 3, 4})].haos_tag == selection.to_tag(
        2, is_improvement_route=True
    )
    assert by_key[frozenset({5})].haos_tag == selection.to_tag(2, is_improvement_route=True)


def test_unchanged_route_reuses_pool_tag() -> None:
    instance = _tiny_instance()
    pool = RoutePool()
    haos = HAOS(config=HAOSConfig(k_min_routes_per_cluster=0, k_min_arm_spacing=1), instance=instance)
    selection = haos.select(iteration=0, rng=random.Random(2))
    original = HAOSTag(3, 0.5, "vertex", "spectral", "ails2", 0, False)
    sp_route = [2, 3, 4]
    pool.add(sp_route, 20.0, haos_tag=original)
    sp = [sp_route, [5]]
    ails = [sp_route, [5]]
    assert (
        add_post_standard_improvement_routes_to_pool(
            pool,
            instance,
            sp,
            ails,
            iteration=3,
            selection=selection,
        )
        == 2
    )
    entry = next(e for e in pool.routes() if frozenset(e.route) == frozenset(sp_route))
    assert entry.haos_tag == original
