"""Tests for route costs and Gurobi SC/SP models."""

from __future__ import annotations

import pytest

from drispi.core.instance import CVRPInstance
from drispi.route_pool.pool import RoutePool
from drispi.sp.model import build_and_solve, compute_route_cost


def test_compute_route_cost_matches_manual_sum(sp_instance: CVRPInstance) -> None:
    route = [2, 3]
    dm = sp_instance.distance_matrix
    manual = dm[1, 2] + dm[2, 3] + dm[3, 1]
    assert compute_route_cost(route, sp_instance) == pytest.approx(float(manual))


def test_build_and_set_covering_lp_weights_bounded(
    requires_gurobi: None,
    sp_instance: CVRPInstance,
) -> None:
    pool = RoutePool()
    pool.add([2, 3], sp_instance.route_cost([2, 3]))
    pool.add([4, 5], sp_instance.route_cost([4, 5]))
    pool.add([6, 7], sp_instance.route_cost([6, 7]))
    pool.add([4, 5, 6], sp_instance.route_cost([4, 5, 6]))

    lp_weights, raw, timed_out, sol_count = build_and_solve(
        pool,
        sp_instance,
        use_sp=False,
        time_limit=300.0,
        mip_gap=0.01,
    )

    assert timed_out is False
    assert sol_count >= 1
    assert all(0.0 <= w <= 1.0 for w in lp_weights.values())

    covered = set()
    for route in raw:
        covered.update(route)
    assert covered == set(sp_instance.customers)


def test_partition_solution_unique_assignment(
    requires_gurobi: None,
    sp_instance: CVRPInstance,
) -> None:
    pool = RoutePool()
    pool.add([2, 3], sp_instance.route_cost([2, 3]))
    pool.add([4, 5], sp_instance.route_cost([4, 5]))
    pool.add([6, 7], sp_instance.route_cost([6, 7]))

    lp_weights, raw, timed_out, sol_count = build_and_solve(
        pool,
        sp_instance,
        use_sp=True,
        time_limit=300.0,
        mip_gap=0.01,
    )

    assert timed_out is False
    assert sol_count >= 1

    counts: dict[int, int] = {}
    for route in raw:
        for c in route:
            counts[c] = counts.get(c, 0) + 1
    assert all(counts[c] == 1 for c in sp_instance.customers)


def test_lp_weights_respect_formulation_constraints(
    requires_gurobi: None,
    sp_instance: CVRPInstance,
) -> None:
    pool = RoutePool()
    pool.add([2, 3], sp_instance.route_cost([2, 3]))
    pool.add([4, 5], sp_instance.route_cost([4, 5]))
    pool.add([6, 7], sp_instance.route_cost([6, 7]))
    pool.add([4, 5, 6], sp_instance.route_cost([4, 5, 6]))

    lp_sc, _, _, _ = build_and_solve(
        pool,
        sp_instance,
        use_sp=False,
        time_limit=300.0,
        mip_gap=0.01,
    )
    lp_sp, _, _, _ = build_and_solve(
        pool,
        sp_instance,
        use_sp=True,
        time_limit=300.0,
        mip_gap=0.01,
    )

    for customer in sp_instance.customers:
        sc_cover = sum(weight for key, weight in lp_sc.items() if customer in key)
        sp_cover = sum(weight for key, weight in lp_sp.items() if customer in key)
        assert sc_cover >= 1.0 - 1e-6
        assert sp_cover == pytest.approx(1.0)
