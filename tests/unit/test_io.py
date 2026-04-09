"""Tests for core I/O and feasibility helpers."""

from __future__ import annotations

from pathlib import Path

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route, Solution
from drispi.utils.io import (
    read_sol,
    reindex_depot_one_to_zero,
    reindex_depot_zero_to_one,
    write_sol,
    write_vrp,
)


def test_sol_round_trip(tmp_path: Path, small_instance: CVRPInstance) -> None:
    routes = [[small_instance.customers[0], small_instance.customers[2]], [small_instance.customers[1]]]
    cost = 123.0
    sol_path = tmp_path / "out.sol"

    write_sol(routes, cost, sol_path)
    loaded_routes, loaded_cost = read_sol(sol_path)

    assert loaded_routes == routes
    assert loaded_cost == cost


def test_reindex_round_trip(small_instance: CVRPInstance) -> None:
    zero_indexed = reindex_depot_one_to_zero(small_instance)
    restored = reindex_depot_zero_to_one(zero_indexed)

    assert restored == small_instance


def test_write_vrp_local_mapping(tmp_path: Path, small_instance: CVRPInstance) -> None:
    selected = [small_instance.customers[1], small_instance.customers[3]]
    vrp_path = tmp_path / "subproblem" / "cluster.vrp"

    global_to_local, local_to_global = write_vrp(small_instance, selected, vrp_path)

    assert global_to_local == {1: 1, selected[0]: 2, selected[1]: 3}
    assert local_to_global == {1: 1, 2: selected[0], 3: selected[1]}
    assert vrp_path.exists()


def test_route_cost_empty_returns_zero(small_instance: CVRPInstance) -> None:
    assert small_instance.route_cost([]) == 0.0


def test_solution_feasible_rejects_duplicate_customer(small_instance: CVRPInstance) -> None:
    duplicate = small_instance.customers[0]
    routes = [
        Route(customers=[duplicate, small_instance.customers[1]], cost=0.0),
        Route(customers=[duplicate, small_instance.customers[2], small_instance.customers[3]], cost=0.0),
    ]
    solution = Solution(routes=routes, total_cost=0.0, instance_name=small_instance.name)

    assert not solution.is_feasible(small_instance)


def test_solution_feasible_rejects_missing_customer(small_instance: CVRPInstance) -> None:
    missing = small_instance.customers[-1]
    assigned = [customer for customer in small_instance.customers if customer != missing]
    routes = [
        Route(customers=assigned[:2], cost=0.0),
        Route(customers=assigned[2:], cost=0.0),
    ]
    solution = Solution(routes=routes, total_cost=0.0, instance_name=small_instance.name)

    assert not solution.is_feasible(small_instance)
