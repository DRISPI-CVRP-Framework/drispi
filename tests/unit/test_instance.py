"""Tests for instance distance matrix and solution cost recomputation."""

from __future__ import annotations

import math
import pickle

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route, Solution


def test_distance_matrix_matches_euclidean_for_all_pairs(small_instance: CVRPInstance) -> None:
    dist = small_instance.distance_matrix
    node_ids = [1, *small_instance.customers]
    for i in node_ids:
        for j in node_ids:
            assert dist[i, j] == small_instance.euclidean_distance(i, j)


def test_distance_matrix_is_symmetric(small_instance: CVRPInstance) -> None:
    dist = small_instance.distance_matrix
    node_ids = [1, *small_instance.customers]
    for i in node_ids:
        for j in node_ids:
            assert dist[i, j] == dist[j, i]


def test_distance_matrix_cached_property_computed_once(small_instance: CVRPInstance) -> None:
    first = small_instance.distance_matrix
    second = small_instance.distance_matrix
    assert first is second


def test_pickle_round_trip_drops_distance_matrix(small_instance: CVRPInstance) -> None:
    _ = small_instance.distance_matrix
    assert "distance_matrix" in small_instance.__dict__
    restored = pickle.loads(pickle.dumps(small_instance, protocol=pickle.HIGHEST_PROTOCOL))
    assert "distance_matrix" not in restored.__dict__
    assert restored.distance_matrix.shape == small_instance.distance_matrix.shape


def test_route_cost_empty_returns_zero(small_instance: CVRPInstance) -> None:
    assert small_instance.route_cost([]) == 0.0


def test_route_cost_uses_tsplib_euc_2d_rounding_per_leg() -> None:
    """Each arc uses ``round`` of Euclidean length (VRPLIB ``EUC_2D``), not raw floats."""
    inst = CVRPInstance(
        name="euc_round_tiny",
        n_customers=1,
        capacity=10,
        depot=(0.0, 0.0),
        customers=[2],
        coordinates={1: (0.0, 0.0), 2: (0.4, 0.4)},
        demands={1: 0, 2: 1},
    )
    leg = round(math.hypot(0.4, 0.4))
    assert leg == 1
    assert inst.route_cost([2]) == 2 * leg


def test_route_cost_matches_manual_euclidean(small_instance: CVRPInstance) -> None:
    route = [small_instance.customers[0], small_instance.customers[1]]
    c1, c2 = route
    x1, y1 = small_instance.coordinates[1]
    x2, y2 = small_instance.coordinates[c1]
    x3, y3 = small_instance.coordinates[c2]

    def leg(ax: float, ay: float, bx: float, by: float) -> float:
        return float(round(math.hypot(ax - bx, ay - by)))

    expected = leg(x1, y1, x2, y2) + leg(x2, y2, x3, y3) + leg(x1, y1, x3, y3)
    assert small_instance.route_cost(route) == expected


def test_recompute_cost_returns_same_routes_with_new_costs(small_instance: CVRPInstance) -> None:
    routes = [
        Route(customers=[small_instance.customers[0], small_instance.customers[2]], cost=999.0),
        Route(customers=[small_instance.customers[1]], cost=111.0),
    ]
    solution = Solution(routes=routes, total_cost=1110.0, instance_name=small_instance.name)

    recomputed = solution.recompute_cost(small_instance)

    assert [route.customers for route in recomputed.routes] == [route.customers for route in routes]
    assert recomputed.routes[0].cost == small_instance.route_cost(routes[0].customers)
    assert recomputed.routes[1].cost == small_instance.route_cost(routes[1].customers)


def test_recompute_cost_total_equals_sum_of_routes(small_instance: CVRPInstance) -> None:
    routes = [
        Route(customers=[small_instance.customers[0], small_instance.customers[1]], cost=1.0),
        Route(customers=[small_instance.customers[2], small_instance.customers[3]], cost=2.0),
    ]
    solution = Solution(routes=routes, total_cost=3.0, instance_name=small_instance.name)
    recomputed = solution.recompute_cost(small_instance)

    assert recomputed.total_cost == sum(route.cost for route in recomputed.routes)
