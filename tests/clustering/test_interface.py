"""Tests for unified clustering interface."""

from __future__ import annotations

import pytest

from drispi.clustering.interface import cluster_instance
from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool


def _build_instance_20() -> CVRPInstance:
    n_customers = 20
    customer_ids = list(range(2, 2 + n_customers))
    coordinates: dict[int, tuple[float, float]] = {1: (0.0, 0.0)}
    demands: dict[int, int] = {1: 0}

    idx = 0
    for row in range(4):
        for col in range(5):
            cid = customer_ids[idx]
            coordinates[cid] = (float(col), float(row))
            demands[cid] = 5
            idx += 1

    return CVRPInstance(
        name="interface_test_20",
        n_customers=n_customers,
        capacity=100,
        depot=(0.0, 0.0),
        customers=customer_ids,
        coordinates=coordinates,
        demands=demands,
    )


def _build_routes(instance: CVRPInstance) -> RoutePool:
    customers = list(instance.customers)
    return [
        customers[0:5],
        customers[5:10],
        customers[10:15],
        customers[15:20],
    ]


def _assert_partition(groups: list[list[int]], instance: CVRPInstance, k: int) -> None:
    assert len(groups) == k
    flat = [customer for group in groups for customer in group]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(instance.customers)


def _has_working_kmedoids() -> bool:
    try:
        from sklearn_extra.cluster import KMedoids  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.parametrize(
    "method",
    [
        "kmeans",
        "agglomerative_avg",
        "agglomerative_complete",
        "agglomerative_single",
        "kmedoids",
        "fcm",
        "spectral",
    ],
)
def test_vertex_methods_cover_all_customers(method: str) -> None:
    if method == "kmedoids" and not _has_working_kmedoids():
        pytest.skip("kmedoids skipped: sklearn_extra binary is incompatible in this environment")
    instance = _build_instance_20()
    groups = cluster_instance(instance=instance, paradigm="vertex", method=method, k=3, seed=123)
    _assert_partition(groups, instance, k=3)


@pytest.mark.parametrize(
    "method",
    ["kmeans", "agglomerative_avg", "agglomerative_complete", "agglomerative_single"],
)
def test_route_methods_cover_all_customers(method: str) -> None:
    instance = _build_instance_20()
    routes = _build_routes(instance)
    groups = cluster_instance(
        instance=instance,
        paradigm="route",
        method=method,
        k=3,
        routes=routes,
        seed=123,
    )
    _assert_partition(groups, instance, k=3)


def test_route_paradigm_with_none_routes_raises_value_error() -> None:
    instance = _build_instance_20()
    with pytest.raises(ValueError):
        cluster_instance(instance=instance, paradigm="route", method="kmeans", k=3, routes=None)


def test_route_paradigm_with_empty_routes_raises_value_error() -> None:
    instance = _build_instance_20()
    with pytest.raises(ValueError):
        cluster_instance(instance=instance, paradigm="route", method="kmeans", k=3, routes=[])


def test_unknown_method_raises_value_error() -> None:
    instance = _build_instance_20()
    with pytest.raises(ValueError):
        cluster_instance(instance=instance, paradigm="vertex", method="unknown_method", k=3)


def test_unknown_paradigm_raises_value_error() -> None:
    instance = _build_instance_20()
    with pytest.raises(ValueError):
        cluster_instance(instance=instance, paradigm="unknown", method="kmeans", k=3)
