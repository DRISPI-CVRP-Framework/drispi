"""Tests for subproblem construction and parallel solve wiring."""

from __future__ import annotations

import pytest

from drispi.core.instance import CVRPInstance
from drispi.pipeline.subproblem import (
    make_subinstance,
    remap_routes_from_subcluster,
    solve_subclusters_parallel,
)


def test_make_subinstance_size_and_mapping(instance_12: CVRPInstance) -> None:
    cluster = [2, 5, 11]
    sub = make_subinstance(instance_12, cluster)
    assert sub.n_customers == 3
    assert sub.customers == [2, 3, 4]
    assert sub.coordinates[2] == instance_12.coordinates[2]
    assert sub.coordinates[3] == instance_12.coordinates[5]
    assert sub.coordinates[4] == instance_12.coordinates[11]
    assert sub.demands[3] == instance_12.demands[5]
    d_parent = instance_12.distance_matrix[2, 5]
    d_sub = sub.distance_matrix[2, 3]
    assert d_sub == pytest.approx(d_parent)


def test_remap_roundtrip(instance_12: CVRPInstance) -> None:
    cluster = [3, 7, 9]
    sub_routes = [[2, 3], [4]]
    parent = remap_routes_from_subcluster(sub_routes, cluster)
    assert parent == [[3, 7], [9]]


def test_solve_subclusters_parallel_real_pyvrp_small(instance_12: CVRPInstance) -> None:
    """Optional smoke: real worker + PyVRP on tiny partition (may be slow)."""
    partition = [instance_12.customers[0:6], instance_12.customers[6:12]]
    out = solve_subclusters_parallel(
        instance_12,
        partition,
        "pyvrp",
        time_per_customer=0.5,
        n_workers=2,
        seed=1,
    )
    assert len(out) == 2
    seen: set[int] = set()
    for grp in out:
        for route in grp:
            seen.update(route)
    assert seen == set(instance_12.customers)
