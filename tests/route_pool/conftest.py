"""Fixtures for route-pool tests."""

from __future__ import annotations

import pytest

from drispi.core.instance import CVRPInstance


@pytest.fixture
def grid_instance() -> CVRPInstance:
    """10-customer synthetic instance on a 2x5 grid."""
    depot_id = 1
    customers = list(range(2, 12))
    coordinates: dict[int, tuple[float, float]] = {depot_id: (0.0, 0.0)}
    demands: dict[int, int] = {depot_id: 0}

    index = 0
    for y in (1.0, 2.0):
        for x in (1.0, 2.0, 3.0, 4.0, 5.0):
            node_id = customers[index]
            coordinates[node_id] = (x, y)
            demands[node_id] = 10
            index += 1

    return CVRPInstance(
        name="grid10",
        n_customers=10,
        capacity=50,
        depot=(0.0, 0.0),
        customers=customers,
        coordinates=coordinates,
        demands=demands,
    )
