"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from drispi.core.instance import CVRPInstance


@pytest.fixture
def small_instance() -> CVRPInstance:
    """Small 5-customer CVRP instance with predictable coordinates."""
    n = 5
    depot_id = 1
    customer_ids = list(range(2, 2 + n))
    coordinates: dict[int, tuple[float, float]] = {depot_id: (0.0, 0.0)}
    demands: dict[int, int] = {depot_id: 0}
    for i, cid in enumerate(customer_ids):
        coordinates[cid] = (float(i + 1), float((i + 1) * 2))
        demands[cid] = 10
    return CVRPInstance(
        name="small_test",
        n_customers=n,
        capacity=100,
        depot=(0.0, 0.0),
        customers=customer_ids,
        coordinates=coordinates,
        demands=demands,
    )
