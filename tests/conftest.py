"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from drispi.core.instance import CVRPInstance


@pytest.fixture
def small_instance() -> CVRPInstance:
    """Minimal 10-customer CVRP instance (depot at origin, unit demands)."""
    n = 10
    depot_id = 1
    customer_ids = list(range(2, 2 + n))
    coordinates: dict[int, tuple[float, float]] = {depot_id: (0.0, 0.0)}
    demands: dict[int, int] = {depot_id: 0}
    for i, cid in enumerate(customer_ids):
        coordinates[cid] = (float(i + 1), 0.0)
        demands[cid] = 1
    return CVRPInstance(
        name="small_test",
        n_customers=n,
        capacity=4,
        depot=(0.0, 0.0),
        customers=customer_ids,
        coordinates=coordinates,
        demands=demands,
    )
