"""Shared fixtures for pipeline tests."""

from __future__ import annotations

import pytest

from drispi.core.instance import CVRPInstance


@pytest.fixture
def instance_12() -> CVRPInstance:
    """12 customers (IDs 2..13), depot at origin, uniform demand, capacity 40."""
    customer_ids = list(range(2, 14))
    coordinates: dict[int, tuple[float, float]] = {1: (0.0, 0.0)}
    demands: dict[int, int] = {1: 0}
    for i, cid in enumerate(customer_ids):
        coordinates[cid] = (float(i % 4), float(i // 4))
        demands[cid] = 10
    return CVRPInstance(
        name="pipeline_12",
        n_customers=12,
        capacity=40,
        depot=(0.0, 0.0),
        customers=customer_ids,
        coordinates=coordinates,
        demands=demands,
    )
