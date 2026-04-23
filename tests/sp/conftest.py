"""Shared fixtures for ``drispi.sp`` tests."""

from __future__ import annotations

import pytest

from drispi.core.instance import CVRPInstance


@pytest.fixture
def requires_gurobi() -> None:
    """Skip the test when Gurobi cannot create a model (no / invalid license)."""
    import gurobipy as gp

    try:
        model = gp.Model("_license_probe")
        model.dispose()
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi not available: {exc}")


@pytest.fixture
def sp_instance() -> CVRPInstance:
    """Six customers (nodes 2..7) on the x-axis, depot at origin."""
    coords: dict[int, tuple[float, float]] = {1: (0.0, 0.0)}
    demands: dict[int, int] = {1: 0}
    customers = list(range(2, 8))
    for cid in customers:
        coords[cid] = (float(cid - 1), 0.0)
        demands[cid] = 10

    return CVRPInstance(
        name="sp_fixture",
        n_customers=6,
        capacity=30,
        depot=(0.0, 0.0),
        customers=customers,
        coordinates=coords,
        demands=demands,
    )
