"""Tests for solver registry."""

from __future__ import annotations

import drispi.solvers.hgs  # noqa: F401
from drispi.solvers.base import SOLVER_REGISTRY


def test_hgs_registered() -> None:
    assert "hgs" in SOLVER_REGISTRY
