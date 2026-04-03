"""Tests for AOLS hierarchy construction."""

from __future__ import annotations

from drispi.aols.hierarchy import HierarchicalAOLS


def test_hierarchical_aols_init() -> None:
    model = HierarchicalAOLS(k_max=5, solver_ids=["hgs", "filo"])
    assert model.level_solver is not None
