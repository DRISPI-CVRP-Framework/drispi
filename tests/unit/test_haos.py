"""Tests for HAOS hierarchy construction."""

from __future__ import annotations

from drispi.haos.hierarchy import HierarchicalHAOS


def test_hierarchical_haos_init() -> None:
    model = HierarchicalHAOS(k_max=5, solver_ids=["hgs", "filo"])
    assert model.level_solver is not None
