"""Tests for set-partitioning formulation types."""

from __future__ import annotations

from drispi.set_partitioning.formulation import LPResult


def test_lp_result_dataclass() -> None:
    r = LPResult(lp_values={0: 1.0}, reduced_costs={0: 0.1}, duals={1: 0.5})
    assert r.lp_values[0] == 1.0
