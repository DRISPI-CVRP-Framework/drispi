"""Unit tests for the BG-AILS instance-size budget."""

from __future__ import annotations

import pytest

from drispi.improvement.bg_ails_budget import (
    bg_ails_budget_seconds,
    check_bg_ails_divisor_coupling,
)


def test_budget_hits_floor_for_small_n() -> None:
    assert bg_ails_budget_seconds(1000, min_budget=60.0, divisor=46.5) == 60.0


def test_budget_scales_above_floor() -> None:
    assert bg_ails_budget_seconds(4650, min_budget=60.0, divisor=46.5) == pytest.approx(100.0)


def test_budget_at_exact_floor_boundary() -> None:
    assert bg_ails_budget_seconds(2790, min_budget=60.0, divisor=46.5) == pytest.approx(60.0)


def test_budget_rejects_non_positive_divisor() -> None:
    with pytest.raises(ValueError, match="divisor"):
        bg_ails_budget_seconds(100, min_budget=60.0, divisor=0.0)


def test_coupling_ok_when_equal() -> None:
    check_bg_ails_divisor_coupling(
        subcluster_time_per_customer=0.06,
        divisor_assumes_time_per_customer=0.06,
    )


def test_coupling_fails_when_mismatched() -> None:
    with pytest.raises(ValueError, match="subcluster_time_per_customer"):
        check_bg_ails_divisor_coupling(
            subcluster_time_per_customer=0.05,
            divisor_assumes_time_per_customer=0.06,
        )
