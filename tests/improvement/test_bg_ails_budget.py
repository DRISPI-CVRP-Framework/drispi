"""Unit tests for the predicted-DR-wall BG-AILS budget."""

from __future__ import annotations

import pytest

from drispi.improvement.bg_ails_budget import (
    bg_ails_budget_seconds,
    lockstep_wave_sum,
    predicted_dr_wall_seconds,
    subcluster_budget_seconds,
)

# sizes 2000/1000/500 at rate 0.06 -> budgets 120/60/30.
_SIZES_DESC = [2000, 1000, 500]
_SIZES_ASC = [500, 1000, 2000]


def test_subcluster_budget_floor_and_scale() -> None:
    assert subcluster_budget_seconds(10, rate=0.06, floor=5.0) == 5.0
    assert subcluster_budget_seconds(2000, rate=0.06, floor=5.0) == pytest.approx(120.0)
    with pytest.raises(ValueError, match="size"):
        subcluster_budget_seconds(-1, rate=0.06, floor=5.0)


def test_lockstep_wave_sum_packs_in_submit_order() -> None:
    # Two waves of 2: max(120, 60) + max(30) — order matters, no sorting.
    assert lockstep_wave_sum([120.0, 60.0, 30.0], 2) == pytest.approx(150.0)
    # Same budgets sorted differently pack differently (submit order is law).
    assert lockstep_wave_sum([30.0, 60.0, 120.0], 2) == pytest.approx(180.0)
    assert lockstep_wave_sum([], 4) == 0.0
    assert lockstep_wave_sum([10.0, 20.0], 8) == 20.0
    with pytest.raises(ValueError, match="n_workers"):
        lockstep_wave_sum([1.0], 0)


def test_predicted_dr_wall_multiplicative_wave_model() -> None:
    # lockstep 150, 2 waves -> 0.976 * 2**(-0.180) * 150 ≈ 129.198.
    pred = predicted_dr_wall_seconds(
        _SIZES_DESC, n_workers=2, rate=0.06, sub_floor=5.0
    )
    assert pred == pytest.approx(0.976 * (2 ** -0.180) * 150.0)
    assert pred == pytest.approx(129.228, abs=0.01)


def test_predicted_dr_wall_empty_is_zero() -> None:
    assert (
        predicted_dr_wall_seconds([], n_workers=2, rate=0.06, sub_floor=5.0) == 0.0
    )


def test_predicted_dr_wall_is_order_sensitive() -> None:
    pred_desc = predicted_dr_wall_seconds(
        _SIZES_DESC, n_workers=2, rate=0.06, sub_floor=5.0
    )
    pred_asc = predicted_dr_wall_seconds(
        _SIZES_ASC, n_workers=2, rate=0.06, sub_floor=5.0
    )
    assert pred_asc == pytest.approx(0.976 * (2 ** -0.180) * 180.0)
    assert pred_asc != pytest.approx(pred_desc)


def test_predicted_dr_wall_is_wave_count_sensitive() -> None:
    # Same sizes, 1 worker -> 3 waves, lockstep 210.
    pred_3 = predicted_dr_wall_seconds(
        _SIZES_DESC, n_workers=1, rate=0.06, sub_floor=5.0
    )
    pred_2 = predicted_dr_wall_seconds(
        _SIZES_DESC, n_workers=2, rate=0.06, sub_floor=5.0
    )
    assert pred_3 == pytest.approx(0.976 * (3 ** -0.180) * 210.0)
    assert pred_3 != pytest.approx(pred_2)


def test_budget_hits_floor_for_small_partitions() -> None:
    # One 6-customer wave: budget 5 s floor, pred ~4.88 s -> floor 60 binds.
    budget = bg_ails_budget_seconds(
        [6],
        n_workers=6,
        rate=0.06,
        sub_floor=5.0,
        floor=60.0,
        margin=1.0,
    )
    assert budget == 60.0


def test_budget_tracks_predicted_wall_above_floor() -> None:
    pred = predicted_dr_wall_seconds(
        _SIZES_DESC, n_workers=2, rate=0.06, sub_floor=5.0
    )
    budget = bg_ails_budget_seconds(
        _SIZES_DESC,
        n_workers=2,
        rate=0.06,
        sub_floor=5.0,
        floor=60.0,
        margin=1.0,
    )
    assert budget == pytest.approx(pred)
    assert budget > 60.0


def test_budget_rejects_negative_floor() -> None:
    with pytest.raises(ValueError, match="floor"):
        bg_ails_budget_seconds(
            [100],
            n_workers=1,
            rate=0.06,
            sub_floor=5.0,
            floor=-1.0,
            margin=1.0,
        )
