"""Unit tests for the predicted-DR-wall BG-AILS budget."""

from __future__ import annotations

import pytest

from drispi.improvement.bg_ails_budget import (
    bg_ails_budget_seconds,
    lockstep_wave_sum,
    predicted_dr_wall_seconds,
    subcluster_budget_seconds,
)


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


def test_predicted_dr_wall_from_cluster_sizes() -> None:
    # sizes 2000/1000/500 at rate 0.06 -> budgets 120/60/30; 2 workers ->
    # wave sum 150; wall = 0.991 * 150 - 2.0 = 146.65.
    pred = predicted_dr_wall_seconds(
        [2000, 1000, 500],
        n_workers=2,
        rate=0.06,
        sub_floor=5.0,
        slope=0.991,
        intercept=-2.0,
    )
    assert pred == pytest.approx(146.65)
    assert (
        predicted_dr_wall_seconds(
            [], n_workers=2, rate=0.06, sub_floor=5.0, slope=0.991, intercept=-2.0
        )
        == 0.0
    )


def test_predicted_dr_wall_never_negative() -> None:
    # Tiny wave sum with negative intercept must clamp at zero.
    pred = predicted_dr_wall_seconds(
        [1], n_workers=1, rate=0.06, sub_floor=1.0, slope=0.991, intercept=-2.0
    )
    assert pred == 0.0


def test_budget_hits_floor_for_small_partitions() -> None:
    # One 6-customer wave: budget 5 s floor, wall ~2.955 s -> floor 60 binds.
    budget = bg_ails_budget_seconds(
        [6],
        n_workers=6,
        rate=0.06,
        sub_floor=5.0,
        floor=60.0,
        margin=0.95,
        slope=0.991,
        intercept=-2.0,
    )
    assert budget == 60.0


def test_budget_tracks_predicted_wall_above_floor() -> None:
    budget = bg_ails_budget_seconds(
        [2000, 1000, 500],
        n_workers=2,
        rate=0.06,
        sub_floor=5.0,
        floor=60.0,
        margin=0.95,
        slope=0.991,
        intercept=-2.0,
    )
    assert budget == pytest.approx(0.95 * 146.65)


def test_budget_rejects_negative_floor() -> None:
    with pytest.raises(ValueError, match="floor"):
        bg_ails_budget_seconds(
            [100],
            n_workers=1,
            rate=0.06,
            sub_floor=5.0,
            floor=-1.0,
            margin=0.95,
            slope=0.991,
            intercept=-2.0,
        )
