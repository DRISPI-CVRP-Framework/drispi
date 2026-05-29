"""Tests for current-iteration phase minimum display duration."""

from __future__ import annotations

import time

from drispi.dashboard.phase_hold import (
    MIN_CURRENT_PHASE_DISPLAY_S,
    resolve_current_iteration_display,
    snapshot_fingerprint,
)


def _snap(iteration: int, phase: int, stage: str = "main", ts: str = "10:00:00") -> dict:
    payload: dict = {
        "iteration": iteration,
        "phase_num": phase,
        "phase_name": "test",
        "timestamp": ts,
    }
    if stage != "main":
        payload["bg_stage"] = stage
    return payload


def test_snapshot_fingerprint_distinguishes_bg_stages() -> None:
    a = _snap(0, 3, "pre", "10:01:00")
    b = _snap(0, 3, "post", "10:02:00")
    assert snapshot_fingerprint(a) != snapshot_fingerprint(b)


def test_phase_hold_advances_after_min_duration() -> None:
    now = time.time()
    s1 = _snap(0, 1, ts="10:00:01")
    s2 = _snap(0, 2, ts="10:00:02")

    first, hold = resolve_current_iteration_display([s1], 0, None, now)
    assert first is not None
    assert first["phase_num"] == 1

    soon, hold = resolve_current_iteration_display([s1, s2], 0, hold, now + 1.0)
    assert soon is not None
    assert soon["phase_num"] == 1

    later, hold = resolve_current_iteration_display(
        [s1, s2],
        0,
        hold,
        now + MIN_CURRENT_PHASE_DISPLAY_S + 0.5,
    )
    assert later is not None
    assert later["phase_num"] == 2
