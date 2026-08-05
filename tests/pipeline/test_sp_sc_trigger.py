"""Tests for async SP/SC trigger gating (warmup + wallclock)."""

from __future__ import annotations

import pytest

from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.pipeline import DRISPIPipeline
from drispi.pipeline.sp_sc_async import AsyncSpScController


def _controller(
    *,
    trigger: str = "wallclock",
    warmup_iterations: int = 10,
    sp_interval: int = 3,
    interval_minutes: float = 20.0,
) -> AsyncSpScController:
    return AsyncSpScController(
        sp_cpus=None,
        gurobi_threads=1,
        ails_apc=1,
        trigger=trigger,  # type: ignore[arg-type]
        warmup_iterations=warmup_iterations,
        sp_interval=sp_interval,
        interval_minutes=interval_minutes,
    )


def test_wallclock_respects_warmup() -> None:
    ctrl = _controller(trigger="wallclock", warmup_iterations=10)
    now = 1_000_000.0
    for i in range(10):
        assert ctrl._is_due(iteration=i, now=now) is False
    assert ctrl._is_due(iteration=10, now=now) is True


def test_wallclock_first_fire_asap_after_warmup_then_interval() -> None:
    ctrl = _controller(
        trigger="wallclock",
        warmup_iterations=5,
        interval_minutes=20.0,
    )
    t0 = 1_000_000.0
    assert ctrl._is_due(iteration=5, now=t0) is True
    ctrl._advance_due(t0)
    assert ctrl._is_due(iteration=6, now=t0 + 60.0) is False
    assert ctrl._is_due(iteration=6, now=t0 + 20.0 * 60.0) is True


def test_iteration_trigger_still_uses_sp_interval() -> None:
    ctrl = _controller(trigger="iteration", warmup_iterations=10, sp_interval=3)
    assert ctrl._is_due(iteration=9, now=0.0) is False
    assert ctrl._is_due(iteration=10, now=0.0) is True
    assert ctrl._is_due(iteration=11, now=0.0) is False
    assert ctrl._is_due(iteration=13, now=0.0) is True


def test_is_due_requires_iteration() -> None:
    ctrl = _controller()
    with pytest.raises(ValueError, match="iteration is required"):
        ctrl._is_due(iteration=None, now=0.0)


def test_sync_wallclock_rejected() -> None:
    config = DRISPIConfig(
        sp_sc_mode="sync",
        sp_sc_trigger="wallclock",
    )
    with pytest.raises(ValueError, match="wallclock.*async"):
        DRISPIPipeline(None, config)  # type: ignore[arg-type]
