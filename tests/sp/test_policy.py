"""Tests for SP/SC scheduling and SP eligibility policy."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.route_pool.pool import RoutePool
from drispi.sp.policy import should_run_sp_sc, should_use_sp


def test_should_run_sp_sc_warmup_off() -> None:
    for i in range(10):
        assert should_run_sp_sc(i, warmup_iterations=10, sp_interval=3) is False


def test_should_run_sp_sc_interval_pattern() -> None:
    assert should_run_sp_sc(10, warmup_iterations=10, sp_interval=3) is True
    assert should_run_sp_sc(11, warmup_iterations=10, sp_interval=3) is False
    assert should_run_sp_sc(12, warmup_iterations=10, sp_interval=3) is False
    assert should_run_sp_sc(13, warmup_iterations=10, sp_interval=3) is True


def test_should_use_sp_false_when_uncovered(sp_instance: CVRPInstance) -> None:
    pool = RoutePool()
    pool.add([2, 3], 1.0)
    assert should_use_sp(pool, sp_instance, min_coverage=1) is False


def test_should_use_sp_true_when_full_coverage(sp_instance: CVRPInstance) -> None:
    pool = RoutePool()
    pool.add([2, 3, 4, 5, 6, 7], 10.0)
    assert should_use_sp(pool, sp_instance, min_coverage=1) is True


def test_should_use_sp_false_when_min_coverage_two(sp_instance: CVRPInstance) -> None:
    pool = RoutePool()
    pool.add([2, 3, 4], 10.0)
    pool.add([5, 6, 7], 10.0)
    assert should_use_sp(pool, sp_instance, min_coverage=2) is False
