"""Tests for CoreManager dynamic core budget."""

from __future__ import annotations

import threading
import time

import pytest

from drispi.pipeline.core_manager import CoreManager


def test_acquire_returns_min_requested_and_available() -> None:
    mgr = CoreManager(total_cores=8, nominal_per_instance=4)
    assert mgr.acquire("a", 5) == 5
    assert mgr.acquire("b", 10) == 3
    assert mgr.available() == 0


def test_release_restores_available() -> None:
    mgr = CoreManager(total_cores=4, nominal_per_instance=2)
    n = mgr.acquire("a", 3)
    assert mgr.available() == 1
    mgr.release("a", n)
    assert mgr.available() == 4


def test_acquire_blocks_when_exhausted() -> None:
    mgr = CoreManager(total_cores=2, nominal_per_instance=1)
    mgr.acquire("a", 2)
    acquired: list[int] = []

    def waiter() -> None:
        acquired.append(mgr.acquire("b", 1))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    assert not acquired
    mgr.release("a", 1)
    t.join(timeout=2.0)
    assert acquired == [1]


def test_concurrent_acquire_never_exceeds_total() -> None:
    mgr = CoreManager(total_cores=6, nominal_per_instance=2)
    peak_in_use = 0
    lock = threading.Lock()

    def worker(i: int) -> None:
        nonlocal peak_in_use
        n = mgr.acquire(f"inst-{i}", 2)
        with lock:
            in_use = mgr.status()["total"] - mgr.status()["available"]
            peak_in_use = max(peak_in_use, in_use)
        mgr.release(f"inst-{i}", n)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert peak_in_use <= 6


def test_status_returns_allocation_map() -> None:
    mgr = CoreManager(total_cores=4, nominal_per_instance=2)
    n = mgr.acquire("x", 2)
    status = mgr.status()
    assert status["total"] == 4
    assert status["available"] == 2
    assert status["allocations"] == {"x": n}
    mgr.release("x", n)


def test_acquire_requires_positive_request() -> None:
    mgr = CoreManager(total_cores=4, nominal_per_instance=2)
    with pytest.raises(ValueError):
        mgr.acquire("a", 0)
