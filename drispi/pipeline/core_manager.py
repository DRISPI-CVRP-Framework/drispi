"""Shared dynamic core budget for parallel subcluster solving."""

from __future__ import annotations

import threading


class CoreManager:
    """
    Shared dynamic core budget for parallel subcluster solving.

    Tracks available cores across all concurrently running pipeline
    instances. Each pipeline instance requests workers at the start
    of the subcluster phase and releases them at the end.
    """

    def __init__(self, total_cores: int, nominal_per_instance: int) -> None:
        if total_cores < 1:
            raise ValueError("total_cores must be at least 1")
        if nominal_per_instance < 1:
            raise ValueError("nominal_per_instance must be at least 1")
        self._total = total_cores
        self.nominal_per_instance = nominal_per_instance
        self._available = total_cores
        self._allocations: dict[str, int] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def acquire(self, instance_id: str, n_requested: int) -> int:
        """
        Acquire as many cores as currently available for this instance.

        Blocks only when ``available == 0``. Returns ``min(n_requested, available)``
        once at least one core is free. Thread-safe.
        """
        if n_requested < 1:
            raise ValueError("n_requested must be at least 1")
        with self._condition:
            while self._available == 0:
                self._condition.wait()
            granted = min(n_requested, self._available)
            self._available -= granted
            self._allocations[instance_id] = self._allocations.get(instance_id, 0) + granted
            return granted

    def release(self, instance_id: str, n_cores: int) -> None:
        """Release n_cores back to the pool after subcluster phase ends. Thread-safe."""
        if n_cores < 1:
            return
        with self._condition:
            self._available = min(self._total, self._available + n_cores)
            current = self._allocations.get(instance_id, 0)
            remaining = current - n_cores
            if remaining <= 0:
                self._allocations.pop(instance_id, None)
            else:
                self._allocations[instance_id] = remaining
            self._condition.notify_all()

    def available(self) -> int:
        """Return current number of available cores."""
        with self._lock:
            return self._available

    def status(self) -> dict:
        """Return current allocation status for logging."""
        with self._lock:
            return {
                "total": self._total,
                "available": self._available,
                "allocations": dict(self._allocations),
            }
