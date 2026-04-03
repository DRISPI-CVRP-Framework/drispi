"""Abstract improvement operator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Solution


class BaseImprovement(ABC):
    """Post-processing or iterative improvement on a solution."""

    @abstractmethod
    def improve(self, solution: Solution, instance: CVRPInstance, time_limit: float) -> Solution:
        """Return an improved copy (or mutate per subclass contract)."""
        raise NotImplementedError
