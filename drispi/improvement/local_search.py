"""Classical local search (2-opt, relocate, etc.) stub."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Solution
from drispi.improvement.base import BaseImprovement


class LocalSearchImprovement(BaseImprovement):
    """Deterministic or first-improvement local search."""

    def improve(self, solution: Solution, instance: CVRPInstance, time_limit: float) -> Solution:
        # TODO: intra/inter route operators until local optimum or time
        raise NotImplementedError
