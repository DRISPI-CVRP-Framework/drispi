"""Biased Greed AILS-style improvement stub."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Solution
from drispi.improvement.base import BaseImprovement


class BgAilsImprovement(BaseImprovement):
    """BG-AILS improvement loop placeholder."""

    def improve(self, solution: Solution, instance: CVRPInstance, time_limit: float) -> Solution:
        # TODO: adaptive large neighborhood search with biased destroy/repair
        raise NotImplementedError
