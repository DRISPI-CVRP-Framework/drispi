"""Reward shaping for feasibility milestones."""

from __future__ import annotations

from typing import Any

from drispi.aols.reward.base import BaseReward


class FeasibilityReward(BaseReward):
    """Bonus when integer feasibility or coverage constraints are met."""

    def compute(self, context: dict[str, Any]) -> float:
        # TODO: map boolean flags in context to scalar
        raise NotImplementedError
