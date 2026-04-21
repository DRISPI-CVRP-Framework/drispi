"""Reward based on optimality gap reduction."""

from __future__ import annotations

from typing import Any

from drispi.haos.reward.base import BaseReward


class GapImprovementReward(BaseReward):
    """Positive reward when LP/MIP gap shrinks."""

    def compute(self, context: dict[str, Any]) -> float:
        # TODO: compare gap before vs after segment
        raise NotImplementedError
