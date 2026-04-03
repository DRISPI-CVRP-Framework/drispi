"""Weighted sum or product of sub-rewards."""

from __future__ import annotations

from typing import Any

from drispi.aols.reward.base import BaseReward


class CompositeReward(BaseReward):
    """Combine multiple ``BaseReward`` instances."""

    def __init__(self, parts: list[tuple[BaseReward, float]]) -> None:
        self.parts = parts

    def compute(self, context: dict[str, Any]) -> float:
        # TODO: sum weight * part.compute(context)
        raise NotImplementedError
