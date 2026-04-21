"""Conditional weight model for method selection given upstream AOLS choices."""

from __future__ import annotations

from typing import Any


class ConditionalWeightModel:
    """Context-sensitive discrete choices (e.g. clustering method given decomposition type)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def select(self, context: dict[str, Any]) -> str:
        """Choose a method identifier conditioned on ``context`` (e.g. decomp_type)."""
        # TODO: nested weight tables or masked softmax
        raise NotImplementedError

    def update(self, context: dict[str, Any], operator_id: str, reward: float, decay: float = 0.8) -> None:
        """Apply reward to the arm taken under ``context``."""
        # TODO: route update to appropriate sub-model
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        """Serialize internal state."""
        # TODO: dump nested weights
        raise NotImplementedError

    def load_snapshot(self, snap: dict[str, Any]) -> None:
        """Restore from ``snapshot()`` output."""
        # TODO: restore nested weights
        raise NotImplementedError
