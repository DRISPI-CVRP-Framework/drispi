"""Discrete operator weighting for AOLS."""

from __future__ import annotations

from typing import Any

from drispi.core.types import WeightVector


class DiscreteWeightModel:
    """Maintains a weight vector over a fixed set of operator IDs."""

    def __init__(self, operator_ids: list[str], initial_weights: WeightVector | None = None) -> None:
        self._operator_ids = list(operator_ids)
        # TODO: initialize numpy array aligned with operator_ids
        self._weights: WeightVector | None = initial_weights

    def weights(self) -> WeightVector:
        """Current weight vector (same order as ``operator_ids``)."""
        # TODO: return copy of internal weights
        raise NotImplementedError

    def update(self, operator_id: str, reward: float, decay: float = 0.8) -> None:
        """Apply exponential smoothing / reward to ``operator_id``."""
        # TODO: map id to index, update rule with decay
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable state."""
        # TODO: store ids and weight list
        raise NotImplementedError

    def load_snapshot(self, snap: dict[str, Any]) -> None:
        """Restore weights from ``snapshot()``."""
        # TODO: validate keys and assign
        raise NotImplementedError
