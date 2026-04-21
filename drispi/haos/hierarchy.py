"""Five-level hierarchical AOLS controller."""

from __future__ import annotations

from typing import Any

from drispi.haos.conditional import ConditionalWeightModel
from drispi.haos.weight_model import DiscreteWeightModel
from drispi.core.types import AOLSDecision


class HierarchicalAOLS:
    """Chains discrete models for k, dissimilarity, decomposition, method, and solver."""

    def __init__(self, k_max: int, solver_ids: list[str]) -> None:
        self.level_k = DiscreteWeightModel([str(k) for k in range(1, k_max + 1)])
        self.level_dissim = DiscreteWeightModel(["spatial", "combined"])
        self.level_decomp = DiscreteWeightModel(["vertex", "route"])
        self.level_method = ConditionalWeightModel()
        self.level_solver = DiscreteWeightModel(list(solver_ids))

    def select(self) -> AOLSDecision:
        """Sample or UCB-select a full hierarchical decision."""
        # TODO: call subordinate models in order, pass context downward
        raise NotImplementedError

    def update(self, decision: AOLSDecision, reward: float) -> None:
        """Propagate ``reward`` to each level according to ``decision``."""
        # TODO: update each DiscreteWeightModel and ConditionalWeightModel arm
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        """Aggregate snapshots from all levels."""
        # TODO: nest level_k.snapshot(), ...
        raise NotImplementedError

    def load_snapshot(self, snap: dict[str, Any]) -> None:
        """Restore all levels from a combined snapshot."""
        # TODO: dispatch to each level's load_snapshot
        raise NotImplementedError
