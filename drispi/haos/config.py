from __future__ import annotations

import math
from dataclasses import dataclass, field

from drispi.core.instance import CVRPInstance


@dataclass
class HAOSRewardConfig:
    """Immediate and deferred HAOS reward scores."""

    reward_new_best: float = 5.0
    reward_improvement: float = 2
    reward_no_improvement: float = 1
    reward_no_solution: float = 0.0

    deferred_new_best: float = 3.0
    deferred_improvement: float = 1
    deferred_no_improvement: float = 0.0


@dataclass
class HAOSConfig:
    """Full configuration for hierarchical adaptive operator selection."""

    decay: float = 0.95
    haos_warmup: int = 10
    rewards: HAOSRewardConfig = field(default_factory=HAOSRewardConfig)

    k_candidates: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    )
    min_weight_k: float = 0.025

    lambda_demand_values: list[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    min_weight_lambda: float = 0.05

    paradigm_values: list[str] = field(default_factory=lambda: ["vertex", "route"])
    min_weight_paradigm: float = 0.10

    vertex_method_values: list[str] = field(
        default_factory=lambda: [
            "kmeans",
            "agglomerative_avg",
            "agglomerative_complete",
            "agglomerative_single",
            "kmedoids",
            "fcm",
            # "spectral",  # disabled in HAOS roll: too slow on large n with default sklearn settings
        ]
    )
    min_weight_vertex_method: float = 0.05

    route_method_values: list[str] = field(
        default_factory=lambda: [
            "kmeans",
            "agglomerative_avg",
            "agglomerative_complete",
            "agglomerative_single",
        ]
    )
    min_weight_route_method: float = 0.05

    solver_values: list[str] = field(
        default_factory=lambda: ["pyvrp", "filo", "filo2", "ails2"]
    )
    min_weight_solver: float = 0.05

    @staticmethod
    def compute_k_values(
        instance: CVRPInstance,
        candidates: list[int] | None = None,
    ) -> list[int]:
        """
        Sorted ``k_candidates`` entries that do not exceed minimum-feasible fleet
        size ``ceil(sum(demand) / capacity)``.

        Values in ``k_candidates`` larger than that bound are dropped; nothing
        else (e.g. benchmark fleet ``k``) is appended. At least one candidate must
        be ``<=`` that bound or HAOS cannot build the k wheel.
        """
        if candidates is None:
            candidates = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
        total_demand = sum(instance.demands[c] for c in instance.customers)
        k_max = math.ceil(total_demand / instance.capacity)
        return sorted({k for k in candidates if k <= k_max})
