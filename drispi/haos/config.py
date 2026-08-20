from __future__ import annotations

import math
from dataclasses import dataclass, field

from drispi.core.instance import CVRPInstance
from drispi.haos.k_domain import DEFAULT_BASE_ARMS, k_domain, parse_n_kmin


@dataclass
class HAOSRewardConfig:
    """Immediate and deferred HAOS reward scores."""

    reward_new_best: float = 8.0
    reward_improvement: float = 3.0
    reward_no_improvement: float = 1.0
    reward_no_solution: float = 0.0

    deferred_new_best: float = 5.0
    deferred_improvement: float = 2.0
    deferred_no_improvement: float = 0.0


@dataclass
class HAOSConfig:
    """Full configuration for hierarchical adaptive operator selection."""

    decay: float = 0.95
    haos_warmup: int = 10
    starting_weight: float = 10.0
    rewards: HAOSRewardConfig = field(default_factory=HAOSRewardConfig)

    # Scale-adaptive k domain (see drispi.haos.k_domain.k_domain).
    k_base_arms: list[int] = field(default_factory=lambda: list(DEFAULT_BASE_ARMS))
    k_max_arms: int = 10
    k_ext_per_1000: int = 4
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

    def compute_k_values(self, instance: CVRPInstance) -> list[int]:
        """
        Scale-adaptive k domain for this instance (variable arity).

        ``(n, K_min)`` come from the instance name (filename DIMENSION and
        fleet size). For synthetic instances without a parseable name, fall
        back to ``n_customers`` and the minimum-feasible fleet
        ``ceil(sum(demand) / capacity)`` as ``K_min``.
        """
        parsed = parse_n_kmin(instance.name)
        if parsed is not None:
            n, k_min = parsed
        else:
            n = instance.n_customers
            total_demand = sum(instance.demands[c] for c in instance.customers)
            k_min = max(1, math.ceil(total_demand / instance.capacity))
        return k_domain(
            n,
            k_min,
            base_arms=list(self.k_base_arms),
            max_arms=self.k_max_arms,
            ext_per_1000=self.k_ext_per_1000,
        )
