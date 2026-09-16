"""Pipeline-level configuration for DRISPI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DRISPIConfig:
    """Single source of truth for all DRISPI parameters."""

    # ── Stopping criteria ─────────────────────────────────────────────
    time_limit: float = 7200.0
    max_no_improve: int = 100

    # ── Parallelism ───────────────────────────────────────────────────
    # Legacy single-instance worker count when ``cores:`` is absent.
    n_workers: int = 8
    # Optional ``cores:`` block (source of truth when any/all of these are set).
    cores_total: int | None = None
    cores_dri: int | None = None
    cores_bg: int | None = None
    cores_sp: int | None = None
    cores_cpu_list: list[int] | None = None

    # ── SP/SC scheduling ──────────────────────────────────────────────
    # mode: off | sync | async
    # trigger: iteration (sync default / always for sync) | wallclock (async only)
    # warmup_iterations always gates every trigger.
    sp_sc_mode: str = "sync"
    sp_sc_trigger: str = "iteration"
    interval_minutes: float = 20.0
    overlap_policy: str = "skip"
    warmup_iterations: int = 10
    sp_interval: int = 3
    min_coverage: int = 5
    sp_time_limit: float = 300.0
    mip_gap: float = 0.0005

    # ── Decomposition (k domain + subcluster budget) ──────────────────
    # Scale-adaptive HAOS level-1 k domain, computed once per instance at
    # init (see drispi.haos.k_domain). base_arms are clipped by K_min and
    # extended geometrically up to k_ext = ext_per_1000 * round(n/1000),
    # clamped by K_min; at most max_arms.
    decomp_k_base_arms: list[int] = field(
        default_factory=lambda: [2, 3, 4, 6, 8, 10, 12]
    )
    decomp_k_max_arms: int = 10
    decomp_k_ext_per_1000: int = 4
    decomp_k_min_routes_per_cluster: int | None = None
    decomp_k_min_arm_spacing: int | None = None
    # Per-cluster solver budget = max(floor_s, size * rate_s_per_customer).
    subcluster_rate_s_per_customer: float = 0.06
    subcluster_floor_s: float = 5.0

    # ── BG-AILS ───────────────────────────────────────────────────────
    # mode: sync (inline on main thread, today's behaviour) | async
    # (perturb+improve on a dedicated BG worker, overlapping decompose+route
    # of the next iteration; requires cores.bg >= 1 for pinning).
    bg_ails_mode: str = "sync"
    # Consecutive/total BG worker crashes tolerated before the run fails.
    bg_ails_crash_threshold: int = 1
    # budget = max(floor_s, margin * predicted_dr_wall) where
    # predicted_dr_wall = scale * n_waves^wave_exponent * lockstep_sum
    # (wall model refit on 835 post-strip pilot iterations).
    bg_ails_budget_mode: str = "predicted_dr_wall"
    bg_ails_budget_floor_s: float = 60.0
    bg_ails_budget_margin: float = 1.0
    bg_ails_wall_model_scale: float = 0.976
    bg_ails_wall_model_wave_exponent: float = -0.180
    bg_ails_initial_omega: float = 10.0
    bg_ails_boundary_threshold: float = 0.5
    bg_ails_small_cluster_cap: int = 20
    bg_ails_small_cluster_alpha: float = 0.5
    bg_ails_pair_selection: str = "greedy"  # "stochastic" | "greedy"
    bg_ails_n_chains_mode: str = "k"  # "k_minus_1" | "k"
    # Arm-E production defaults: unique-first kick + first-LS boundary mask.
    bg_ails_unique_first_routes: bool = True
    bg_ails_boundary_mask_first_ls: bool = True

    # ── Standard improvement ──────────────────────────────────────────
    standard_improvement_time_limit: float = 120.0

    # ── Route pool ────────────────────────────────────────────────────
    max_pool_size: int = 10000
    pool_diversity_weight: float = 1.0

    # ── HAOS ──────────────────────────────────────────────────────────
    haos_decay: float = 0.95
    haos_warmup: int = 10
    haos_starting_weight: float = 10.0
    haos_min_weight_k: float = 0.025
    haos_min_weight_lambda: float = 0.05
    haos_min_weight_paradigm: float = 0.10
    haos_min_weight_vertex_method: float = 0.05
    haos_min_weight_route_method: float = 0.05
    haos_min_weight_solver: float = 0.05
    haos_lambda_demand_values: list[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    haos_paradigm_values: list[str] = field(
        default_factory=lambda: ["vertex", "route"]
    )
    haos_vertex_method_values: list[str] = field(
        default_factory=lambda: [
            "kmeans",
            "agglomerative_avg",
            "agglomerative_complete",
            "kmedoids",
            "fcm",
        ]
    )
    haos_route_method_values: list[str] = field(
        default_factory=lambda: [
            "kmeans",
            "agglomerative_avg",
            "agglomerative_complete",
        ]
    )
    haos_solver_values: list[str] = field(
        default_factory=lambda: ["pyvrp", "filo", "filo2", "ails2"]
    )

    # ── HAOS rewards ──────────────────────────────────────────────────
    haos_reward_new_best: float = 8.0
    haos_reward_improvement: float = 3.0
    haos_reward_no_improvement: float = 1.0
    haos_reward_no_solution: float = 0.0
    haos_deferred_new_best: float = 5.0
    haos_deferred_improvement: float = 2.0
    haos_deferred_no_improvement: float = 0.0

    # ── Seed ──────────────────────────────────────────────────────────
    seed: int = 123

    # ── Output ────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("artifacts/runs"))
    run_analysis: bool = False
