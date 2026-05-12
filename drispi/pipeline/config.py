"""Pipeline-level configuration for DRISPI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from drispi.haos.config import HAOSConfig


@dataclass
class DRISPIConfig:
    """Full configuration for the DRISPI pipeline."""

    time_limit: float = 3600.0
    max_no_improve: int = 100

    n_workers: int = 4

    warmup_iterations: int = 10
    sp_interval: int = 3
    min_coverage: int = 5
    sp_time_limit: float = 300.0
    mip_gap: float = 0.001

    subcluster_time_per_customer: float = 0.1
    bg_ails_time_limit: float = 90.0
    bg_ails_initial_omega: float = 0.8
    bg_ails_boundary_threshold: float = 0.5
    standard_improvement_time_limit: float = 180.0

    max_pool_size: int = 10000
    pool_diversity_weight: float = 1.0

    haos_config: HAOSConfig = field(default_factory=HAOSConfig)

    output_dir: Path = field(default_factory=lambda: Path("output"))
    seed: int = 420
