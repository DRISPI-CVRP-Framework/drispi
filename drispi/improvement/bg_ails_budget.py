"""Single source of truth for the BG-AILS time budget (predicted DR wall).

The budget follows the wall-time model refit on 835 post-strip pilot
iterations::

    dr_wall ≈ scale * n_waves**wave_exponent * sum_over_waves(max(budget_i in wave))

Waves are the lockstep packing of per-cluster budgets in **submit order**
(``ProcessPoolExecutor`` is a greedy queue). Do not sort budgets before packing.
A linear alternative ``0.5775*lockstep - 4.749*(n_waves-1) + 38.56`` was
recorded and is not shipped.
"""

from __future__ import annotations

import math

DEFAULT_WALL_MODEL_SCALE = 0.976
DEFAULT_WALL_MODEL_WAVE_EXPONENT = -0.180


def subcluster_budget_seconds(size: int, *, rate: float, floor: float) -> float:
    """Per-cluster solver budget: ``max(floor, size * rate)``."""
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    return max(float(floor), float(size) * float(rate))


def lockstep_wave_sum(budgets: list[float], n_workers: int) -> float:
    """Sum of per-wave maxima with waves packed in submit order.

    Wave ``w`` is ``budgets[w * n_workers : (w + 1) * n_workers]``. Do not
    sort budgets before calling.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    total = 0.0
    for start in range(0, len(budgets), n_workers):
        total += max(budgets[start : start + n_workers])
    return total


def predicted_dr_wall_seconds(
    cluster_sizes: list[int],
    *,
    n_workers: int,
    rate: float,
    sub_floor: float,
    scale: float = DEFAULT_WALL_MODEL_SCALE,
    wave_exponent: float = DEFAULT_WALL_MODEL_WAVE_EXPONENT,
) -> float:
    """Predicted route-phase wall from cluster sizes in submit order."""
    if not cluster_sizes:
        return 0.0
    budgets = [
        subcluster_budget_seconds(s, rate=rate, floor=sub_floor) for s in cluster_sizes
    ]
    lockstep = lockstep_wave_sum(budgets, n_workers)
    n_waves = max(1, math.ceil(len(cluster_sizes) / n_workers))
    return float(scale) * (n_waves ** float(wave_exponent)) * lockstep


def bg_ails_budget_seconds(
    cluster_sizes: list[int],
    *,
    n_workers: int,
    rate: float,
    sub_floor: float,
    floor: float,
    margin: float,
    scale: float = DEFAULT_WALL_MODEL_SCALE,
    wave_exponent: float = DEFAULT_WALL_MODEL_WAVE_EXPONENT,
) -> float:
    """BG-AILS budget: ``max(floor, margin * predicted_dr_wall)``.

    ``cluster_sizes`` must be in ``ProcessPoolExecutor`` submit order (i.e.
    partition order) and ``n_workers`` must be the granted DRI worker count.
    """
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}")
    predicted = predicted_dr_wall_seconds(
        cluster_sizes,
        n_workers=n_workers,
        rate=rate,
        sub_floor=sub_floor,
        scale=scale,
        wave_exponent=wave_exponent,
    )
    return max(float(floor), float(margin) * predicted)
