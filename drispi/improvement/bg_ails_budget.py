"""Single source of truth for the BG-AILS time budget (instance-size formula)."""

from __future__ import annotations

import math


def bg_ails_budget_seconds(
    n: int,
    *,
    min_budget: float,
    divisor: float,
) -> float:
    """
    Return ``max(min_budget, n / divisor)``.

    ``n`` is the instance customer count. The budget does not depend on DR wall
    time, cluster sizes, ``k``, or ``n_chains``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    if min_budget < 0:
        raise ValueError(f"min_budget must be non-negative, got {min_budget}")
    return max(float(min_budget), float(n) / float(divisor))


def check_bg_ails_divisor_coupling(
    *,
    subcluster_time_per_customer: float,
    divisor_assumes_time_per_customer: float,
) -> None:
    """
    Fail if routing per-customer time differs from the value the divisor assumes.

    ``bg_ails_divisor`` is calibrated under ``divisor_assumes_time_per_customer``.
    Changing ``subcluster_time_per_customer`` without updating that assumption
    (or re-deriving the divisor) is an inconsistency.
    """
    if not math.isclose(
        subcluster_time_per_customer,
        divisor_assumes_time_per_customer,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "subcluster_time_per_customer="
            f"{subcluster_time_per_customer!r} differs from "
            "bg_ails_divisor_assumes_time_per_customer="
            f"{divisor_assumes_time_per_customer!r}; "
            "bg_ails_divisor was calibrated under the assumed per-customer "
            "routing budget — update the assumption or re-derive the divisor"
        )
