"""Scale-adaptive HAOS level-1 k domain, computed once per instance at init.

The campaign/pilot imbalance law ``max_share = 1.062 * k**(-0.795)`` is a
recorded measurement only — it is not applied here. The domain is the
configured base arms, clipped by filename ``K_min``, plus an optional
geometric extension up to ``k_ext``. Extension arms are lower-bounded by
``min_routes_per_cluster`` (routes per subcluster) and spaced by
``min_arm_spacing``. The route clamp is applied to ``k_ext`` unconditionally,
so ``k_ext`` changes on 24 of 100 XL instances while the domain changes on
19. ``k_ext`` is not a change-detector; compare ``k_domain``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DEFAULT_BASE_ARMS = [2, 3, 4, 6, 8, 10, 12]

_NAME_RE = re.compile(r"(?:XL?)-n(\d+)-k(\d+)")

KExtBound = Literal["n_bound", "route_bound", "kmin_bound", "disabled"]


@dataclass(frozen=True)
class KDomainResolution:
    """Resolved level-1 domain plus the k_ext audit fields."""

    domain: list[int]
    k_ext: int
    k_ext_bound: KExtBound
    spacing_rejected: int


def parse_n_kmin(instance_name: str) -> tuple[int, int] | None:
    """Parse ``(n, k_min)`` from an X/XL instance name (filename DIMENSION).

    ``XL-n5288-k1246`` / ``X-n5288-k1246`` → ``(5288, 1246)``. Returns ``None``
    for names that do not follow the CVRPLib convention (synthetic tests).
    """
    match = _NAME_RE.search(instance_name or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def k_ext_value(
    n: int,
    k_min: int,
    ext_per_1000: int,
    *,
    min_routes_per_cluster: int,
) -> int:
    """Ceiling of the geometric extension, or 0 when extension is disabled."""
    if min_routes_per_cluster < 0:
        raise ValueError(
            f"min_routes_per_cluster must be >= 0, got {min_routes_per_cluster}"
        )
    if ext_per_1000 <= 0:
        return 0
    k_ext = min(ext_per_1000 * max(1, round(n / 1000)), k_min)
    if min_routes_per_cluster > 0:
        k_ext = min(k_ext, max(1, k_min // min_routes_per_cluster))
    return k_ext


def k_ext_bound(
    n: int,
    k_min: int,
    ext_per_1000: int,
    *,
    min_routes_per_cluster: int,
) -> KExtBound:
    """Which term produced the resolved ``k_ext`` (ties keep the earlier term)."""
    if min_routes_per_cluster < 0:
        raise ValueError(
            f"min_routes_per_cluster must be >= 0, got {min_routes_per_cluster}"
        )
    if ext_per_1000 <= 0:
        return "disabled"
    n_bound = ext_per_1000 * max(1, round(n / 1000))
    kmin_bound = k_min
    k_ext = min(n_bound, kmin_bound)
    bound: KExtBound = "n_bound" if n_bound <= kmin_bound else "kmin_bound"
    if min_routes_per_cluster > 0:
        route_bound = max(1, k_min // min_routes_per_cluster)
        if route_bound < k_ext:
            return "route_bound"
    return bound


def resolve_k_domain(
    n: int,
    k_min: int,
    *,
    min_routes_per_cluster: int,
    min_arm_spacing: int,
    base_arms: list[int] | None = None,
    max_arms: int = 10,
    ext_per_1000: int = 4,
) -> KDomainResolution:
    """Deterministic k domain for HAOS level 1, fixed at initialisation.

    Base arms are those in ``base_arms`` (default ``DEFAULT_BASE_ARMS``) with
    ``k <= k_min``. When ``2 <= k_min < 12`` and ``k_min`` is not already in
    the base, it is appended so a clipped fleet size stays playable.

    ``k_ext = ext_per_1000 * round(n / 1000)``, clamped by ``k_min``, then by
    ``max(1, k_min // min_routes_per_cluster)`` when that clamp is enabled.
    If ``ext_per_1000 <= 0``, extension is a no-op (no division by empty range).
    Remaining slots up to ``max_arms`` are filled with a geometric ladder from
    the current top to ``k_ext``. A ladder candidate is accepted only if it is
    at least ``min_arm_spacing`` above the largest arm accepted so far.
    Rejected slots are not refilled.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if k_min < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")
    if max_arms < 1:
        raise ValueError(f"max_arms must be >= 1, got {max_arms}")
    if min_arm_spacing < 1:
        raise ValueError(f"min_arm_spacing must be >= 1, got {min_arm_spacing}")
    base = list(DEFAULT_BASE_ARMS if base_arms is None else base_arms)

    dom = [k for k in base if k <= k_min]
    if 2 <= k_min < 12 and k_min not in dom:
        dom.append(k_min)
    if not dom:
        dom = [k_min]  # K_min=1 synthetic: only legal arm

    k_ext = k_ext_value(
        n, k_min, ext_per_1000, min_routes_per_cluster=min_routes_per_cluster
    )
    bound = k_ext_bound(
        n, k_min, ext_per_1000, min_routes_per_cluster=min_routes_per_cluster
    )
    spacing_rejected = 0
    slots = max_arms - len(dom)
    if k_ext > max(dom) and slots > 0:
        top = max(dom)
        for j in range(1, slots + 1):
            k = round(top * (k_ext / top) ** (j / slots))
            if k <= k_ext and k >= max(dom) + min_arm_spacing:
                dom.append(k)
            elif k <= k_ext:
                spacing_rejected += 1
    return KDomainResolution(
        domain=sorted(dom),
        k_ext=k_ext,
        k_ext_bound=bound,
        spacing_rejected=spacing_rejected,
    )


def k_domain(
    n: int,
    k_min: int,
    *,
    min_routes_per_cluster: int,
    min_arm_spacing: int,
    base_arms: list[int] | None = None,
    max_arms: int = 10,
    ext_per_1000: int = 4,
) -> list[int]:
    return resolve_k_domain(
        n,
        k_min,
        min_routes_per_cluster=min_routes_per_cluster,
        min_arm_spacing=min_arm_spacing,
        base_arms=base_arms,
        max_arms=max_arms,
        ext_per_1000=ext_per_1000,
    ).domain
