"""Scale-adaptive HAOS level-1 k domain, computed once per instance at init.

The imbalance constants (``imbalance_c``, ``imbalance_beta``) come from the
100-instance XL campaign fit ``max_share = c * k**(-beta)`` and are used only
to compute the subproblem-size floor ``k_lo`` here. They are never applied to
actual partitions at runtime — those are measured directly.
"""

from __future__ import annotations

import math
import re

DEFAULT_BASE_ARMS = [2, 3, 4, 6, 8, 10, 12]

_NAME_RE = re.compile(r"(?:XL?)-n(\d+)-k(\d+)")


def parse_n_kmin(instance_name: str) -> tuple[int, int] | None:
    """Parse ``(n, k_min)`` from an X/XL instance name (filename DIMENSION).

    ``XL-n5288-k1246`` / ``X-n5288-k1246`` → ``(5288, 1246)``. Returns ``None``
    for names that do not follow the CVRPLib convention (synthetic tests).
    """
    match = _NAME_RE.search(instance_name or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def k_domain(
    n: int,
    k_min: int,
    *,
    base_arms: list[int] | None = None,
    max_arms: int = 10,
    s_hi: float = 2500.0,
    ext_per_1000: int = 6,
    imbalance_c: float = 0.980,
    imbalance_beta: float = 0.725,
) -> list[int]:
    """Deterministic k domain for HAOS level 1, fixed at initialisation.

    ``k_lo`` is a subproblem-size *filter*, not an arm: it is the smallest k
    whose predicted largest subcluster (``imbalance_c * n * k**-imbalance_beta``)
    stays within ``s_hi`` customers. ``ceil`` is deliberate — flooring would
    admit exactly the subproblem size the threshold exists to exclude.

    ``k_ext = ext_per_1000 * round(n / 1000)`` targets a roughly constant
    largest subcluster at the top arm; it is clamped by ``k_min`` (the filename
    fleet size, a lower bound on the live route count).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if k_min < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")
    if max_arms < 1:
        raise ValueError(f"max_arms must be >= 1, got {max_arms}")
    base = list(DEFAULT_BASE_ARMS if base_arms is None else base_arms)

    k_lo = max(2, math.ceil((imbalance_c * n / s_hi) ** (1.0 / imbalance_beta)))
    if ext_per_1000 > 0:
        k_ext = min(ext_per_1000 * max(1, round(n / 1000)), k_min)
        k_lo = min(k_lo, k_ext)  # the floor must never cross the ceiling
    else:
        k_ext = 0  # extension disabled (e.g. pilot drop-low cell)
        k_lo = min(k_lo, k_min)

    dom = [k for k in base if k_lo <= k <= k_min]
    if not dom:
        dom = [min(k_lo, k_min)]  # only when k_lo exceeds the whole base
    if 2 <= k_min < 12 and k_min not in dom:
        dom.append(k_min)

    slots = max_arms - len(dom)
    if dom and k_ext > max(dom) and slots > 0:
        top = max(dom)
        for j in range(1, slots + 1):
            k = round(top * (k_ext / top) ** (j / slots))
            if k not in dom and k <= k_ext:
                dom.append(k)
    return sorted(dom)
