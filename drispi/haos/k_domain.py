"""Scale-adaptive HAOS level-1 k domain, computed once per instance at init.

The campaign/pilot imbalance law ``max_share = 1.062 * k**(-0.795)`` is a
recorded measurement only — it is not applied here. The domain is the
configured base arms, clipped by filename ``K_min``, plus an optional
geometric extension up to ``k_ext``.
"""

from __future__ import annotations

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


def k_ext_value(n: int, k_min: int, ext_per_1000: int) -> int:
    """Ceiling of the geometric extension, or 0 when extension is disabled."""
    if ext_per_1000 <= 0:
        return 0
    return min(ext_per_1000 * max(1, round(n / 1000)), k_min)


def k_domain(
    n: int,
    k_min: int,
    *,
    base_arms: list[int] | None = None,
    max_arms: int = 10,
    ext_per_1000: int = 6,
) -> list[int]:
    """Deterministic k domain for HAOS level 1, fixed at initialisation.

    Base arms are those in ``base_arms`` (default ``DEFAULT_BASE_ARMS``) with
    ``k <= k_min``. When ``2 <= k_min < 12`` and ``k_min`` is not already in
    the base, it is appended so a clipped fleet size stays playable.

    ``k_ext = ext_per_1000 * round(n / 1000)``, clamped by ``k_min``. If
    ``ext_per_1000 <= 0``, extension is a no-op (no division by empty range).
    Remaining slots up to ``max_arms`` are filled with a geometric ladder from
    the current top to ``k_ext``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if k_min < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")
    if max_arms < 1:
        raise ValueError(f"max_arms must be >= 1, got {max_arms}")
    base = list(DEFAULT_BASE_ARMS if base_arms is None else base_arms)

    dom = [k for k in base if k <= k_min]
    if 2 <= k_min < 12 and k_min not in dom:
        dom.append(k_min)
    if not dom:
        dom = [k_min]  # K_min=1 synthetic: only legal arm

    k_ext = k_ext_value(n, k_min, ext_per_1000)
    slots = max_arms - len(dom)
    if k_ext > max(dom) and slots > 0:
        top = max(dom)
        for j in range(1, slots + 1):
            k = round(top * (k_ext / top) ** (j / slots))
            if k not in dom and k <= k_ext:
                dom.append(k)
    return sorted(dom)
