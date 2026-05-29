"""Display formatting helpers for the DRISPI monitor."""

from __future__ import annotations

import html
from typing import Any

_METHOD_DISPLAY: dict[str, str] = {
    "agglomerative_avg": "ac_avg",
    "agglomerative_complete": "ac_complete",
    "agglomerative_single": "ac_single",
    "agglomerative_single_linkage": "ac_single",
}


def format_method(method: str | None) -> str:
    if not method:
        return "—"
    return _METHOD_DISPLAY.get(str(method), str(method))


def format_paradigm_short(paradigm: str | None) -> str:
    if not paradigm:
        return "—"
    p = str(paradigm).lower()
    if p in ("vb", "vertex"):
        return "vb"
    if p in ("rb", "route"):
        return "rb"
    return p


def haos_roll_parts(roll: dict[str, Any] | None) -> list[str]:
    """HAOS operator selection fields for display."""
    if roll is None:
        return []
    lam = roll.get("lambda_demand", "—")
    if isinstance(lam, float):
        lam = f"{lam:g}"
    return [
        f"k = {roll.get('k', '—')}",
        f"λ = {lam}",
        format_paradigm_short(str(roll.get("paradigm", ""))),
        format_method(str(roll.get("method", ""))),
        str(roll.get("solver", "—")),
    ]


_BEST_ORIGIN_DISPLAY: dict[str, str] = {
    "bg_ails": "boundary-guided ails",
    "sp_sc": "sp/sc",
    "standard_ails": "post-sc/sp improvement",
}


def format_best_origin(phase_found: str | None) -> str:
    if not phase_found:
        return ""
    key = str(phase_found).strip()
    return _BEST_ORIGIN_DISPLAY.get(key, key.replace("_", " "))


def best_origin_inline(best: dict[str, Any] | None) -> str:
    if best is None:
        return ""
    origin = format_best_origin(str(best.get("phase_found") or ""))
    if not origin:
        return ""
    iter_found = best.get("iteration_found")
    if iter_found is not None:
        return f"origin: {origin} · iteration {int(iter_found) + 1}"
    return f"origin: {origin}"


def haos_roll_inline(roll: dict[str, Any] | None) -> str:
    """Single-line HAOS roll with interpunct separators."""
    parts = haos_roll_parts(roll)
    if not parts:
        return ""
    return " · ".join(parts)
