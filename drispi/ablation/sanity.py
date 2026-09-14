"""Diff-based per-customer modification flag (instrumentation sanity check)."""

from __future__ import annotations


def _neighbour_pairs(seqs: list[list[int]]) -> dict[int, frozenset[int]]:
    """Unordered {pred, succ} including depot sentinel 1."""
    out: dict[int, frozenset[int]] = {}
    for seq in seqs:
        if not seq:
            continue
        nodes = [1, *seq, 1]
        for i in range(1, len(nodes) - 1):
            out[nodes[i]] = frozenset({nodes[i - 1], nodes[i + 1]})
    return out


def _route_membership(seqs: list[list[int]]) -> dict[int, frozenset[int]]:
    out: dict[int, frozenset[int]] = {}
    for seq in seqs:
        s = frozenset(seq)
        for c in seq:
            out[c] = s
    return out


def modified_customers(
    before: list[list[int]],
    after: list[list[int]],
) -> set[int]:
    """Customers whose unordered neighbour pair or route membership changed."""
    nb = _neighbour_pairs(before)
    na = _neighbour_pairs(after)
    mb = _route_membership(before)
    ma = _route_membership(after)
    keys = set(nb) | set(na) | set(mb) | set(ma)
    changed: set[int] = set()
    for c in keys:
        if nb.get(c) != na.get(c) or mb.get(c) != ma.get(c):
            changed.add(c)
    return changed


def assert_touch_covers_modifications(
    changed: set[int],
    accept: dict[int, int],
    perturb: dict[int, int],
) -> None:
    """Fail if a neighbour-pair change has zero accept+perturb touches."""
    missing = [
        c
        for c in changed
        if accept.get(c, 0) + perturb.get(c, 0) == 0
    ]
    if missing:
        raise AssertionError(
            f"customers with a changed neighbour/membership but zero "
            f"accept+perturb touches: {sorted(missing)[:20]} (n={len(missing)})"
        )
