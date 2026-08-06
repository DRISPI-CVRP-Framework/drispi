"""Diversity metrics for route-pool entries."""

from __future__ import annotations

from drispi.route_pool._entry import RouteEntry


def jaccard_distance(r1: RouteEntry, r2: RouteEntry) -> float:
    """Jaccard distance between two routes based on customer sets."""
    union = r1.customer_set | r2.customer_set
    if not union:
        return 0.0
    symmetric_diff = r1.customer_set ^ r2.customer_set
    return len(symmetric_diff) / len(union)


def mean_jaccard_diversity(entry: RouteEntry, all_entries: list[RouteEntry]) -> float:
    """Mean Jaccard distance from ``entry`` to all other entries (reference)."""
    others = [candidate for candidate in all_entries if candidate is not entry]
    if not others:
        return 1.0
    return sum(jaccard_distance(entry, other) for other in others) / len(others)


def _customer_set_mask(customer_set: frozenset[int]) -> int:
    mask = 0
    for customer in customer_set:
        mask |= 1 << customer
    return mask


def mean_jaccard_diversities(entries: list[RouteEntry]) -> list[float]:
    """
    Exact mean pairwise Jaccard distance for every entry.

    Same metric as :func:`mean_jaccard_diversity`, computed once per unordered
    pair via customer-id bitmasks (``int.bit_count``) for large pools.
    """
    n = len(entries)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    masks = [_customer_set_mask(entry.customer_set) for entry in entries]
    sums = [0.0] * n
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            union = mi | mj
            if union == 0:
                dist = 0.0
            else:
                dist = (mi ^ mj).bit_count() / union.bit_count()
            sums[i] += dist
            sums[j] += dist

    denom = float(n - 1)
    return [total / denom for total in sums]
