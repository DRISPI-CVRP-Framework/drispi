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
    """Mean Jaccard distance from ``entry`` to all other entries."""
    others = [candidate for candidate in all_entries if candidate is not entry]
    if not others:
        return 1.0
    return sum(jaccard_distance(entry, other) for other in others) / len(others)
