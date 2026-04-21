"""Shared helpers for clustering outputs and validation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def groups_from_labels(
    customer_ids: list[int],
    labels: np.ndarray,
    k: int,
) -> list[list[int]]:
    """Build ``k`` customer groups from integer cluster labels."""
    if labels.shape[0] != len(customer_ids):
        raise RuntimeError("Label vector length must match number of customers.")

    groups: list[list[int]] = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        label_int = int(label)
        if not 0 <= label_int < k:
            raise RuntimeError(f"Invalid cluster label {label_int} for k={k}.")
        groups[label_int].append(customer_ids[idx])
    return groups


def ensure_partition(groups: list[list[int]], customer_ids: Iterable[int]) -> None:
    """Validate groups are a disjoint exact partition of ``customer_ids``."""
    expected = set(customer_ids)
    flattened = [customer for group in groups for customer in group]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Cluster output has duplicate customer assignments.")
    if set(flattened) != expected:
        raise RuntimeError("Cluster output does not cover all customers exactly once.")
