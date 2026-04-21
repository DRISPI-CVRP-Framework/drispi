"""Fuzzy C-means vertex clustering."""

from __future__ import annotations

import numpy as np
from skfuzzy.cluster import cmeans

from drispi.clustering._utils import ensure_partition, groups_from_labels
from drispi.core.instance import CVRPInstance


def cluster(
    instance: CVRPInstance,
    k: int,
    dissimilarity_matrix: np.ndarray,
    seed: int = 42,
) -> list[list[int]]:
    """Cluster customers using fuzzy C-means on coordinates."""
    _ = dissimilarity_matrix
    customer_ids = list(instance.customers)
    coords = np.asarray([instance.coordinates[cid] for cid in customer_ids], dtype=float)
    data = coords.T

    np.random.seed(seed)
    _, membership, _, _, _, _, _ = cmeans(
        data=data,
        c=k,
        m=2.0,
        error=0.005,
        maxiter=1000,
    )
    labels = np.argmax(membership, axis=0)
    groups = groups_from_labels(customer_ids, labels, k)
    ensure_partition(groups, customer_ids)
    return groups
