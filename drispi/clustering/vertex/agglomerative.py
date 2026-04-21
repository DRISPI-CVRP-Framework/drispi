"""Agglomerative vertex clustering on precomputed dissimilarity."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from drispi.clustering._utils import ensure_partition, groups_from_labels
from drispi.core.instance import CVRPInstance

VALID_LINKAGES = {"average", "complete", "single"}


def cluster(
    instance: CVRPInstance,
    k: int,
    dissimilarity_matrix: np.ndarray,
    seed: int = 42,
    linkage: str = "average",
) -> list[list[int]]:
    """Cluster customers from a precomputed dissimilarity matrix."""
    _ = seed
    if linkage not in VALID_LINKAGES:
        raise ValueError(f"Unsupported linkage '{linkage}'. Valid: {sorted(VALID_LINKAGES)}")

    customer_ids = list(instance.customers)
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage=linkage,
    )
    labels = model.fit_predict(dissimilarity_matrix)
    groups = groups_from_labels(customer_ids, labels, k)
    ensure_partition(groups, customer_ids)
    return groups
