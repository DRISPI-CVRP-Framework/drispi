"""Spectral vertex clustering from precomputed dissimilarity."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering

from drispi.clustering._utils import ensure_partition, groups_from_labels
from drispi.core.instance import CVRPInstance

def cluster(
    instance: CVRPInstance,
    k: int,
    dissimilarity_matrix: np.ndarray,
    seed: int = 42,
) -> list[list[int]]:
    """Cluster customers with spectral clustering on an RBF affinity."""
    customer_ids = list(instance.customers)
    positive = dissimilarity_matrix[dissimilarity_matrix > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    if sigma <= 0.0:
        sigma = 1.0
    affinity = np.exp(-(dissimilarity_matrix**2) / (2.0 * sigma**2))

    model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=seed,
    )
    labels = model.fit_predict(affinity)
    groups = groups_from_labels(customer_ids, labels, k)
    ensure_partition(groups, customer_ids)
    return groups
