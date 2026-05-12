"""K-medoids vertex clustering on precomputed dissimilarity."""

from __future__ import annotations

import warnings

import numpy as np

from drispi.clustering._utils import ensure_partition, groups_from_labels
from drispi.core.instance import CVRPInstance


def cluster(
    instance: CVRPInstance,
    k: int,
    dissimilarity_matrix: np.ndarray,
    seed: int = 42,
) -> list[list[int]]:
    """Cluster customers using K-medoids with precomputed distances."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from sklearn_extra.cluster import KMedoids
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "kmedoids requires a working 'scikit-learn-extra' installation. "
            "If you see binary incompatibility, reinstall it for your current NumPy version."
        ) from exc

    customer_ids = list(instance.customers)
    model = KMedoids(
        n_clusters=k,
        metric="precomputed",
        random_state=seed,
    )
    labels = model.fit_predict(dissimilarity_matrix)
    groups = groups_from_labels(customer_ids, labels, k)
    ensure_partition(groups, customer_ids)
    return groups
