"""Agglomerative route clustering from route centroids."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from drispi.clustering._utils import ensure_partition
from drispi.clustering.route.kmeans import _assign_customers, _route_centroid
from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool

VALID_LINKAGES = {"average", "complete", "single"}


def cluster(
    instance: CVRPInstance,
    routes: RoutePool,
    k: int,
    seed: int = 42,
    linkage: str = "average",
) -> list[list[int]]:
    """Cluster routes and propagate labels to customers."""
    _ = seed
    if not routes:
        raise ValueError("Route clustering requires at least one route.")
    if linkage not in VALID_LINKAGES:
        raise ValueError(f"Unsupported linkage '{linkage}'. Valid: {sorted(VALID_LINKAGES)}")

    route_features = np.asarray([_route_centroid(instance, route) for route in routes], dtype=float)
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="euclidean",
        linkage=linkage,
    )
    route_labels = model.fit_predict(route_features)
    groups = _assign_customers(instance, routes, route_labels, k)
    ensure_partition(groups, instance.customers)
    return groups
