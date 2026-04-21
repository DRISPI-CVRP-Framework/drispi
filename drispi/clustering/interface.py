"""Unified clustering interface."""

from __future__ import annotations

import importlib

from drispi.clustering._utils import ensure_partition
from drispi.clustering.dissimilarity import compute_dissimilarity_matrix
from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool

VERTEX_METHODS = {
    "kmeans",
    "agglomerative_avg",
    "agglomerative_complete",
    "agglomerative_single",
    "kmedoids",
    "fcm",
    "spectral",
}
ROUTE_METHODS = {
    "kmeans",
    "agglomerative_avg",
    "agglomerative_complete",
    "agglomerative_single",
}

LINKAGE_MAP = {
    "avg": "average",
    "average": "average",
    "complete": "complete",
    "single": "single",
}


def cluster_instance(
    instance: CVRPInstance,
    paradigm: str,
    method: str,
    k: int,
    routes: RoutePool | None = None,
    lambda_demand: float = 0.0,
    angular_offset: float = 0.0,
    seed: int = 42,
) -> list[list[int]]:
    """Cluster instance customers using vertex- or route-based methods."""
    if paradigm not in {"vertex", "route"}:
        raise ValueError(
            f"Unknown paradigm '{paradigm}'. Valid values are 'vertex' and 'route'."
        )

    if paradigm == "vertex" and method not in VERTEX_METHODS:
        raise ValueError(f"Unknown vertex method '{method}'. Valid values: {sorted(VERTEX_METHODS)}")
    if paradigm == "route" and method not in ROUTE_METHODS:
        raise ValueError(f"Unknown route method '{method}'. Valid values: {sorted(ROUTE_METHODS)}")
    if paradigm == "route" and (routes is None or len(routes) == 0):
        raise ValueError("Route paradigm requires a non-empty routes argument.")

    dissimilarity_matrix = compute_dissimilarity_matrix(
        instance=instance,
        lambda_demand=lambda_demand,
        angular_offset=angular_offset,
    )

    if paradigm == "vertex":
        if method == "kmeans":
            module = importlib.import_module("drispi.clustering.vertex.kmeans")
            groups = module.cluster(instance, k, dissimilarity_matrix, seed=seed)
        elif method == "kmedoids":
            module = importlib.import_module("drispi.clustering.vertex.kmedoids")
            groups = module.cluster(instance, k, dissimilarity_matrix, seed=seed)
        elif method == "fcm":
            module = importlib.import_module("drispi.clustering.vertex.fcm")
            groups = module.cluster(instance, k, dissimilarity_matrix, seed=seed)
        elif method == "spectral":
            module = importlib.import_module("drispi.clustering.vertex.spectral")
            groups = module.cluster(instance, k, dissimilarity_matrix, seed=seed)
        else:
            linkage_key = method.removeprefix("agglomerative_")
            linkage = LINKAGE_MAP[linkage_key]
            module = importlib.import_module("drispi.clustering.vertex.agglomerative")
            groups = module.cluster(
                instance,
                k,
                dissimilarity_matrix,
                seed=seed,
                linkage=linkage,
            )
    else:
        assert routes is not None
        if method == "kmeans":
            module = importlib.import_module("drispi.clustering.route.kmeans")
            groups = module.cluster(instance, routes, k, seed=seed)
        else:
            linkage_key = method.removeprefix("agglomerative_")
            linkage = LINKAGE_MAP[linkage_key]
            module = importlib.import_module("drispi.clustering.route.agglomerative")
            groups = module.cluster(
                instance,
                routes,
                k,
                seed=seed,
                linkage=linkage,
            )

    if len(groups) != k:
        raise RuntimeError(f"Expected {k} groups, got {len(groups)}.")
    ensure_partition(groups, instance.customers)
    return groups
