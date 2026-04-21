"""K-means route clustering from route centroids."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from drispi.clustering._utils import ensure_partition
from drispi.core.instance import CVRPInstance
from drispi.core.types import RoutePool


def _route_centroid(instance: CVRPInstance, route: list[int]) -> tuple[float, float]:
    coords = np.asarray([instance.coordinates[cid] for cid in route], dtype=float)
    return float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1]))


def _cluster_centroids(
    instance: CVRPInstance,
    groups: list[list[int]],
) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for cluster_id, customers in enumerate(groups):
        if not customers:
            centroids[cluster_id] = np.asarray(instance.depot, dtype=float)
            continue
        coords = np.asarray([instance.coordinates[cid] for cid in customers], dtype=float)
        centroids[cluster_id] = np.mean(coords, axis=0)
    return centroids


def _assign_customers(
    instance: CVRPInstance,
    routes: RoutePool,
    route_labels: np.ndarray,
    k: int,
) -> list[list[int]]:
    customer_to_clusters: dict[int, set[int]] = defaultdict(set)
    for route_idx, route in enumerate(routes):
        label = int(route_labels[route_idx])
        for customer in route:
            customer_to_clusters[customer].add(label)

    groups: list[list[int]] = [[] for _ in range(k)]
    cluster_centroids = _cluster_centroids(
        instance=instance,
        groups=[
            [
                customer
                for customer, clusters in customer_to_clusters.items()
                if cluster_id in clusters
            ]
            for cluster_id in range(k)
        ],
    )

    for customer in instance.customers:
        cluster_candidates = customer_to_clusters.get(customer, set())
        if not cluster_candidates:
            raise RuntimeError(f"Customer {customer} is not present in the provided route pool.")
        if len(cluster_candidates) == 1:
            chosen = next(iter(cluster_candidates))
        else:
            point = np.asarray(instance.coordinates[customer], dtype=float)
            chosen = min(
                cluster_candidates,
                key=lambda cid: (float(np.linalg.norm(point - cluster_centroids[cid])), cid),
            )
        groups[chosen].append(customer)
    return groups


def cluster(
    instance: CVRPInstance,
    routes: RoutePool,
    k: int,
    seed: int = 42,
) -> list[list[int]]:
    """Cluster routes via centroid features and map to customer groups."""
    if not routes:
        raise ValueError("Route clustering requires at least one route.")

    route_features = np.asarray([_route_centroid(instance, route) for route in routes], dtype=float)
    model = KMeans(n_clusters=k, random_state=seed)
    route_labels = model.fit_predict(route_features)
    groups = _assign_customers(instance, routes, route_labels, k)
    ensure_partition(groups, instance.customers)
    return groups
