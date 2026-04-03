"""Hierarchical route clustering stub."""

from __future__ import annotations

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.types import Cluster


@register_clustering("route:hierarchical")
class RouteHierarchicalClustering(BaseClustering):
    """Agglomerative or divisive hierarchical clustering on route-related data."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: linkage clustering, cut at n_clusters
        raise NotImplementedError
