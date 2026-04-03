"""Overlap-graph route clustering stub."""

from __future__ import annotations

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.types import Cluster


@register_clustering("route:overlap_graph")
class RouteOverlapGraphClustering(BaseClustering):
    """Cluster using overlap graph between candidate routes or visits."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: build overlap graph, partition (e.g. community detection)
        raise NotImplementedError
