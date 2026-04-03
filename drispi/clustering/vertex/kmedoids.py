"""K-medoids vertex clustering stub."""

from __future__ import annotations

from typing import Any

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.instance import CVRPInstance
from drispi.core.types import Cluster


@register_clustering("vertex:kmedoids")
class VertexKMedoidsClustering(BaseClustering):
    """K-medoids (e.g. PAM) on distance matrix between customers."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: build distances, run k-medoids, return clusters
        raise NotImplementedError
