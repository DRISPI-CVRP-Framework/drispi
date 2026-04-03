"""Similarity-based route clustering stub."""

from __future__ import annotations

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.types import Cluster


@register_clustering("route:similarity")
class RouteSimilarityClustering(BaseClustering):
    """Cluster customers using route or tour similarity features."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: derive similarity matrix from provisional routes or features
        raise NotImplementedError
