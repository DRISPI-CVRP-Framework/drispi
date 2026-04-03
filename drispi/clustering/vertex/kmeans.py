"""K-means vertex clustering stub."""

from __future__ import annotations

from typing import Any

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.instance import CVRPInstance
from drispi.core.types import Cluster


@register_clustering("vertex:kmeans")
class VertexKMeansClustering(BaseClustering):
    """K-means clustering on customer coordinates or features."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: fit k-means, map labels to customer ID lists
        raise NotImplementedError
