"""K-means in route-feature space stub."""

from __future__ import annotations

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.types import Cluster


@register_clustering("route:kmeans")
class RouteKMeansClustering(BaseClustering):
    """K-means on engineered features derived from routes or geometry."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: feature matrix per customer, k-means
        raise NotImplementedError
