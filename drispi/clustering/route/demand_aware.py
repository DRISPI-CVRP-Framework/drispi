"""Demand-aware route clustering stub."""

from __future__ import annotations

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.types import Cluster


@register_clustering("route:demand_aware")
class RouteDemandAwareClustering(BaseClustering):
    """Clustering that balances spatial and demand considerations."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: combine demand and distance features for grouping
        raise NotImplementedError
