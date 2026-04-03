"""DBSCAN vertex clustering stub."""

from __future__ import annotations

from typing import Any

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.instance import CVRPInstance
from drispi.core.types import Cluster


@register_clustering("vertex:dbscan")
class VertexDBSCANClustering(BaseClustering):
    """Density-based clustering for customers."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: DBSCAN may ignore n_clusters; document semantics or cap components
        raise NotImplementedError
