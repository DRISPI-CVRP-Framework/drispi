"""Angular / directional vertex clustering stub."""

from __future__ import annotations

from typing import Any

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.instance import CVRPInstance
from drispi.core.types import Cluster


@register_clustering("vertex:angular")
class VertexAngularClustering(BaseClustering):
    """Cluster customers by polar angle or similar directional feature."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: compute angles from depot, bin or cluster in angle space
        raise NotImplementedError
