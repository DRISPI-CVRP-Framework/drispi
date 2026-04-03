"""Spectral vertex clustering stub."""

from __future__ import annotations

from typing import Any

from drispi.clustering.base import BaseClustering, register_clustering
from drispi.core.instance import CVRPInstance
from drispi.core.types import Cluster


@register_clustering("vertex:spectral")
class VertexSpectralClustering(BaseClustering):
    """Spectral clustering on similarity graph of customers."""

    def cluster(self, n_clusters: int) -> list[Cluster]:
        # TODO: affinity matrix, spectral embedding, k-means in embedded space
        raise NotImplementedError
