"""Tests for clustering registry and imports."""

from __future__ import annotations

import drispi.clustering.vertex.kmeans  # noqa: F401 — registration side effect
from drispi.clustering.base import CLUSTERING_REGISTRY


def test_vertex_kmeans_registered() -> None:
    """Vertex k-means is registered under a namespaced key."""
    assert "vertex:kmeans" in CLUSTERING_REGISTRY
