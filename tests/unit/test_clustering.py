"""Tests for clustering package exports."""

from __future__ import annotations

import drispi.clustering as clustering


def test_cluster_instance_exported() -> None:
    """Unified entrypoint is exported from package root."""
    assert callable(clustering.cluster_instance)
