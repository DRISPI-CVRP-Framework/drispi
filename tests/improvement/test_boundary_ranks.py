"""Tests for boundary rank computation used by BG-AILS."""

from __future__ import annotations

import numpy as np
import pytest

from drispi.improvement.bg_ails import (
    _apply_threshold,
    compute_boundary_ranks,
    export_boundary_ranks_csv,
)


def _tiny_instance_customers() -> list[int]:
    return [2, 3, 4, 5]


def test_partition_must_cover_customers_exactly() -> None:
    customers = _tiny_instance_customers()
    bad = [[2, 3], [4]]  # missing 5
    d = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="partition must cover"):
        compute_boundary_ranks(
            d, bad, customers, small_cluster_cap=20, small_cluster_alpha=0.5
        )


def test_compute_boundary_ranks_two_clusters() -> None:
    """Singleton vs larger cluster: min cross-cluster dissimilarity differs by row."""
    customers = [2, 3, 4, 5]
    partition = [[2], [3, 4, 5]]
    d = np.array(
        [
            [0.0, 5.0, 1.0, 1.0],
            [5.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    ranks = compute_boundary_ranks(
        d, partition, customers, small_cluster_cap=20, small_cluster_alpha=0.5
    )
    assert ranks.shape == (4,)
    assert np.all((ranks >= 0) & (ranks <= 1))
    # Customer 2 (index 0) has min cross distance 1 to the other cluster; customer 3 has 5.
    assert ranks[0] > ranks[1]


def test_small_cluster_boost_changes_distribution() -> None:
    customers = [2, 3, 4, 5]
    partition = [[2], [3, 4, 5]]
    d = np.ones((4, 4), dtype=np.float64) * 5.0
    np.fill_diagonal(d, 0.0)
    ranks_singleton = compute_boundary_ranks(
        d, partition, customers, small_cluster_cap=20, small_cluster_alpha=0.5
    )
    partition_even = [[2, 3], [4, 5]]
    ranks_even = compute_boundary_ranks(
        d, partition_even, customers, small_cluster_cap=20, small_cluster_alpha=0.5
    )
    assert not np.allclose(ranks_singleton, ranks_even)


def test_apply_threshold_renormalizes() -> None:
    r = np.array([0.2, 0.6, 0.9], dtype=np.float64)
    out = _apply_threshold(r, 0.5)
    assert out.shape == r.shape
    assert np.max(out) <= 1.0 + 1e-9 and np.min(out) >= -1e-9


def test_export_boundary_ranks_csv(tmp_path) -> None:
    p = tmp_path / "r.csv"
    customers = [2, 3, 4]
    ranks = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    export_boundary_ranks_csv(p, customers, ranks)
    text = p.read_text(encoding="utf-8")
    assert "node_id,boundary_rank" in text
    assert "2," in text


@pytest.mark.slow
def test_compute_boundary_ranks_large_n() -> None:
    """Representative scale n=4000, k=400 (matrix ~128 MiB)."""
    n = 4000
    k = 400
    rng = np.random.default_rng(0)
    d = rng.random((n, n)).astype(np.float64)
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    perm = rng.permutation(n)
    customers = [2 + i for i in range(n)]
    partition: list[list[int]] = []
    chunk = n // k
    for i in range(k):
        lo = i * chunk
        hi = (i + 1) * chunk if i < k - 1 else n
        partition.append([customers[int(j)] for j in perm[lo:hi]])
    ranks = compute_boundary_ranks(
        d, partition, customers, small_cluster_cap=20, small_cluster_alpha=0.5
    )
    assert ranks.shape == (n,)
    assert np.all(np.isfinite(ranks))
