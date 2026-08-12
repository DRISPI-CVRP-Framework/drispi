"""Tests for clustering dissimilarity matrix."""

from __future__ import annotations

import numpy as np

from drispi.clustering.dissimilarity import compute_dissimilarity_matrix
from drispi.core.instance import CVRPInstance


def _build_instance_10() -> CVRPInstance:
    n_customers = 10
    customer_ids = list(range(2, 2 + n_customers))
    coordinates: dict[int, tuple[float, float]] = {1: (0.0, 0.0)}
    demands: dict[int, int] = {1: 0}
    for idx, cid in enumerate(customer_ids):
        row = idx // 5
        col = idx % 5
        coordinates[cid] = (float(col + 1), float(row + 1))
        demands[cid] = 1 + idx
    return CVRPInstance(
        name="dissimilarity_test_10",
        n_customers=n_customers,
        capacity=50,
        depot=(0.0, 0.0),
        customers=customer_ids,
        coordinates=coordinates,
        demands=demands,
    )


def test_matrix_shape_diagonal_and_symmetry() -> None:
    instance = _build_instance_10()
    matrix = compute_dissimilarity_matrix(instance)
    assert matrix.shape == (instance.n_customers, instance.n_customers)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(matrix, matrix.T)


def test_demand_weight_changes_matrix() -> None:
    instance = _build_instance_10()
    base = compute_dissimilarity_matrix(instance, lambda_demand=0.0)
    demand_scaled = compute_dissimilarity_matrix(instance, lambda_demand=1.0)
    assert not np.allclose(base, demand_scaled)


def test_angular_offset_keeps_matrix_with_circular_difference() -> None:
    instance = _build_instance_10()
    no_offset = compute_dissimilarity_matrix(instance, angular_offset=0.0)
    offset = compute_dissimilarity_matrix(instance, angular_offset=1.0)
    # With circular angular differences, a global rotation offset leaves pairwise
    # angular distances unchanged.
    assert np.allclose(no_offset, offset)


def test_off_diagonal_entries_positive() -> None:
    instance = _build_instance_10()
    matrix = compute_dissimilarity_matrix(instance)
    off_diagonal = matrix[~np.eye(instance.n_customers, dtype=bool)]
    assert np.all(off_diagonal > 0.0)


def test_bit_identical_recompute_across_pickle_round_trip() -> None:
    """Option 3' contract: the async BG worker rebuilds the matrix from a
    pickled instance copy and must get a bit-identical float64 result."""
    import pickle

    instance = _build_instance_10()
    lam, offset = 0.4, 1.2345
    main_thread = compute_dissimilarity_matrix(instance, lam, offset)
    worker_copy = pickle.loads(pickle.dumps(instance, protocol=pickle.HIGHEST_PROTOCOL))
    worker_side = compute_dissimilarity_matrix(worker_copy, lam, offset)
    assert main_thread.dtype == np.float64
    assert worker_side.dtype == np.float64
    assert np.array_equal(main_thread, worker_side)  # bitwise, not allclose
