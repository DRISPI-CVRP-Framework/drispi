"""Customer dissimilarity matrix for clustering."""

from __future__ import annotations

import math

import numpy as np

from drispi.core.instance import CVRPInstance


def compute_dissimilarity_matrix(
    instance: CVRPInstance,
    lambda_demand: float = 0.0,
    angular_offset: float = 0.0,
) -> np.ndarray:
    """Compute customer-level spatial-angular(-demand) dissimilarity."""
    customer_ids = list(instance.customers)
    n_customers = len(customer_ids)
    if n_customers == 0:
        return np.zeros((0, 0), dtype=np.float64)

    depot_x, depot_y = instance.depot
    coords = np.asarray([instance.coordinates[cid] for cid in customer_ids], dtype=np.float64)
    demands = np.asarray([instance.demands[cid] for cid in customer_ids], dtype=np.float64)

    lambda_theta = float(np.sum(coords[:, 0] + coords[:, 1]) / (2.0 * n_customers))

    raw_angles = np.arctan2(coords[:, 1] - depot_y, coords[:, 0] - depot_x)
    angles = np.mod(raw_angles + angular_offset, 2.0 * math.pi)

    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dtheta_raw = np.abs(angles[:, None] - angles[None, :])
    dtheta = np.minimum(dtheta_raw, 2.0 * math.pi - dtheta_raw)

    base = np.sqrt(dx * dx + dy * dy + lambda_theta * dtheta * dtheta)

    if lambda_demand > 0.0:
        demand_delta = np.abs(demands[:, None] - demands[None, :])
        demand_factor = 1.0 + float(lambda_demand) * demand_delta / float(instance.capacity)
        matrix = base * demand_factor
    else:
        matrix = base

    matrix = matrix.astype(np.float64, copy=False)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return matrix
