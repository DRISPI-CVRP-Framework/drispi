"""Fixed-instance clustering smoke test for reproducible outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from drispi.clustering.interface import cluster_instance
from drispi.core.instance import CVRPInstance

INSTANCE_NAME = "XL-n1654-k11"
K = 5
SEED = 42
LAMBDA_DEMAND = 0.2
ANGULAR_OFFSET = 0.7


def _has_working_kmedoids() -> bool:
    try:
        from sklearn_extra.cluster import KMedoids  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.parametrize(
    "method",
    [
        "kmeans",
        "agglomerative_avg",
        "agglomerative_complete",
        "agglomerative_single",
        "kmedoids",
        "fcm",
        "spectral",
    ],
)
def test_fixed_instance_vertex_clustering_outputs(method: str, tmp_path: Path) -> None:
    """Run all vertex methods on a fixed instance and persist group outputs."""
    if method == "kmedoids" and not _has_working_kmedoids():
        pytest.skip("kmedoids skipped: sklearn_extra binary is incompatible in this environment")

    instance = CVRPInstance.from_vrplib(INSTANCE_NAME)
    groups = cluster_instance(
        instance=instance,
        paradigm="vertex",
        method=method,
        k=K,
        lambda_demand=LAMBDA_DEMAND,
        angular_offset=ANGULAR_OFFSET,
        seed=SEED,
    )

    out_file = tmp_path / f"{INSTANCE_NAME}_{method}_clusters.txt"
    lines = [f"cluster_{idx}: {sorted(group)}" for idx, group in enumerate(groups)]
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    flattened = [customer for group in groups for customer in group]
    assert len(groups) == K
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(instance.customers)
