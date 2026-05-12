"""Regression: solution total cost matches VRPLIB ``EUC_2D`` (integer-rounded legs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from drispi.core.instance import CVRPInstance
from drispi.utils.io import read_sol

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOL_PATH = _REPO_ROOT / "artifacts" / "xl_n1281_long" / "X-n1281-k29.sol"

# Official / checker total for this route set under TSPLIB EUC_2D (see ``CVRPInstance.distance_matrix``).
_EXPECTED_TSPLIB_TOTAL = 31220


def _load_instance_matching_sol(routes: list[list[int]]) -> CVRPInstance:
    covered = {c for r in routes for c in r}
    for name in ("XL-n1281-k29", "X-n1281-k29"):
        try:
            inst = CVRPInstance.from_vrplib(name)
        except FileNotFoundError:
            continue
        if set(inst.customers) == covered:
            return inst
    pytest.skip("No VRPLIB instance on INSTANCE_SEARCH_DIRS matches sol customer set")


@pytest.mark.skipif(not _SOL_PATH.is_file(), reason=f"missing solution file {_SOL_PATH}")
def test_xl_n1281_artifact_sol_matches_tsplib_total_cost() -> None:
    """
    ``artifacts/xl_n1281_long/X-n1281-k29.sol`` must sum to the TSPLIB-rounded cost.

    A continuous-Euclidean matrix previously inflated totals (e.g. ``Cost: 31292``);
    ``distance_matrix`` now uses per-edge ``round`` like VRPLIB ``EUC_2D``.
    """
    routes, _file_cost = read_sol(_SOL_PATH)
    instance = _load_instance_matching_sol(routes)
    recomputed = float(sum(instance.route_cost(r) for r in routes))
    assert recomputed == _EXPECTED_TSPLIB_TOTAL
