"""Tests for subproblem construction and parallel solve wiring."""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from drispi.core.instance import CVRPInstance
from drispi.pipeline.subproblem import (
    SubclusterSolveError,
    SubclusterWallTimeoutError,
    make_subinstance,
    remap_routes_from_subcluster,
    solve_subclusters_parallel,
    subcluster_wall_timeout,
)


def test_make_subinstance_size_and_mapping(instance_12: CVRPInstance) -> None:
    cluster = [2, 5, 11]
    sub = make_subinstance(instance_12, cluster)
    assert sub.n_customers == 3
    assert sub.customers == [2, 3, 4]
    assert sub.coordinates[2] == instance_12.coordinates[2]
    assert sub.coordinates[3] == instance_12.coordinates[5]
    assert sub.coordinates[4] == instance_12.coordinates[11]
    assert sub.demands[3] == instance_12.demands[5]
    d_parent = instance_12.distance_matrix[2, 5]
    d_sub = sub.distance_matrix[2, 3]
    assert d_sub == pytest.approx(d_parent)


def test_remap_roundtrip(instance_12: CVRPInstance) -> None:
    cluster = [3, 7, 9]
    sub_routes = [[2, 3], [4]]
    parent = remap_routes_from_subcluster(sub_routes, cluster)
    assert parent == [[3, 7], [9]]


def test_subcluster_wall_timeout_formula() -> None:
    assert subcluster_wall_timeout(0.0) == 120.0
    assert subcluster_wall_timeout(100.0) == 220.0
    assert subcluster_wall_timeout(489.5) == pytest.approx(979.0)
    assert subcluster_wall_timeout(100.0, n_rounds=3) == 660.0


def test_solve_subclusters_parallel_raises_on_wall_timeout(instance_12: CVRPInstance) -> None:
    partition = [instance_12.customers[0:6], instance_12.customers[6:12]]
    pending_future: Future = Future()
    with patch("drispi.pipeline.subproblem.ProcessPoolExecutor") as mock_executor_cls:
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor
        mock_executor.submit.return_value = pending_future
        with patch(
            "drispi.pipeline.subproblem.wait",
            return_value=({pending_future}, {pending_future}),
        ):
            with pytest.raises(SubclusterWallTimeoutError, match="wall timeout"):
                solve_subclusters_parallel(
                    instance_12,
                    partition,
                    "pyvrp",
                    time_per_customer=0.05,
                    n_workers=2,
                    seed=1,
                )
    mock_executor.shutdown.assert_called_with(wait=False, cancel_futures=True)


def test_solve_subclusters_parallel_wraps_worker_failure(instance_12: CVRPInstance) -> None:
    partition = [instance_12.customers[0:6], instance_12.customers[6:12]]
    failed: Future = Future()
    failed.set_exception(
        RuntimeError("ails2 subprocess exceeded hard timeout (563.28 s).")
    )
    with patch("drispi.pipeline.subproblem.ProcessPoolExecutor") as mock_executor_cls:
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor
        mock_executor.submit.return_value = failed
        with patch(
            "drispi.pipeline.subproblem.wait",
            return_value=({failed}, set()),
        ):
            with pytest.raises(SubclusterSolveError, match="Subcluster worker failed"):
                solve_subclusters_parallel(
                    instance_12,
                    partition,
                    "ails2",
                    time_per_customer=0.05,
                    n_workers=2,
                    seed=1,
                )
    mock_executor.shutdown.assert_called_with(wait=True, cancel_futures=False)


def test_solve_subclusters_parallel_real_pyvrp_small(instance_12: CVRPInstance) -> None:
    """Optional smoke: real worker + PyVRP on tiny partition (may be slow)."""
    partition = [instance_12.customers[0:6], instance_12.customers[6:12]]
    out, rounds = solve_subclusters_parallel(
        instance_12,
        partition,
        "pyvrp",
        time_per_customer=0.5,
        n_workers=2,
        seed=1,
    )
    assert rounds == 1
    assert len(out) == 2
    seen: set[int] = set()
    for grp in out:
        for route in grp:
            seen.update(route)
    assert seen == set(instance_12.customers)
