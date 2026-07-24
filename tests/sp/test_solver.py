"""Tests for ``run_sp_sc`` orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from drispi.core.instance import CVRPInstance
from drispi.route_pool.manager import RoutePoolManager
from drispi.route_pool.pool import RoutePool
from drispi.sp.solver import run_sp_sc

_SP_SC_KW = {
    "time_limit": 300.0,
    "mip_gap": 0.0005,
    "min_coverage": 1,
    "warmup_iterations": 10,
    "sp_interval": 3,
}


def _manager(max_pool_size: int = 100) -> RoutePoolManager:
    return RoutePoolManager(
        max_pool_size=max_pool_size,
        min_coverage=1,
        diversity_weight=1.0,
        warmup_iterations=10,
        sp_interval=3,
    )


def _pool_snapshot(pool: RoutePool) -> dict[frozenset[int], float]:
    return {frozenset(entry.route): entry.cost for entry in pool.routes()}


def test_run_sp_sc_skips_before_warmup(sp_instance: CVRPInstance) -> None:
    pool = RoutePool()
    pool.add([2, 3, 4, 5, 6, 7], sp_instance.route_cost([2, 3, 4, 5, 6, 7]))
    manager = _manager()

    out = run_sp_sc(
        pool,
        sp_instance,
        manager,
        iteration=5,
        best_solution=[[2, 3], [4, 5], [6, 7]],
        **_SP_SC_KW,
    )
    assert out == (None, False)


def test_run_sp_sc_calls_update_scores_after_solve(sp_instance: CVRPInstance) -> None:
    pool = RoutePool()
    pool.add([2, 3, 4, 5, 6, 7], sp_instance.route_cost([2, 3, 4, 5, 6, 7]))
    manager = MagicMock(spec=RoutePoolManager)

    lp = {frozenset({2, 3}): 0.5, frozenset({4, 5, 6, 7}): 0.5}
    with patch(
        "drispi.sp.solver.build_and_solve",
        return_value=(lp, [[2, 3], [4, 5, 6, 7]], False, 1, None),
    ):
        run_sp_sc(
            pool,
            sp_instance,
            manager,
            iteration=10,
            best_solution=[[2, 3], [4, 5], [6, 7]],
            **_SP_SC_KW,
        )

    manager.update_scores_after_solve.assert_called_once_with(pool, lp)


def test_run_sp_sc_timeout_no_incumbent_uses_best_and_keeps_pool(
    sp_instance: CVRPInstance,
) -> None:
    pool = RoutePool()
    pool.add([2, 3], sp_instance.route_cost([2, 3]))
    pool.add([4, 5], sp_instance.route_cost([4, 5]))
    pool.add([6, 7], sp_instance.route_cost([6, 7]))
    before = _pool_snapshot(pool)

    manager = MagicMock(spec=RoutePoolManager)
    best = [[2, 3], [4, 5], [6, 7]]

    lp = {frozenset({2, 3}): 0.33, frozenset({4, 5}): 0.33, frozenset({6, 7}): 0.34}
    with (
        patch.object(RoutePoolManager, "reset_pool_to_best") as reset,
        patch("drispi.sp.solver.build_and_solve", return_value=(lp, [], True, 0, None)),
    ):
        sol, use_sp = run_sp_sc(
            pool,
            sp_instance,
            manager,
            iteration=10,
            best_solution=best,
            **_SP_SC_KW,
        )

    reset.assert_not_called()
    assert sol == [[2, 3], [4, 5], [6, 7]]
    assert use_sp is True
    assert _pool_snapshot(pool) == before


def test_run_sp_sc_sp_path_skips_remove_duplicates(sp_instance: CVRPInstance) -> None:
    pool = RoutePool()
    pool.add([2, 3], sp_instance.route_cost([2, 3]))
    pool.add([4, 5], sp_instance.route_cost([4, 5]))
    pool.add([6, 7], sp_instance.route_cost([6, 7]))
    manager = MagicMock(spec=RoutePoolManager)

    lp = {frozenset({2, 3}): 1.0, frozenset({4, 5}): 1.0, frozenset({6, 7}): 1.0}
    raw = [[2, 3], [4, 5], [6, 7]]
    with (
        patch("drispi.sp.solver.build_and_solve", return_value=(lp, raw, False, 1, None)),
        patch("drispi.sp.solver.remove_duplicates") as rd,
    ):
        sol, use_sp = run_sp_sc(
            pool,
            sp_instance,
            manager,
            iteration=10,
            best_solution=raw,
            **_SP_SC_KW,
        )

    assert use_sp is True
    rd.assert_not_called()
    assert sol == raw


def test_run_sp_sc_end_to_end_partition(
    requires_gurobi: None,
    sp_instance: CVRPInstance,
) -> None:
    pool = RoutePool()
    pool.add([2, 3], sp_instance.route_cost([2, 3]))
    pool.add([4, 5], sp_instance.route_cost([4, 5]))
    pool.add([6, 7], sp_instance.route_cost([6, 7]))

    manager = _manager()
    best = [[2, 3], [4, 5], [6, 7]]

    sol, use_sp = run_sp_sc(
        pool,
        sp_instance,
        manager,
        iteration=10,
        best_solution=best,
        **_SP_SC_KW,
        min_coverage=1,
    )

    assert sol is not None
    counts: dict[int, int] = {}
    for route in sol:
        for c in route:
            counts[c] = counts.get(c, 0) + 1
    assert all(counts[c] == 1 for c in sp_instance.customers)
    assert use_sp is True


def test_run_sp_sc_end_to_end_set_covering(
    requires_gurobi: None,
    sp_instance: CVRPInstance,
) -> None:
    pool = RoutePool()
    pool.add([2, 3], sp_instance.route_cost([2, 3]))
    pool.add([4, 5], sp_instance.route_cost([4, 5]))
    pool.add([6, 7], sp_instance.route_cost([6, 7]))

    manager = _manager()
    best = [[2, 3], [4, 5], [6, 7]]

    sol, use_sp = run_sp_sc(
        pool,
        sp_instance,
        manager,
        iteration=10,
        best_solution=best,
        **_SP_SC_KW,
        min_coverage=2,
    )

    assert sol is not None
    counts: dict[int, int] = {}
    for route in sol:
        for c in route:
            counts[c] = counts.get(c, 0) + 1
    assert all(counts[c] == 1 for c in sp_instance.customers)
    assert use_sp is False
