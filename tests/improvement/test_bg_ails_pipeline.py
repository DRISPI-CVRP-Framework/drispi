"""BG-AILS / standard improvement orchestration (mocked AILS-II)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route, Solution
from drispi.improvement.bg_ails import BgAilsImprovement, run_bg_ails, run_standard_improvement
from drispi.solvers.ails2 import Ails2Solver

_BG_KW = {
    "boundary_threshold": 0.5,
    "small_cluster_cap": 20,
    "small_cluster_alpha": 0.5,
    "seed": 1,
}


def _inst() -> CVRPInstance:
    return CVRPInstance(
        name="toy",
        n_customers=4,
        capacity=30,
        depot=(0.0, 0.0),
        customers=[2, 3, 4, 5],
        coordinates={
            1: (0.0, 0.0),
            2: (0.0, 1.0),
            3: (1.0, 0.0),
            4: (0.0, 2.0),
            5: (2.0, 0.0),
        },
        demands={1: 0, 2: 5, 3: 5, 4: 5, 5: 5},
    )


def test_run_bg_ails_single_cluster_delegates_to_standard(monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _inst()
    routes = [Route(customers=[2, 3], cost=1.0), Route(customers=[4, 5], cost=1.0)]
    d = np.zeros((4, 4), dtype=np.float64)
    called: dict[str, object] = {}

    def fake_std(i, s, **kw):
        called["std"] = True
        assert i is inst
        assert s == routes
        return s

    monkeypatch.setattr(
        "drispi.improvement.bg_ails.run_standard_improvement",
        fake_std,
    )
    _pert, perturbed, out = run_bg_ails(
        inst,
        routes,
        d,
        [[2, 3, 4, 5]],
        initial_omega=12.0,
        time_limit=30.0,
        **_BG_KW,
    )
    assert called.get("std") is True
    assert out == routes
    assert perturbed == []


def test_run_standard_improvement_invokes_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _inst()
    routes = [Route(customers=[2, 3, 4, 5], cost=1.0)]
    mock_solver = MagicMock(spec=Ails2Solver)
    mock_solver.run_improvement.return_value = [
        Route(customers=[2, 4, 3, 5], cost=2.0),
    ]

    run_standard_improvement(inst, routes, time_limit=5.0, solver=mock_solver, seed=1)
    mock_solver.run_improvement.assert_called_once()
    args, kwargs = mock_solver.run_improvement.call_args
    assert args[0] is inst
    assert args[1] == 5.0
    assert isinstance(args[2], Path)
    assert kwargs.get("initial_omega") is None


def test_run_bg_ails_passes_initial_omega(monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _inst()
    routes = [Route(customers=[2, 3], cost=1.0), Route(customers=[4, 5], cost=1.0)]
    d = np.ones((4, 4), dtype=np.float64) * 3.0
    np.fill_diagonal(d, 0.0)
    partition = [[2, 3], [4, 5]]
    mock_solver = MagicMock(spec=Ails2Solver)
    mock_solver.run_improvement.return_value = routes

    run_bg_ails(
        inst,
        routes,
        d,
        partition,
        initial_omega=18.0,
        time_limit=10.0,
        solver=mock_solver,
        **_BG_KW,
    )
    _args, kwargs = mock_solver.run_improvement.call_args
    assert kwargs["initial_omega"] == 18.0


def test_run_bg_ails_preserves_customer_multiset(monkeypatch: pytest.MonkeyPatch) -> None:
    inst = _inst()
    routes = [Route(customers=[2, 3], cost=1.0), Route(customers=[4, 5], cost=1.0)]
    d = np.ones((4, 4), dtype=np.float64) * 2.0
    np.fill_diagonal(d, 0.0)
    partition = [[2, 3], [4, 5]]

    def _pert_identity(*_a, **_k):
        return routes, [], []

    monkeypatch.setattr("drispi.improvement.bg_ails.perturb_routes", _pert_identity)
    mock_solver = MagicMock(spec=Ails2Solver)
    mock_solver.run_improvement.return_value = routes
    _pert, _perturbed, out = run_bg_ails(
        inst,
        routes,
        d,
        partition,
        initial_omega=5.0,
        time_limit=5.0,
        solver=mock_solver,
        **_BG_KW,
    )
    flat = sorted(c for r in out for c in r.customers)
    assert flat == sorted(inst.customers)


def test_bg_ails_improvement_delegates() -> None:
    inst = _inst()
    d = np.ones((4, 4), dtype=np.float64) * 2.0
    np.fill_diagonal(d, 0.0)
    partition = [[2, 3], [4, 5]]
    mock = MagicMock(spec=Ails2Solver)
    mock.run_improvement.return_value = [
        Route([2, 3], 0.0),
        Route([4, 5], 0.0),
    ]
    op = BgAilsImprovement(
        d,
        partition,
        initial_omega=10.0,
        boundary_threshold=0.2,
        small_cluster_cap=20,
        small_cluster_alpha=0.5,
        solver=mock,
        seed=1,
    )
    sol = Solution(
        routes=[Route([2, 3], 0.0), Route([4, 5], 0.0)],
        total_cost=0.0,
        instance_name="toy",
    )
    out = op.improve(sol, inst, time_limit=7.0)
    assert isinstance(out, Solution)
    mock.run_improvement.assert_called_once()
