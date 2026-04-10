"""Tests for solver package exports and construction."""

from __future__ import annotations

import pytest

import drispi.solvers as solvers
from drispi.core.instance import CVRPInstance
from drispi.solvers import (
    Ails2Solver,
    BaseSolver,
    Filo2Solver,
    FiloSolver,
    PyVRPSolver,
)


def test_solvers_exported_in_all() -> None:
    expected = {
        "Ails2Solver",
        "BaseSolver",
        "Filo2Solver",
        "FiloSolver",
        "PyVRPSolver",
    }
    assert expected.issubset(set(solvers.__all__))


def test_pyvrp_solver_constructible() -> None:
    """``PyVRPSolver`` should construct when ``pyvrp`` is installed."""
    PyVRPSolver()


def test_pyvrp_solves_small_instance(small_instance: CVRPInstance) -> None:
    pool = PyVRPSolver().solve(small_instance, time_limit=3.0, seed=1)
    assert len(pool) > 0
    assert all(len(r.customers) > 0 for r in pool)


def test_binary_solvers_resolve_when_present() -> None:
    """Construct wrappers; raises only if binaries/JAR are missing."""
    FiloSolver()
    Filo2Solver()
    Ails2Solver()


def test_base_solver_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        BaseSolver()  # type: ignore[call-arg]
