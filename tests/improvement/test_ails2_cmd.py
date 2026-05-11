"""AILS-II command-line construction."""

from __future__ import annotations

from pathlib import Path

from drispi.solvers.ails2 import Ails2Solver


def test_build_cmd_routing_phase_no_optional_flags() -> None:
    class T(Ails2Solver):
        def __init__(self) -> None:
            self._binary_path = Path("/fake/AILSII.jar")

    t = T()
    cmd = t._build_cmd(Path("a.vrp"), Path("out.sol"), 60.0, 0)
    assert "-initialSolution" not in cmd
    assert "-initialOmega" not in cmd
    assert cmd[cmd.index("-outpath") + 1] == "out.sol"


def test_build_cmd_with_initial_solution_and_omega() -> None:
    class T(Ails2Solver):
        def __init__(self) -> None:
            self._binary_path = Path("/fake/AILSII.jar")

    t = T()
    cmd = t._build_cmd(
        Path("a.vrp"),
        Path("out.sol"),
        120.0,
        0,
        initial_solution_path=Path("/tmp/init.sol"),
        initial_omega=22.5,
    )
    i = cmd.index("-initialSolution")
    assert cmd[i + 1] == "/tmp/init.sol"
    j = cmd.index("-initialOmega")
    assert cmd[j + 1] == "22.5"
