"""AILS-II (Java JAR) solver integration."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from drispi.solvers._subprocess_base import _SubprocessSolver, _temp_vrp_path

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance

from drispi.core.solution import Route
from drispi.utils.io import write_vrp


class Ails2Solver(_SubprocessSolver):
    """Subprocess wrapper around the AILS-II ``AILSII.jar``."""

    _ENV_VAR = "AILS2_JAR"
    _DEFAULT_REL_PATH = "ext/ails2/build/AILSII.jar"
    name = "ails2"

    # Legacy default when ``cores:`` is absent: treat SP as one logical processor
    # for ActiveProcessorCount (matches the usual 7+1 production reservation).
    DEFAULT_ACTIVE_PROCESSOR_COUNT = 1
    DEFAULT_XMX = "4g"

    def __init__(
        self,
        binary_path: str | Path | None = None,
        *,
        active_processor_count: int | None = None,
        xmx: str | None = None,
    ) -> None:
        super().__init__(binary_path)
        self._active_processor_count = (
            self.DEFAULT_ACTIVE_PROCESSOR_COUNT
            if active_processor_count is None
            else int(active_processor_count)
        )
        self._xmx = self.DEFAULT_XMX if xmx is None else str(xmx)

    @property
    def jvm_flags(self) -> list[str]:
        """Identical flags for sync and async arms (reproducibility / stability)."""
        return [
            "-XX:+UseSerialGC",
            f"-XX:ActiveProcessorCount={self._active_processor_count}",
            f"-Xmx{self._xmx}",
        ]

    def _build_cmd(
        self,
        vrp_path: Path,
        sol_path: Path,
        time_limit: float,
        seed: int,
        *,
        initial_solution_path: Path | None = None,
        initial_omega: float | None = None,
    ) -> list[str]:
        """
        Invoke AILS-II as::

            java <jvm_flags> -jar <jar> -file <vrp> -stoppingCriterion Time -limit <sec>
                -rounded true [-initialSolution <path>] [-initialOmega <v>] -outpath <sol_path>

        The ``seed`` argument is ignored (no seed flag in AILS-II yet). ``sol_path``
        is the direct output file path. JVM options precede ``-jar``.
        """
        _ = seed  # AILS-II has no RNG seed flag; ignored until -seed lands.
        cmd: list[str] = [
            "java",
            *self.jvm_flags,
            "-jar",
            str(self.binary),
            "-file",
            str(vrp_path),
            "-stoppingCriterion",
            "Time",
            "-limit",
            str(float(time_limit)),
            "-rounded",
            "true",
        ]
        if initial_solution_path is not None:
            cmd.extend(["-initialSolution", str(initial_solution_path)])
        if initial_omega is not None:
            cmd.extend(["-initialOmega", str(float(initial_omega))])
        cmd.extend(["-outpath", str(sol_path)])
        return cmd

    def solve(
        self,
        instance: CVRPInstance,
        time_limit: float,
        seed: int = 42,
    ) -> list[Route]:
        """Routing phase: no initial solution, no ``-initialOmega``."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vrp_path = _temp_vrp_path(root, instance)
            sol_path = root / "solution.sol"
            write_vrp(instance, instance.customers, vrp_path)
            cmd = self._build_cmd(vrp_path, sol_path, time_limit, seed)
            self._run_subprocess(cmd, time_limit)
            return self._solution_to_pool(sol_path, instance)

    def run_improvement(
        self,
        instance: CVRPInstance,
        time_limit: float,
        initial_solution_path: Path,
        *,
        initial_omega: float | None = None,
        seed: int = 42,
    ) -> list[Route]:
        """
        Improve from an existing solution file (``-initialSolution``).

        When ``initial_omega`` is set, passes ``-initialOmega`` (BG-AILS); otherwise
        post-SP/SC style improvement only.
        """
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vrp_path = _temp_vrp_path(root, instance)
            sol_path = root / "solution.sol"
            write_vrp(instance, instance.customers, vrp_path)
            cmd = self._build_cmd(
                vrp_path,
                sol_path,
                time_limit,
                seed,
                initial_solution_path=initial_solution_path,
                initial_omega=initial_omega,
            )
            self._run_subprocess(cmd, time_limit)
            return self._solution_to_pool(sol_path, instance)
