"""AILS-II (Java JAR) solver integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drispi.solvers._subprocess_base import _SubprocessSolver

if TYPE_CHECKING:
    from pathlib import Path


class Ails2Solver(_SubprocessSolver):
    """Subprocess wrapper around the AILS-II ``AILSII.jar``."""

    _ENV_VAR = "AILS2_JAR"
    _DEFAULT_REL_PATH = "ext/ails2/build/AILSII.jar"
    name = "ails2"

    def _build_cmd(
        self,
        vrp_path: Path,
        sol_path: Path,
        time_limit: float,
        seed: int,
    ) -> list[str]:
        """
        Invoke AILS-II as::

            java -jar <jar> -file <vrp> -stoppingCriterion Time -limit <sec>
                -rounded true -outpath <sol_path>

        The ``seed`` argument is ignored (no seed flag in AILS-II). ``sol_path``
        is the direct output file path.
        """
        _ = seed  # AILS-II has no RNG seed flag; ignored by design.
        return [
            "java",
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
            "-outpath",
            str(sol_path),
        ]
