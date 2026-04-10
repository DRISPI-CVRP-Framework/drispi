"""FILO2 external solver integration."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from drispi.solvers._subprocess_base import (
    _ensure_outpath_dir,
    _SubprocessSolver,
    _temp_vrp_path,
)
from drispi.utils.io import write_vrp

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance
    from drispi.core.types import RoutePool


class Filo2Solver(_SubprocessSolver):
    """Subprocess wrapper around the FILO2 executable."""

    _ENV_VAR = "FILO2_BIN"
    _DEFAULT_REL_PATH = "ext/filo2/build/filo2"
    name = "filo2"

    def _build_cmd(
        self,
        vrp_path: Path,
        sol_path: Path,
        time_limit: float,
        seed: int,
    ) -> list[str]:
        """
        FILO2 writes its solution under ``--outpath`` as
        ``<stem>_seed-<seed>.vrp.sol``.

        Uses ``--optimization-seconds`` when built with ``ENABLE_TIMELIMIT``,
        ``--seed``, and ``--outpath`` as the parent directory of ``sol_path``.
        """
        return [
            str(self.binary),
            str(vrp_path),
            "--outpath",
            _ensure_outpath_dir(sol_path.parent),
            "--optimization-seconds",
            str(max(1, int(round(float(time_limit))))),
            "--seed",
            str(int(seed)),
        ]

    def solve(
        self,
        instance: CVRPInstance,
        time_limit: float,
        seed: int = 42,
    ) -> RoutePool:
        """Same as base, but reads FILO2's ``*_seed-*.vrp.sol`` output path."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vrp_path = _temp_vrp_path(root, instance)
            placeholder = root / "unused.sol"
            write_vrp(instance, instance.customers, vrp_path)
            cmd = self._build_cmd(vrp_path, placeholder, time_limit, seed)
            self._run_subprocess(cmd, time_limit)
            actual_sol = root / f"{vrp_path.name}_seed-{int(seed)}.vrp.sol"
            return self._solution_to_pool(actual_sol, instance)
