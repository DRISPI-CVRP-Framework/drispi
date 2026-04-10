"""Shared subprocess-based solver harness (binaries and JARs)."""

from __future__ import annotations

import os
import re
import subprocess
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from drispi.core.solution import Route
from drispi.solvers.base import BaseSolver
from drispi.utils.io import read_sol, write_vrp

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance
    from drispi.core.types import RoutePool

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_outpath_dir(path: Path) -> str:
    """Return an outpath string with a trailing separator (FILO/FILO2 convention)."""
    s = os.fspath(path.resolve())
    if not s.endswith(os.sep):
        s += os.sep
    return s


def _temp_vrp_path(root: Path, instance: CVRPInstance) -> Path:
    """Build a safe ``.vrp`` path under ``root`` from ``instance.name``."""
    base = instance.name.strip().replace("/", "_").replace("\\", "_") or "instance"
    if base.lower().endswith(".vrp"):
        return root / base
    return root / f"{base}.vrp"


class _SubprocessSolver(BaseSolver):
    """Private base for solvers invoked via ``subprocess``."""

    _ENV_VAR: str
    _DEFAULT_REL_PATH: str

    def __init__(self, binary_path: str | Path | None = None) -> None:
        """
        Resolve the executable or JAR in order:

        1. Explicit ``binary_path`` when provided.
        2. Environment variable named :py:attr:`_ENV_VAR`.
        3. ``<repo_root> / _DEFAULT_REL_PATH`` (repo root is two levels above
           the ``drispi`` package directory containing this module).

        Raises:
            FileNotFoundError: If no existing file is found after resolution.
        """
        self._binary_path: Path = self._resolve_binary_path(binary_path)

    def _resolve_binary_path(self, binary_path: str | Path | None) -> Path:
        candidates: list[Path] = []
        if binary_path is not None:
            candidates.append(Path(binary_path).expanduser())
        env_val = os.environ.get(self._ENV_VAR)
        if env_val:
            candidates.append(Path(env_val).expanduser())
        candidates.append(_REPO_ROOT / self._DEFAULT_REL_PATH)

        for path in candidates:
            if path.is_file():
                return path.resolve()

        searched = ", ".join(os.fspath(p) for p in candidates)
        raise FileNotFoundError(
            f"{self.__class__.__name__}: could not find binary/JAR. "
            f"Tried: {searched}. Set {self._ENV_VAR} or pass binary_path=..."
        )

    @property
    def binary(self) -> Path:
        """Resolved path to the solver binary or JAR."""
        return self._binary_path

    @abstractmethod
    def _build_cmd(
        self,
        vrp_path: Path,
        sol_path: Path,
        time_limit: float,
        seed: int,
    ) -> list[str]:
        """Build the command line (including executable or ``java -jar`` prefix)."""
        raise NotImplementedError

    def _run_subprocess(self, cmd: list[str], time_limit: float) -> None:
        """Run ``cmd`` with hard timeout ``time_limit + 60`` seconds."""
        hard_timeout = float(time_limit) + 60.0
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=hard_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{self.name} subprocess exceeded hard timeout ({hard_timeout} s)."
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            detail = stderr or stdout or "(no output)"
            raise RuntimeError(
                f"{self.name} failed with exit code {completed.returncode}: {detail}"
            )

    def _solution_to_pool(self, sol_path: Path, instance: CVRPInstance) -> RoutePool:
        """
        Parse a VRPLIB-style ``.sol`` into :class:`~drispi.core.solution.Route` objects.

        Normalizes ``Cost <value>`` (no colon) to ``Cost:`` so :func:`read_sol` accepts
        FILO/FILO2/AILS-II exports as well as ``Cost:`` from :func:`write_sol`.
        """
        raw = sol_path.read_text(encoding="utf-8")
        fixed = re.sub(r"^(\s*)Cost\s+", r"\1Cost: ", raw, flags=re.MULTILINE)
        path_for_read = sol_path
        if fixed != raw:
            path_for_read = sol_path.parent / f"{sol_path.stem}.normalized.sol"
            path_for_read.write_text(fixed, encoding="utf-8")
        customer_routes, _cost = read_sol(path_for_read)
        return [
            Route(customers=seq, cost=instance.route_cost(seq)) for seq in customer_routes
        ]

    def solve(
        self,
        instance: CVRPInstance,
        time_limit: float,
        seed: int = 42,
    ) -> RoutePool:
        """
        Write the instance to a temp ``.vrp``, run the solver, read the solution.

        Uses :class:`tempfile.TemporaryDirectory` for all I/O. Subprocess hard
        timeout is ``time_limit + 60`` seconds.

        Raises:
            RuntimeError: If the subprocess exits with a non-zero status.
        """
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vrp_path = _temp_vrp_path(root, instance)
            sol_path = root / "solution.sol"
            write_vrp(instance, instance.customers, vrp_path)
            cmd = self._build_cmd(vrp_path, sol_path, time_limit, seed)
            self._run_subprocess(cmd, time_limit)
            return self._solution_to_pool(sol_path, instance)
