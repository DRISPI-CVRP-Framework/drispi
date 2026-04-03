"""Instance and result file I/O stubs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Solution


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    # TODO: yaml.safe_load
    raise NotImplementedError


def write_solution_json(path: Path, solution: Solution) -> None:
    """Serialize ``solution`` to JSON."""
    # TODO: dataclasses/asdict or custom encoder
    raise NotImplementedError


def read_instance(path: Path) -> CVRPInstance:
    """Dispatch to VRPLIB or other parsers based on extension."""
    # TODO: CVRPInstance.from_vrplib or similar
    raise NotImplementedError
