"""CVRP instance representation and VRPLIB loading stubs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CVRPInstance:
    """Capacitated vehicle routing problem instance (Euclidean, single depot).

    Attributes:
        name: Instance identifier.
        n_customers: Number of customer nodes (excluding depot in count semantics TBD).
        capacity: Vehicle capacity.
        depot: Depot coordinates (x, y).
        customers: Customer node IDs (1-indexed; depot convention per project).
        coordinates: Node ID to (x, y).
        demands: Node ID to integer demand.
    """

    name: str
    n_customers: int
    capacity: int
    depot: tuple[float, float]
    customers: list[int]
    coordinates: dict[int, tuple[float, float]]
    demands: dict[int, int]

    @classmethod
    def from_vrplib(cls, path: Path) -> CVRPInstance:
        """Load instance from a VRPLIB-format file.

        Args:
            path: Path to the .vrp (or compatible) file.

        Returns:
            Parsed frozen instance.

        Raises:
            NotImplementedError: Stub only.
        """
        # TODO: parse VRPLIB sections (NODE_COORD_SECTION, DEMAND_SECTION, etc.)
        raise NotImplementedError

    def euclidean_distance(self, i: int, j: int) -> float:
        """Euclidean distance between nodes ``i`` and ``j`` using ``coordinates``."""
        # TODO: sqrt((xi-xj)^2 + (yi-yj)^2) with validation
        raise NotImplementedError
