"""CVRP instance representation and VRPLIB loading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import vrplib
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class CVRPInstance:
    """Capacitated vehicle routing problem instance (Euclidean, single depot)."""

    name: str
    n_customers: int
    capacity: int
    depot: tuple[float, float]
    customers: list[int]
    coordinates: dict[int, tuple[float, float]]
    demands: dict[int, int]
    INSTANCE_SEARCH_DIRS = (Path("data/instances/xl"),
                            Path("data/instances/x"))
    __hash__ = None

    def __post_init__(self) -> None:
        """Validate core instance invariants."""
        if self.n_customers != len(self.customers):
            raise ValueError("n_customers must equal len(customers)")
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if 1 not in self.coordinates:
            raise ValueError("Depot node ID 1 must exist in coordinates")
        if self.demands.get(1, 0) < 0:
            raise ValueError("Depot demand must be non-negative")
        for customer in self.customers:
            if customer not in self.coordinates:
                raise ValueError(f"Customer {customer} missing from coordinates")
            if customer not in self.demands:
                raise ValueError(f"Customer {customer} missing from demands")
        if any(demand < 0 for demand in self.demands.values()):
            raise ValueError("All demands must be non-negative")

    def euclidean_distance(self, i: int, j: int) -> float:
        """Euclidean distance between internal node IDs ``i`` and ``j``."""
        xi, yi = self.coordinates[i]
        xj, yj = self.coordinates[j]
        return math.hypot(xi - xj, yi - yj)

    def route_cost(self, customers: list[int]) -> float:
        """Route cost for customer order, with transient depot bookends."""
        if not customers:
            return 0.0
        nodes = [1, *customers, 1]
        dist = self.distance_matrix
        return float(sum(dist[nodes[k], nodes[k + 1]] for k in range(len(nodes) - 1)))

    @cached_property
    def distance_matrix(self) -> np.ndarray:
        """Full pairwise Euclidean distance matrix indexed directly by node ID."""
        size = self.n_customers + 2
        coords = np.zeros((size, 2), dtype=float)
        for node_id, (x, y) in self.coordinates.items():
            coords[node_id, 0] = x
            coords[node_id, 1] = y
        return cdist(coords, coords, metric="euclidean")

    @classmethod
    def from_vrplib(cls, instance_name: str) -> CVRPInstance:
        """Load a CVRPLib instance by name using hardcoded search directories."""
        candidate_names = [instance_name]
        if not instance_name.endswith(".vrp"):
            candidate_names.append(f"{instance_name}.vrp")

        instance_path: Path | None = None
        for directory in cls.INSTANCE_SEARCH_DIRS:
            for candidate_name in candidate_names:
                candidate_path = directory / candidate_name
                if candidate_path.exists():
                    instance_path = candidate_path
                    break
            if instance_path is not None:
                break
        if instance_path is None:
            searched = ", ".join(str(path) for path in cls.INSTANCE_SEARCH_DIRS)
            raise FileNotFoundError(
                f"Could not find instance '{instance_name}' in configured paths: {searched}"
            )

        data = vrplib.read_instance(instance_path)
        node_coords = data["node_coord"]
        demands_raw = data["demand"]
        capacity = int(data["capacity"])
        name = str(data.get("name", instance_path.stem))

        coordinates: dict[int, tuple[float, float]] = {
            idx + 1: (float(coord[0]), float(coord[1])) for idx, coord in enumerate(node_coords)
        }
        demands: dict[int, int] = {idx + 1: int(demand) for idx, demand in enumerate(demands_raw)}

        n_nodes = len(node_coords)
        if n_nodes < 1:
            raise ValueError("Instance must contain at least depot node")

        customers = list(range(2, n_nodes + 1))
        n_customers = n_nodes - 1

        if customers != list(range(2, n_customers + 2)):
            raise ValueError("Customer IDs must be 2..n in internal format")
        if 1 not in coordinates or 1 not in demands:
            raise ValueError("Depot node 1 is missing")
        if demands[1] != 0:
            raise ValueError("Depot demand must be 0")
        if sum(demands[cid] for cid in customers) <= 0:
            raise ValueError("Sum of customer demands must be positive")

        return cls(
            name=name,
            n_customers=n_customers,
            capacity=capacity,
            depot=coordinates[1],
            customers=customers,
            coordinates=coordinates,
            demands=demands,
        )
