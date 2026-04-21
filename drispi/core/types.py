"""Shared type aliases and AOLS decision record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Route: TypeAlias = list[int]
RoutePool: TypeAlias = list[Route]
Cluster: TypeAlias = list[int]
WeightVector: TypeAlias = np.ndarray
OperatorID: TypeAlias = str


@dataclass
class AOLSDecision:
    """One hierarchical AOLS choice for a pipeline segment."""

    k: int
    dissimilarity: str
    decomp_type: str
    method: str
    solver: str
