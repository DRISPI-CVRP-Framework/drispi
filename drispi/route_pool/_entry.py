"""Internal route-pool entry model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean

from drispi.core.types import Route
from drispi.haos.tag import HAOSTag


@dataclass
class RouteEntry:
    """One route record stored in the route pool."""

    route: Route
    cost: float
    customer_set: frozenset[int]
    is_elite: bool = False
    haos_tag: HAOSTag | None = None
    quality_scores: deque[float] = field(default_factory=lambda: deque(maxlen=3))
    diversity_scores: deque[float] = field(default_factory=lambda: deque(maxlen=3))

    @property
    def quality_rank_score(self) -> float:
        """Running average of the last 1-3 quality scores."""
        return mean(self.quality_scores) if self.quality_scores else 0.0

    @property
    def diversity_rank_score(self) -> float:
        """Running average of the last 1-3 diversity scores."""
        return mean(self.diversity_scores) if self.diversity_scores else 0.0
