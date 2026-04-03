"""Abstract arm selector."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from drispi.core.types import WeightVector


class BaseSelector(ABC):
    """Chooses an index given weights or sufficient statistics."""

    @abstractmethod
    def select(self, weights: WeightVector, n_pulls: np.ndarray | None = None) -> int:
        """Return chosen arm index in ``[0, len(weights))``."""
        raise NotImplementedError
