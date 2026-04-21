"""UCB1-style selector stub."""

from __future__ import annotations

import numpy as np

from drispi.haos.selectors.base import BaseSelector
from drispi.core.types import WeightVector


class UcbSelector(BaseSelector):
    """Upper confidence bound using empirical means and pull counts."""

    def __init__(self, exploration: float = 1.41) -> None:
        self.exploration = exploration

    def select(self, weights: WeightVector, n_pulls: np.ndarray | None = None) -> int:
        # TODO: classic UCB1 using running means; weights may seed priors
        raise NotImplementedError
