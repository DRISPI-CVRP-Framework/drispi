"""Roulette / softmax selection stub."""

from __future__ import annotations

import numpy as np

from drispi.haos.selectors.base import BaseSelector
from drispi.core.types import WeightVector


class RouletteSelector(BaseSelector):
    """Probability proportional to positive weights."""

    def select(self, weights: WeightVector, n_pulls: np.ndarray | None = None) -> int:
        # TODO: np.random.choice with p = softmax or normalized weights
        raise NotImplementedError
