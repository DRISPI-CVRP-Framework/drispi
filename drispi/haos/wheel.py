from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# Initial weight per option and hard lower bound: rewards add on top of this
# baseline and end-of-iteration decay can never push a weight below it.
WEIGHT_FLOOR = 10.0


@dataclass
class RouletteWheel:
    """Single HAOS roulette wheel with probability floor selection."""

    choices: list[Any]
    min_weight: float
    _weights: list[float] = field(init=False)

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("RouletteWheel must have at least one choice")
        if not (0.0 <= self.min_weight < 1.0):
            raise ValueError("min_weight must be in [0.0, 1.0)")
        if self.min_weight * len(self.choices) > 1.0:
            raise ValueError("min_weight * n_choices > 1.0: floor is infeasible")
        self._weights = [WEIGHT_FLOOR] * len(self.choices)

    def _effective_probabilities(self) -> list[float]:
        total = sum(self._weights)
        if total <= 0.0:
            raw = [1.0 / len(self._weights)] * len(self._weights)
        else:
            raw = [weight / total for weight in self._weights]
        floored = [max(prob, self.min_weight) for prob in raw]
        floored_total = sum(floored)
        return [prob / floored_total for prob in floored]

    def select(self, rng: random.Random) -> tuple[int, Any]:
        probabilities = self._effective_probabilities()
        index = rng.choices(range(len(self.choices)), weights=probabilities, k=1)[0]
        return index, self.choices[index]

    def update(self, index: int, reward: float) -> None:
        """Add ``reward`` to the weight at ``index`` (floored at WEIGHT_FLOOR)."""
        if index < 0 or index >= len(self._weights):
            raise IndexError("index out of range for wheel update")
        self._weights[index] = max(self._weights[index] + reward, WEIGHT_FLOOR)

    def decay_all(self, decay: float) -> None:
        """Multiply every weight by ``decay``, never dropping below WEIGHT_FLOOR."""
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0.0, 1.0]")
        self._weights = [max(weight * decay, WEIGHT_FLOOR) for weight in self._weights]

    def probabilities(self) -> list[float]:
        return self._effective_probabilities()

    def raw_weights(self) -> list[float]:
        return list(self._weights)

    def set_weights(self, weights: list[float]) -> None:
        if len(weights) != len(self.choices):
            raise ValueError("weights length must match number of choices")
        if any(weight < 0.0 for weight in weights):
            raise ValueError("weights cannot contain negative values")
        self._weights = list(weights)
