from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


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
        self._weights = [1.0] * len(self.choices)

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

    def update(self, index: int, reward: float, decay: float) -> None:
        if index < 0 or index >= len(self._weights):
            raise IndexError("index out of range for wheel update")
        updated = decay * self._weights[index] + reward
        self._weights[index] = max(updated, 0.0)

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
