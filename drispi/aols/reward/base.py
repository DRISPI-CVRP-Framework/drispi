"""Abstract reward function."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReward(ABC):
    """Maps pipeline outcomes to a scalar reward."""

    @abstractmethod
    def compute(self, context: dict[str, Any]) -> float:
        """Evaluate reward from logged metrics in ``context``."""
        raise NotImplementedError
