"""Abstract clustering interface and name registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from drispi.core.instance import CVRPInstance
from drispi.core.types import Cluster

CLUSTERING_REGISTRY: dict[str, type[BaseClustering]] = {}

C = TypeVar("C", bound="BaseClustering")


def register_clustering(name: str) -> Callable[[type[C]], type[C]]:
    """Decorator to register a clustering implementation under ``name``."""

    def _decorator(cls: type[C]) -> type[C]:
        CLUSTERING_REGISTRY[name] = cls
        return cls

    return _decorator


class BaseClustering(ABC):
    """Base class for instance clustering into customer groups."""

    def __init__(self, instance: CVRPInstance, config: dict[str, Any]) -> None:
        self.instance = instance
        self.config = config

    @abstractmethod
    def cluster(self, n_clusters: int) -> list[Cluster]:
        """Partition customers into ``n_clusters`` clusters (lists of node IDs)."""
        raise NotImplementedError
