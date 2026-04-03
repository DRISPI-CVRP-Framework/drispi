"""DRSCI pipeline stub (related baseline or variant)."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Solution


class DRSCIPipeline:
    """Placeholder for DRSCI workflow."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, instance: CVRPInstance) -> Solution:
        # TODO: implement DRSCI-specific control flow
        raise NotImplementedError
