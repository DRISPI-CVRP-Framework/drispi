"""Smoke tests for pipeline entry points."""

from __future__ import annotations

import pytest

from drispi.core.instance import CVRPInstance
from drispi.pipeline.drispi import DRISPIPipeline


def test_drispi_pipeline_run_not_implemented(small_instance: CVRPInstance) -> None:
    pipe = DRISPIPipeline(config={}, n_solver_workers=1)
    with pytest.raises(NotImplementedError):
        pipe.run(small_instance)
