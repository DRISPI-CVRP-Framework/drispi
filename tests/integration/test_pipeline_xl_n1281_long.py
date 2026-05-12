"""Long-run XL test (opt-in via ``pytest -m slow``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from drispi.core.instance import CVRPInstance
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.pipeline import DRISPIPipeline

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def xl_n1281() -> CVRPInstance:
    path = Path("data/instances/xl/XL-n1281-k29.vrp")
    if not path.is_file():
        pytest.skip(f"Missing instance file: {path}")
    return CVRPInstance.from_vrplib("XL-n1281-k29")


def test_xl_n1281_pipeline_long_budget(xl_n1281: CVRPInstance) -> None:
    """
    Ceiling time_limit >= 1200s; may stop earlier on max_no_improve or resource limits.
    Writes under ``artifacts/xl_n1281_long/``. Run explicitly:
    ``pytest -m "slow and integration" tests/integration/test_pipeline_xl_n1281_long.py``.
    """
    out = Path("artifacts") / "xl_n1281_long"
    out.mkdir(parents=True, exist_ok=True)

    cfg = DRISPIConfig(
        time_limit=7200.0,
        max_no_improve=100,
        n_workers=4,
        output_dir=out,
        warmup_iterations=10,
        sp_interval=3,
        subcluster_time_per_customer=0.05,
        bg_ails_time_limit=90.0,
        standard_improvement_time_limit=150.0,
        sp_time_limit=300.0,
    )
    routes = DRISPIPipeline(xl_n1281, cfg).run()
    assert isinstance(routes, list)
    assert (out / "haos_weights_final.json").is_file()
