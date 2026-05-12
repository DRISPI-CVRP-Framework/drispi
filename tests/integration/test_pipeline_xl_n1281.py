"""Opt-in integration smoke on XL-n1281-k29 (not run in default pytest)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.haos.config import HAOSConfig
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.pipeline import DRISPIPipeline, _seqs_to_solution_routes

pytestmark = pytest.mark.integration


def _instance_path() -> Path:
    return Path("data/instances/xl/XL-n1281-k29.vrp")


@pytest.fixture(scope="module")
def xl_n1281() -> CVRPInstance:
    path = _instance_path()
    if not path.is_file():
        pytest.skip(f"Missing instance file: {path}")
    return CVRPInstance.from_vrplib("XL-n1281-k29")


def _even_k_partition(instance: CVRPInstance, k: int) -> list[list[int]]:
    customers = list(instance.customers)
    chunk = (len(customers) + k - 1) // k
    return [customers[i * chunk : min((i + 1) * chunk, len(customers))] for i in range(k)]


def _fake_cluster_instance(
    instance: CVRPInstance,
    paradigm: str,
    method: str,
    k: int,
    routes: list | None = None,
    **kwargs: object,
) -> list[list[int]]:
    del paradigm, method, routes, kwargs
    return _even_k_partition(instance, k)


def _fake_cluster_routes(
    instance: CVRPInstance,
    partition: list[list[int]],
    *_args: object,
    **_kwargs: object,
) -> list[list[list[int]]]:
    return [[[c] for c in group] for group in partition]


def test_xl_n1281_pipeline_smoke_short(xl_n1281: CVRPInstance) -> None:
    """Short wall clock; heavily mocked heavy steps. Writes under ``artifacts/xl_n1281_smoke/``."""

    out = Path("artifacts") / "xl_n1281_smoke"
    out.mkdir(parents=True, exist_ok=True)

    cfg = DRISPIConfig(
        time_limit=90.0,
        max_no_improve=500,
        n_workers=2,
        output_dir=out,
        haos_config=HAOSConfig(haos_warmup=10),
        warmup_iterations=1000,
        sp_interval=1000,
        subcluster_time_per_customer=0.02,
        bg_ails_time_limit=1.0,
        standard_improvement_time_limit=1.0,
        sp_time_limit=1.0,
    )

    def fake_bg(inst: CVRPInstance, sol: list[SolutionRoute], *a, **k):
        del a, k
        return _seqs_to_solution_routes(inst, [list(inst.customers)])

    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch(
            "drispi.pipeline.pipeline.solve_subclusters_parallel",
            side_effect=_fake_cluster_routes,
        ),
        patch("drispi.pipeline.pipeline.run_bg_ails", side_effect=fake_bg),
        patch("drispi.pipeline.pipeline.run_sp_sc", return_value=(None, False)),
    ):
        routes = DRISPIPipeline(xl_n1281, cfg).run()
    assert isinstance(routes, list)
    assert len(routes) >= 1
