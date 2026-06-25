"""Smoke tests for pipeline entry points."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from drispi.core.instance import CVRPInstance
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.pipeline import DRISPIPipeline


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


def test_drispi_pipeline_smoke_one_iteration(small_instance: CVRPInstance, tmp_path: Path) -> None:
    from drispi.core.solution import Route as SolutionRoute
    from drispi.pipeline.pipeline import _seqs_to_solution_routes

    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=100,
        output_dir=tmp_path,
        haos_warmup=10,
        warmup_iterations=1000,
        sp_interval=1000,
    )

    def fake_bg_perturb(inst: CVRPInstance, sol: list[SolutionRoute], *a, **k):
        del a, k, sol
        routes = _seqs_to_solution_routes(inst, [list(inst.customers)])
        return routes, []

    def fake_bg_improve(
        inst: CVRPInstance,
        perturbed: list[SolutionRoute],
        partition: object,
        initial_omega: float,
        **k: object,
    ) -> list[SolutionRoute]:
        del inst, partition, initial_omega, k
        return perturbed

    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch("drispi.pipeline.pipeline.solve_subclusters_parallel", side_effect=_fake_cluster_routes),
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=fake_bg_improve),
        patch("drispi.pipeline.pipeline.run_sp_sc", return_value=(None, False)),
    ):
        pipe = DRISPIPipeline(small_instance, cfg)
        pipe._run_iteration(0)
    assert pipe._pool.size() >= 1
