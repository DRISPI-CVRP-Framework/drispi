"""Pipeline orchestration tests with heavy steps mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.haos.config import HAOSConfig
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.pipeline import DRISPIPipeline


def _fake_cluster_routes(
    instance: CVRPInstance,
    partition: list[list[int]],
    *_args: object,
    **_kwargs: object,
) -> list[list[list[int]]]:
    return [[[c] for c in group] for group in partition]


def _even_k_partition(instance: CVRPInstance, k: int) -> list[list[int]]:
    customers = list(instance.customers)
    if k <= 0:
        return [customers]
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


def _fake_bg_ails(instance: CVRPInstance, solution: list[SolutionRoute], *args, **kwargs):
    del args, kwargs
    seqs = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
    return [SolutionRoute(customers=list(s), cost=instance.route_cost(s)) for s in seqs]


def _fake_run_sp_sc(*args, **kwargs):
    del args, kwargs
    return None, False


def _fake_standard_improvement(
    instance: CVRPInstance,
    solution: list[SolutionRoute],
    *args: object,
    **kwargs: object,
) -> list[SolutionRoute]:
    del instance, args, kwargs
    return solution


def test_pipeline_runs_five_iterations(instance_12: CVRPInstance, tmp_path: Path) -> None:
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=1000,
        output_dir=tmp_path,
        haos_config=HAOSConfig(haos_warmup=2),
        warmup_iterations=100,
        sp_interval=100,
    )
    pipe = DRISPIPipeline(instance_12, cfg)

    def fake_solve(
        instance: CVRPInstance,
        partition: list[list[int]],
        solver_name: str,
        time_per_customer: float,
        n_workers: int,
        seed: int,
    ) -> list[list[list[int]]]:
        del solver_name, time_per_customer, n_workers, seed
        return _fake_cluster_routes(instance, partition)

    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch("drispi.pipeline.pipeline.solve_subclusters_parallel", side_effect=fake_solve),
        patch("drispi.pipeline.pipeline.run_bg_ails", side_effect=_fake_bg_ails),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
        patch(
            "drispi.pipeline.pipeline.run_standard_improvement",
            side_effect=_fake_standard_improvement,
        ),
    ):
        for it in range(5):
            pipe._run_iteration(it)
    assert pipe._pool.size() > 0


def test_pipeline_stops_on_time_limit(instance_12: CVRPInstance, tmp_path: Path) -> None:
    cfg = DRISPIConfig(
        time_limit=0.001,
        max_no_improve=1000,
        output_dir=tmp_path,
        warmup_iterations=1000,
    )
    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch(
            "drispi.pipeline.pipeline.solve_subclusters_parallel",
            side_effect=_fake_cluster_routes,
        ),
        patch("drispi.pipeline.pipeline.run_bg_ails", side_effect=_fake_bg_ails),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
    ):
        routes = DRISPIPipeline(instance_12, cfg).run()
    assert isinstance(routes, list)


def test_pipeline_stops_on_max_no_improve(instance_12: CVRPInstance, tmp_path: Path) -> None:
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=2,
        output_dir=tmp_path,
        warmup_iterations=1000,
    )
    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch(
            "drispi.pipeline.pipeline.solve_subclusters_parallel",
            side_effect=_fake_cluster_routes,
        ),
        patch("drispi.pipeline.pipeline.run_bg_ails", side_effect=_fake_bg_ails),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
    ):
        pipe = DRISPIPipeline(instance_12, cfg)
        pipe.run()
    assert pipe._no_improve_count >= 2


def test_update_best_resets_no_improve(instance_12: CVRPInstance, tmp_path: Path) -> None:
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=100,
        output_dir=tmp_path,
        warmup_iterations=1000,
    )
    pipe = DRISPIPipeline(instance_12, cfg)
    sol = [[2, 3], [4, 5, 6]]
    cost = sum(instance_12.route_cost(r) for r in sol)
    pipe._update_best(sol, cost, iteration=0)
    assert pipe._best_cost == cost
    assert pipe._no_improve_count == 0
    pipe._update_best(sol, cost + 1.0, iteration=1)
    assert pipe._no_improve_count == 1


def test_finalize_writes_weights_and_sol(instance_12: CVRPInstance, tmp_path: Path) -> None:
    import time

    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=100,
        output_dir=tmp_path,
        warmup_iterations=1000,
    )
    pipe = DRISPIPipeline(instance_12, cfg)
    sol = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
    cost = sum(instance_12.route_cost(r) for r in sol)
    pipe._best_solution = sol
    pipe._best_cost = cost
    pipe._iterations_completed = 3
    pipe._start_time = time.perf_counter() - 1.0
    pipe._finalize()
    assert (tmp_path / "haos_weights_final.json").is_file()
    assert (tmp_path / f"{instance_12.name}.sol").is_file()


def test_haos_weights_change_after_warmup(instance_12: CVRPInstance, tmp_path: Path) -> None:
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=100,
        output_dir=tmp_path,
        haos_config=HAOSConfig(haos_warmup=2),
        warmup_iterations=100,
        sp_interval=100,
    )
    pipe = DRISPIPipeline(instance_12, cfg)
    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch(
            "drispi.pipeline.pipeline.solve_subclusters_parallel",
            side_effect=_fake_cluster_routes,
        ),
        patch("drispi.pipeline.pipeline.run_bg_ails", side_effect=_fake_bg_ails),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
    ):
        pipe._run_iteration(0)
        pipe._run_iteration(1)
        before = pipe._haos.state_dict()
        pipe._run_iteration(2)
        after = pipe._haos.state_dict()
    assert before != after
