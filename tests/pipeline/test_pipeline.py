"""Pipeline orchestration tests with heavy steps mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.haos.haos import HAOSSelection
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.pipeline import DRISPIPipeline
from drispi.pipeline.subproblem import SubclusterSolveError, SubclusterWallTimeoutError


def _fake_cluster_routes(
    instance: CVRPInstance,
    partition: list[list[int]],
    *_args: object,
    **_kwargs: object,
) -> tuple[list[list[list[int]]], int]:
    return [[[c] for c in group] for group in partition], 1


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


def _fake_bg_perturb(instance: CVRPInstance, solution: list[SolutionRoute], *args, **kwargs):
    del args, kwargs, solution
    seqs = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
    routes = [SolutionRoute(customers=list(s), cost=instance.route_cost(s)) for s in seqs]
    return routes, []


def _fake_bg_improve(
    instance: CVRPInstance,
    perturbed: list[SolutionRoute],
    partition: object,
    initial_omega: float,
    **kwargs: object,
) -> list[SolutionRoute]:
    del partition, initial_omega, kwargs
    return perturbed


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


def test_bg_ails_improvement_kept_on_sp_sc_iteration(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    """When bg_ails improves S* but standard_ails regresses, S* stays at bg cost."""
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=1000,
        output_dir=tmp_path,
        warmup_iterations=0,
        sp_interval=1,
        min_coverage=1,
    )
    pipe = DRISPIPipeline(instance_12, cfg, bks_cost=None)
    pipe._best_cost = 1e9
    pipe._best_solution = None

    cheap_seqs = [[2, 3], [4, 5, 6]]
    cheap_cost = float(sum(instance_12.route_cost(r) for r in cheap_seqs))
    expensive_seqs = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
    expensive_cost = float(sum(instance_12.route_cost(r) for r in expensive_seqs))
    assert cheap_cost < expensive_cost

    def fake_bg_perturb(
        instance: CVRPInstance,
        solution: list[SolutionRoute],
        *args: object,
        **kwargs: object,
    ) -> tuple[list[SolutionRoute], list[int]]:
        del instance, solution, args, kwargs
        routes = [
            SolutionRoute(customers=list(s), cost=instance_12.route_cost(s)) for s in cheap_seqs
        ]
        return routes, []

    def fake_bg_improve(
        instance: CVRPInstance,
        perturbed: list[SolutionRoute],
        partition: object,
        initial_omega: float,
        **kwargs: object,
    ) -> list[SolutionRoute]:
        del instance, perturbed, partition, initial_omega, kwargs
        return [
            SolutionRoute(customers=list(s), cost=instance_12.route_cost(s)) for s in cheap_seqs
        ]

    def fake_sp(*args: object, **kwargs: object):
        del args, kwargs
        return cheap_seqs, False

    def fake_std(
        instance: CVRPInstance,
        solution: list[SolutionRoute],
        *args: object,
        **kwargs: object,
    ) -> list[SolutionRoute]:
        del instance, solution, args, kwargs
        return [
            SolutionRoute(customers=list(s), cost=instance_12.route_cost(s))
            for s in expensive_seqs
        ]

    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch(
            "drispi.pipeline.pipeline.solve_subclusters_parallel",
            side_effect=_fake_cluster_routes,
        ),
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=fake_bg_improve),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=fake_sp),
        patch("drispi.pipeline.pipeline.run_standard_improvement", side_effect=fake_std),
    ):
        pipe._run_iteration(0)

    assert pipe._best_cost == cheap_cost
    assert pipe._best_cost < expensive_cost


def test_pipeline_runs_five_iterations(instance_12: CVRPInstance, tmp_path: Path) -> None:
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=1000,
        output_dir=tmp_path,
        haos_warmup=2,
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
        **kwargs: object,
    ) -> tuple[list[list[list[int]]], int]:
        del solver_name, time_per_customer, n_workers, seed, kwargs
        return _fake_cluster_routes(instance, partition)

    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch("drispi.pipeline.pipeline.solve_subclusters_parallel", side_effect=fake_solve),
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=_fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=_fake_bg_improve),
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
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=_fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=_fake_bg_improve),
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
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=_fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=_fake_bg_improve),
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
    for route in sol:
        pipe._pool.add(route, instance_12.route_cost(route))
    cost = sum(instance_12.route_cost(r) for r in sol)
    pipe._update_best(sol, iteration=0, phase_name="bg_ails")
    assert pipe._best_cost == cost
    assert pipe._no_improve_count == 0
    pipe._update_best(sol, iteration=1, phase_name="bg_ails")
    assert pipe._no_improve_count == 1


def test_bg_improvement_tags_routes_before_update_best(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    """BG-AILS pool routes must keep the iteration HAOS tag after a new best."""
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=100,
        output_dir=tmp_path,
        warmup_iterations=100,
        sp_interval=100,
    )
    route_selection = HAOSSelection(
        k=2,
        lambda_demand=0.0,
        paradigm="vertex",
        method="kmeans",
        solver="pyvrp",
        k_index=1,
        lambda_index=0,
        paradigm_index=0,
        method_index=0,
        solver_index=0,
    )
    pipe = DRISPIPipeline(instance_12, cfg)

    with (
        patch.object(pipe._haos, "select", return_value=route_selection),
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch(
            "drispi.pipeline.pipeline.solve_subclusters_parallel",
            side_effect=_fake_cluster_routes,
        ),
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=_fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=_fake_bg_improve),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
    ):
        pipe._run_iteration(0)

    expected_tag = route_selection.to_tag(0)
    for route in pipe._best_solution or []:
        assert pipe._pool.get_haos_tag(route) == expected_tag


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
    run_dir = pipe.run_dir
    assert (run_dir / "haos_weights_final.json").is_file()
    assert (run_dir / f"{instance_12.name}.sol").is_file()


def test_route_clustering_uses_best_solution_not_pool(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    """Route-based clustering must use S* routes, not the full route pool."""
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=100,
        output_dir=tmp_path,
        warmup_iterations=100,
        sp_interval=100,
    )
    pipe = DRISPIPipeline(instance_12, cfg)
    best = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
    pipe._best_solution = [list(r) for r in best]
    pipe._best_cost = sum(instance_12.route_cost(r) for r in best)
    pipe._pool.add([2, 3, 4], instance_12.route_cost([2, 3, 4]))
    pipe._pool.add([5, 6, 7, 8, 9, 10, 11, 12, 13], instance_12.route_cost([5, 6, 7, 8, 9, 10, 11, 12, 13]))

    route_selection = HAOSSelection(
        k=2,
        lambda_demand=0.0,
        paradigm="route",
        method="kmeans",
        solver="pyvrp",
        k_index=1,
        lambda_index=0,
        paradigm_index=1,
        method_index=0,
        solver_index=0,
    )
    cluster_calls: list[dict[str, object]] = []

    def capture_cluster(
        instance: CVRPInstance,
        paradigm: str,
        method: str,
        k: int,
        routes: list | None = None,
        **kwargs: object,
    ) -> list[list[int]]:
        del instance, method, kwargs
        cluster_calls.append({"paradigm": paradigm, "k": k, "routes": routes})
        return _even_k_partition(instance_12, k)

    with (
        patch.object(pipe._haos, "select", return_value=route_selection),
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=capture_cluster),
        patch(
            "drispi.pipeline.pipeline.solve_subclusters_parallel",
            side_effect=_fake_cluster_routes,
        ),
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=_fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=_fake_bg_improve),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
    ):
        pipe._run_iteration(1)

    assert len(cluster_calls) == 1
    assert cluster_calls[0]["paradigm"] == "route"
    assert cluster_calls[0]["routes"] == best


def test_haos_weights_change_after_warmup(instance_12: CVRPInstance, tmp_path: Path) -> None:
    # The no-improvement reward must be large enough to survive the global
    # end-of-iteration decay + weight floor: (10 + r) * decay > 10.
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=100,
        output_dir=tmp_path,
        haos_warmup=2,
        haos_reward_no_improvement=2.0,
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
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=_fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=_fake_bg_improve),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
    ):
        pipe._run_iteration(0)
        pipe._run_iteration(1)
        before = pipe._haos.state_dict()
        pipe._run_iteration(2)
        after = pipe._haos.state_dict()
    assert before != after


def test_pipeline_skips_iteration_on_subcluster_timeout(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=1000,
        output_dir=tmp_path,
        haos_warmup=2,
        warmup_iterations=100,
        sp_interval=100,
    )
    pipe = DRISPIPipeline(instance_12, cfg)
    calls = {"n": 0}

    def fake_solve(*args: object, **kwargs: object) -> list[list[list[int]]]:
        del args, kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            raise SubclusterWallTimeoutError("wall timeout")
        return _fake_cluster_routes(instance_12, _even_k_partition(instance_12, 2))

    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch("drispi.pipeline.pipeline.solve_subclusters_parallel", side_effect=fake_solve),
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=_fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=_fake_bg_improve),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
        patch(
            "drispi.pipeline.pipeline.run_standard_improvement",
            side_effect=_fake_standard_improvement,
        ),
    ):
        routes = pipe.run()

    assert calls["n"] >= 2
    assert pipe._no_improve_count >= 1
    assert isinstance(routes, list)


def test_pipeline_skips_iteration_on_subcluster_worker_failure(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    cfg = DRISPIConfig(
        time_limit=1e9,
        max_no_improve=1000,
        output_dir=tmp_path,
        haos_warmup=2,
        warmup_iterations=100,
        sp_interval=100,
    )
    pipe = DRISPIPipeline(instance_12, cfg)
    calls = {"n": 0}

    def fake_solve(*args: object, **kwargs: object) -> list[list[list[int]]]:
        del args, kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            raise SubclusterSolveError(
                "Subcluster worker failed: ails2 subprocess exceeded hard timeout"
            )
        return _fake_cluster_routes(instance_12, _even_k_partition(instance_12, 2))

    with (
        patch("drispi.pipeline.pipeline.cluster_instance", side_effect=_fake_cluster_instance),
        patch("drispi.pipeline.pipeline.solve_subclusters_parallel", side_effect=fake_solve),
        patch("drispi.pipeline.pipeline.run_bg_ails_perturb", side_effect=_fake_bg_perturb),
        patch("drispi.pipeline.pipeline.run_bg_ails_improve", side_effect=_fake_bg_improve),
        patch("drispi.pipeline.pipeline.run_sp_sc", side_effect=_fake_run_sp_sc),
        patch(
            "drispi.pipeline.pipeline.run_standard_improvement",
            side_effect=_fake_standard_improvement,
        ),
    ):
        routes = pipe.run()

    assert calls["n"] >= 2
    assert pipe._no_improve_count >= 1
    assert isinstance(routes, list)
