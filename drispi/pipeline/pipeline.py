"""
Full DRISPI orchestration: HAOS-driven decomposition, parallel cluster solves,
boundary-guided improvement, route pool maintenance, SP/SC, and post-MIP AILS.

This module is the single narrative entry point for how subsystems connect.
"""

from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np

from drispi.clustering.dissimilarity import compute_dissimilarity_matrix
from drispi.clustering.interface import cluster_instance
from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.core.types import Route
from drispi.haos.config import HAOSConfig, HAOSRewardConfig
from drispi.haos.haos import HAOS, HAOSSelection
from drispi.haos.tag import HAOSTag
from drispi.haos.weights_io import save_weights
from drispi.improvement.bg_ails import (
    _resolve_n_chains,
    _route_cluster_ids,
    run_bg_ails_improve,
    run_bg_ails_perturb,
    run_standard_improvement,
)
from drispi.improvement.bg_ails_budget import (
    bg_ails_budget_seconds,
    check_bg_ails_divisor_coupling,
)
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.core_manager import CoreManager
from drispi.pipeline.cores import CoresConfig, log_current_affinity, resolve_cores, set_affinity
from drispi.pipeline.logger import PipelineLogger
from drispi.pipeline.snapshot import SnapshotWriter
from drispi.pipeline.sp_sc_async import AsyncSpScController, SpScResult
from drispi.pipeline.subproblem import SubclusterSolveError, solve_subclusters_parallel
from drispi.route_pool.coverage import coverage_counts
from drispi.route_pool.manager import RoutePoolManager
from drispi.route_pool.pool import RoutePool
from drispi.route_pool.post_sp_improvement import add_post_standard_improvement_routes_to_pool
from drispi.solvers.ails2 import Ails2Solver
from drispi.sp.policy import should_run_sp_sc
from drispi.sp.solver import run_sp_sc
from drispi.utils.io import write_sol
from drispi.utils.time import local_now

_PHASE_SNAPSHOT_NAMES: dict[int, str] = {
    1: "dissim+cluster",
    2: "subclusters",
    3: "bg_ails",
    4: "sp_sc",
    5: "standard_ails",
}


def make_run_label(instance_name: str, *, now: datetime | None = None) -> str:
    """Build timestamped run directory label: ``<instance>_<MMDD_HHMM>``."""
    t = now or local_now()
    return f"{instance_name}_{t.strftime('%m%d_%H%M')}"


def _seqs_to_solution_routes(instance: CVRPInstance, seqs: list[list[int]]) -> list[SolutionRoute]:
    """Wrap plain customer sequences as ``SolutionRoute`` for Java-backed improvement."""
    return [SolutionRoute(customers=list(s), cost=instance.route_cost(s)) for s in seqs]


def _solution_routes_to_seqs(routes: list[SolutionRoute]) -> list[list[int]]:
    """Flatten ``SolutionRoute`` objects back to customer ID lists for the pool / SP."""
    return [list(r.customers) for r in routes]


def _partition_to_cluster_assignments(partition: list[list[int]]) -> dict[int, int]:
    return {c: k for k, cluster in enumerate(partition) for c in cluster}


def _changed_route_indices(before: list[Route], after: list[Route]) -> list[int]:
    before_sets = {frozenset(r) for r in before}
    return [i for i, r in enumerate(after) if frozenset(r) not in before_sets]


def _build_haos_config(config: DRISPIConfig) -> HAOSConfig:
    """Build internal HAOSConfig from flat DRISPIConfig fields."""
    return HAOSConfig(
        decay=config.haos_decay,
        haos_warmup=config.haos_warmup,
        starting_weight=config.haos_starting_weight,
        rewards=HAOSRewardConfig(
            reward_new_best=config.haos_reward_new_best,
            reward_improvement=config.haos_reward_improvement,
            reward_no_improvement=config.haos_reward_no_improvement,
            reward_no_solution=config.haos_reward_no_solution,
            deferred_new_best=config.haos_deferred_new_best,
            deferred_improvement=config.haos_deferred_improvement,
            deferred_no_improvement=config.haos_deferred_no_improvement,
        ),
        k_candidates=list(config.haos_k_candidates),
        min_weight_k=config.haos_min_weight_k,
        lambda_demand_values=list(config.haos_lambda_demand_values),
        min_weight_lambda=config.haos_min_weight_lambda,
        paradigm_values=list(config.haos_paradigm_values),
        min_weight_paradigm=config.haos_min_weight_paradigm,
        vertex_method_values=list(config.haos_vertex_method_values),
        min_weight_vertex_method=config.haos_min_weight_vertex_method,
        route_method_values=list(config.haos_route_method_values),
        min_weight_route_method=config.haos_min_weight_route_method,
        solver_values=list(config.haos_solver_values),
        min_weight_solver=config.haos_min_weight_solver,
    )


class DRISPIPipeline:
    """
    End-to-end DRISPI loop: operator selection, clustering, subcluster routing,
    BG-AILS, pool + SP/SC scheduling, standard AILS after MIP, and HAOS rewards.
    """

    def __init__(
        self,
        instance: CVRPInstance,
        config: DRISPIConfig,
        bks_cost: float | None = None,
        *,
        run_label: str | None = None,
        core_manager: CoreManager | None = None,
        instance_id: str | None = None,
        cli_cpus: str | None = None,
    ) -> None:
        self._instance = instance
        self._config = config
        self._bks_cost = bks_cost

        check_bg_ails_divisor_coupling(
            subcluster_time_per_customer=config.subcluster_time_per_customer,
            divisor_assumes_time_per_customer=(
                config.bg_ails_divisor_assumes_time_per_customer
            ),
        )

        mode = (config.sp_sc_mode or "sync").strip().lower()
        if mode not in ("off", "sync", "async"):
            raise ValueError(f"Invalid sp_sc_mode: {config.sp_sc_mode!r}")
        trigger = (config.sp_sc_trigger or "iteration").strip().lower()
        if trigger not in ("iteration", "wallclock"):
            raise ValueError(f"Invalid sp_sc_trigger: {config.sp_sc_trigger!r}")
        if mode == "sync" and trigger == "wallclock":
            raise ValueError(
                "sp_sc_trigger='wallclock' is only supported with "
                "sp_sc_mode='async'; use trigger='iteration' for sync "
                "(warmup + sp_interval)"
            )
        self._sp_sc_mode = mode
        self._sp_sc_trigger = trigger

        self._cores: CoresConfig = resolve_cores(
            cores_total=config.cores_total,
            cores_dri=config.cores_dri,
            cores_sp=config.cores_sp,
            cores_cpu_list=config.cores_cpu_list,
            n_workers=config.n_workers,
            cli_cpus=cli_cpus,
        )
        # CoreManager budget is DRI-only when a cores: block is present.
        if core_manager is not None:
            self._core_manager = core_manager
        elif self._cores.from_cores_block:
            self._core_manager = CoreManager(self._cores.dri, self._cores.dri)
        else:
            self._core_manager = None
        self._instance_id = instance_id if instance_id is not None else instance.name
        self._batch_rounds_hist: list[int] = []

        if self._cores.dri_cpus is not None:
            set_affinity(self._cores.dri_cpus, label="pipeline-main-dri")
        else:
            log_current_affinity("pipeline-main")

        rng = np.random.default_rng(config.seed)
        self._solver_seed = int(rng.integers(0, 2**31))
        self._haos_seed = int(rng.integers(0, 2**31))
        self._bg_seed = int(rng.integers(0, 2**31))
        self._rng = random.Random(self._haos_seed)

        self._haos_config = _build_haos_config(config)
        self._haos = HAOS(self._haos_config, instance, rng=self._rng)
        self._pool = RoutePool()
        self._manager = RoutePoolManager(
            max_pool_size=config.max_pool_size,
            min_coverage=config.min_coverage,
            diversity_weight=config.pool_diversity_weight,
            warmup_iterations=config.warmup_iterations,
            sp_interval=config.sp_interval,
        )

        self._best_solution: list[Route] | None = None
        self._best_cost = float("inf")
        self._last_cost = float("inf")
        self._initial_cost: float | None = None
        self._no_improve_count = 0
        self._start_time = 0.0
        self._iterations_completed = 0
        self._stopped_by_time = False
        self._stopped_by_no_improve = False
        self._cancelled = False
        self._last_improve_phase: str = ""

        label = run_label if run_label is not None else make_run_label(instance.name)
        self._logger = PipelineLogger(
            label,
            config.output_dir,
            bks=bks_cost,
        )
        self._snapshot_writer = SnapshotWriter(self.run_dir)
        self._logger.log_init(config, instance)

        self._spsc_adopted = 0
        self._spsc_invocations = 0
        self._drain_latencies: list[float] = []
        self._async_ctrl: AsyncSpScController | None = None
        if self._sp_sc_mode == "async":
            sp = max(1, self._cores.sp if self._cores.from_cores_block else 1)
            self._async_ctrl = AsyncSpScController(
                sp_cpus=self._cores.sp_cpus,
                gurobi_threads=sp,
                ails_apc=sp,
                xmx=Ails2Solver.DEFAULT_XMX,
                overlap_policy=config.overlap_policy,  # type: ignore[arg-type]
                trigger=self._sp_sc_trigger,  # type: ignore[arg-type]
                interval_minutes=config.interval_minutes,
                warmup_iterations=config.warmup_iterations,
                sp_interval=config.sp_interval,
            )
            self._async_ctrl.start()

    @property
    def run_dir(self) -> Path:
        return self._logger.run_dir

    @property
    def best_cost(self) -> float:
        return self._best_cost

    @property
    def iterations_completed(self) -> int:
        return self._iterations_completed

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason()

    def run(self) -> list[Route]:
        """
        Run until wall time or stagnation limits, then persist HAOS weights and
        the best solution on disk.
        """
        self._start_time = time.perf_counter()
        self._write_status("running")
        iteration = 0
        try:
            while True:
                elapsed = time.perf_counter() - self._start_time
                if iteration > 0 and elapsed >= self._config.time_limit:
                    self._stopped_by_time = True
                    break
                if iteration > 0 and self._no_improve_count >= self._config.max_no_improve:
                    self._stopped_by_no_improve = True
                    break

                self._run_iteration(iteration)
                iteration += 1
        except KeyboardInterrupt:
            self._cancelled = True

        self._iterations_completed = iteration
        self._finalize()
        if self._best_solution is None:
            return []
        return self._best_solution

    def _write_phase_snapshot(
        self,
        iteration: int,
        phase_num: int,
        is_spsc: bool,
        total_phases: int,
        *,
        cluster_assignments: dict[int, int] | None = None,
        routes: list[Route] | None = None,
        route_cluster_ids: list[int] | None = None,
        perturbed_route_indices: list[int] | None = None,
        changed_route_indices: list[int] | None = None,
        selected_route_indices: list[int] | None = None,
        phase_tag: str | None = None,
    ) -> None:
        self._snapshot_writer.write_phase(
            iteration,
            phase_num,
            _PHASE_SNAPSHOT_NAMES[phase_num],
            is_spsc,
            total_phases,
            self._instance,
            cluster_assignments=cluster_assignments,
            routes=routes,
            route_cluster_ids=route_cluster_ids,
            perturbed_route_indices=perturbed_route_indices,
            changed_route_indices=changed_route_indices,
            selected_route_indices=selected_route_indices,
            phase_tag=phase_tag,
        )

    def _run_iteration(self, iteration: int) -> None:
        iter_t0 = time.perf_counter()
        # Async results apply only at iteration start (HAOS-safe).
        if self._async_ctrl is not None:
            result = self._async_ctrl.poll_result(nonblocking=True)
            if result is not None:
                self._apply_async_spsc_result(result, iteration)

        selection = self._haos.select(iteration, self._rng)
        best_routes = self._best_solution
        selection = self._haos.coerce_vertex_when_no_routes(
            selection,
            best_solution_available=best_routes is not None and len(best_routes) > 0,
            rng=self._rng,
        )
        if selection.paradigm == "route" and best_routes is not None:
            selection = self._haos.cap_k_for_route_clustering(selection, len(best_routes))
        is_spsc = self._should_run_sync_spsc(iteration)
        total_phases = 5 if is_spsc else 3
        joint_prob = self._haos.joint_probability(selection)
        self._logger.log_haos_roll(
            iteration,
            selection,
            is_spsc,
            joint_prob,
            levels=None,
        )
        angular_offset = self._rng.uniform(0.0, 2.0 * math.pi)

        t0 = time.perf_counter()
        dissim = compute_dissimilarity_matrix(
            self._instance,
            selection.lambda_demand,
            angular_offset,
        )

        routes_for_cluster = (
            [list(r) for r in best_routes] if selection.paradigm == "route" else None
        )

        partition = cluster_instance(
            self._instance,
            paradigm=selection.paradigm,
            method=selection.method,
            k=selection.k,
            routes=routes_for_cluster,
            lambda_demand=selection.lambda_demand,
            angular_offset=angular_offset,
            seed=self._rng.randint(0, 2**31 - 1),
        )
        cluster_assignments = _partition_to_cluster_assignments(partition)
        self._write_phase_snapshot(
            iteration,
            1,
            is_spsc,
            total_phases,
            cluster_assignments=cluster_assignments,
        )
        self._snapshot_writer.cleanup_old_snapshots(iteration)

        self._logger.log_phase_done(
            iteration,
            1,
            total_phases,
            "decompose",
            time.perf_counter() - t0,
            None,
            {
                "lambda_demand": selection.lambda_demand,
                "paradigm": selection.paradigm,
                "method": selection.method,
            },
        )

        cluster_sizes = [len(c) for c in partition]
        max_budget = max(
            (max(10.0, float(sz) * self._config.subcluster_time_per_customer) for sz in cluster_sizes),
            default=0.0,
        )
        t0 = time.perf_counter()
        n_clusters = len(partition)
        try:
            cluster_routes, n_rounds = self._solve_subclusters_parallel(
                partition,
                selection,
                iteration,
                n_clusters,
            )
            self._batch_rounds_hist.append(n_rounds)
        except SubclusterSolveError as exc:
            self._skip_iteration_subcluster_failure(
                iteration,
                selection,
                is_spsc,
                total_phases,
                joint_prob,
                time.perf_counter() - t0,
                max_budget,
                cluster_sizes,
                exc,
            )
            return
        dr_stage_wall_seconds = time.perf_counter() - t0
        self._logger.log_phase_done(
            iteration,
            2,
            total_phases,
            "route",
            dr_stage_wall_seconds,
            max_budget,
            {"solver": selection.solver, "k": selection.k, "batch_rounds": n_rounds},
            cluster_sizes=cluster_sizes,
        )

        combined_seqs = [seq for group in cluster_routes for seq in group]
        route_cluster_ids = _route_cluster_ids(combined_seqs, partition)
        self._write_phase_snapshot(
            iteration,
            2,
            is_spsc,
            total_phases,
            cluster_assignments=cluster_assignments,
            routes=combined_seqs,
            route_cluster_ids=route_cluster_ids,
        )

        combined_sol = _seqs_to_solution_routes(self._instance, combined_seqs)

        bg_seed = self._bg_seed + iteration
        n_chains = _resolve_n_chains(
            partition,
            n_chains=None,
            n_chains_mode=self._config.bg_ails_n_chains_mode,  # type: ignore[arg-type]
        )
        perturbed_sol, perturbed_indices = run_bg_ails_perturb(
            self._instance,
            combined_sol,
            dissim,
            partition,
            boundary_threshold=self._config.bg_ails_boundary_threshold,
            small_cluster_cap=self._config.bg_ails_small_cluster_cap,
            small_cluster_alpha=self._config.bg_ails_small_cluster_alpha,
            seed=bg_seed,
            pair_selection=self._config.bg_ails_pair_selection,
            n_chains_mode=self._config.bg_ails_n_chains_mode,
        )
        self._write_phase_snapshot(
            iteration,
            3,
            is_spsc,
            total_phases,
            cluster_assignments=cluster_assignments,
            routes=combined_seqs,
            route_cluster_ids=route_cluster_ids,
            perturbed_route_indices=perturbed_indices,
            phase_tag="pre",
        )

        bg_ails_budget = bg_ails_budget_seconds(
            self._instance.n_customers,
            min_budget=self._config.bg_ails_min_budget,
            divisor=self._config.bg_ails_divisor,
        )
        t0 = time.perf_counter()
        bg_solution = run_bg_ails_improve(
            self._instance,
            perturbed_sol,
            partition,
            self._config.bg_ails_initial_omega,
            time_limit=bg_ails_budget,
            seed=bg_seed,
            solver=self._ails2_solver(),
        )
        bg_elapsed = time.perf_counter() - t0
        pert_seqs = _solution_routes_to_seqs(perturbed_sol)
        bg_seqs = _solution_routes_to_seqs(bg_solution)
        bg_changed = _changed_route_indices(pert_seqs, bg_seqs)
        bg_cost = sum(self._instance.route_cost(r) for r in bg_seqs)
        bg_improved = bg_cost < self._best_cost

        self._write_phase_snapshot(
            iteration,
            3,
            is_spsc,
            total_phases,
            cluster_assignments=cluster_assignments,
            routes=bg_seqs,
            route_cluster_ids=route_cluster_ids,
            changed_route_indices=bg_changed,
            phase_tag="post",
        )

        bg_cost_before = sum(self._instance.route_cost(r) for r in combined_seqs)
        self._logger.log_phase_done(
            iteration,
            3,
            total_phases,
            "bg_ails",
            bg_elapsed,
            bg_ails_budget,
            bg_cost_before=bg_cost_before,
            bg_cost_after=bg_cost,
            bg_ails_budget_seconds=bg_ails_budget,
            bg_ails_actual_wall_seconds=bg_elapsed,
            k=selection.k,
            n_chains=n_chains,
            max_cluster_size=max(cluster_sizes) if cluster_sizes else 0,
            dr_stage_wall_seconds=dr_stage_wall_seconds,
        )
        if bg_improved:
            self._log_improvement(iteration, bg_cost, bg_seqs, "bg_ails")

        imm = self._haos.compute_reward(
            bg_cost,
            self._best_cost,
            self._last_cost,
            self._haos_config.rewards,
            is_deferred=False,
        )
        self._haos.update_immediate(selection, iteration, imm)

        op_tag = selection.to_tag(iteration)
        for seq in bg_seqs:
            self._pool.add(seq, self._instance.route_cost(seq), haos_tag=op_tag)

        if bg_improved:
            self._update_best(bg_seqs, iteration, "bg_ails")

        self._manager.maybe_evict(self._pool)

        if self._sp_sc_mode == "off":
            # Hard tripwire: SC/SP must never run when mode is off.
            if is_spsc:
                raise RuntimeError(
                    "SC/SP scheduled while sp_sc.mode=off — disable path violated"
                )

        sp_result: list[Route] | None = None
        used_sp = False
        if self._sp_sc_mode == "async" and self._async_ctrl is not None:
            best_for_job = (
                self._best_solution if self._best_solution is not None else bg_seqs
            )
            self._async_ctrl.maybe_trigger(
                pool=self._pool,
                best_solution=best_for_job,
                best_cost=float(self._best_cost)
                if math.isfinite(self._best_cost)
                else float(bg_cost),
                instance=self._instance,
                time_limit=self._config.sp_time_limit,
                mip_gap=self._config.mip_gap,
                min_coverage=self._config.min_coverage,
                std_improve_limit=self._config.standard_improvement_time_limit,
                seed=self._rng.randint(0, 2**31 - 1),
                iteration=iteration,
            )
        elif is_spsc and self._sp_sc_mode == "sync":
            t0 = time.perf_counter()
            sp_result, used_sp = run_sp_sc(
                self._pool,
                self._instance,
                self._manager,
                iteration,
                self._best_solution if self._best_solution is not None else bg_seqs,
                time_limit=self._config.sp_time_limit,
                mip_gap=self._config.mip_gap,
                min_coverage=self._config.min_coverage,
                warmup_iterations=self._config.warmup_iterations,
                sp_interval=self._config.sp_interval,
                threads=(
                    self._cores.sp
                    if self._cores.from_cores_block and self._cores.sp > 0
                    else None
                ),
            )
            sp_elapsed = time.perf_counter() - t0
            self._spsc_invocations += 1
            mode_done = "SP" if used_sp else "SC"
            if sp_result is not None and len(sp_result) > 0:
                self._write_phase_snapshot(
                    iteration,
                    4,
                    is_spsc,
                    total_phases,
                    routes=sp_result,
                    selected_route_indices=list(range(len(sp_result))),
                )
            cov = sorted(coverage_counts(self._pool, self._instance).values())
            cov_avg = sum(cov) / len(cov) if cov else 0.0
            self._logger.log_phase_done(
                iteration,
                4,
                total_phases,
                "sp_sc",
                sp_elapsed,
                self._config.sp_time_limit,
                {
                    "sc_or_sp": mode_done,
                    "pool_size": self._pool.size(),
                    "avg_coverage": cov_avg,
                    "min_coverage": self._config.min_coverage,
                    "coverage_min": cov[0] if cov else 0,
                    "coverage_median": float(median(cov)) if cov else 0.0,
                    "coverage_max": cov[-1] if cov else 0,
                },
                lp_fractionality=self._manager.last_lp_fractionality,
            )

        iter_cost = bg_cost
        if sp_result is not None and len(sp_result) > 0:
            sp_input_sol = _seqs_to_solution_routes(self._instance, sp_result)
            t0 = time.perf_counter()
            final_sol = run_standard_improvement(
                self._instance,
                sp_input_sol,
                self._config.standard_improvement_time_limit,
                seed=self._rng.randint(0, 2**31 - 1),
                solver=self._ails2_solver(),
            )
            std_elapsed = time.perf_counter() - t0
            final_seqs = _solution_routes_to_seqs(final_sol)
            final_cost = sum(self._instance.route_cost(r) for r in final_seqs)
            iter_cost = final_cost
            std_changed = _changed_route_indices(sp_result, final_seqs)

            self._write_phase_snapshot(
                iteration,
                5,
                is_spsc,
                total_phases,
                routes=final_seqs,
                changed_route_indices=std_changed,
            )

            self._logger.log_phase_done(
                iteration,
                5,
                total_phases,
                "standard_ails",
                std_elapsed,
                self._config.standard_improvement_time_limit,
            )
            if final_cost < self._best_cost:
                self._log_improvement(iteration, final_cost, final_seqs, "standard_ails")

            deferred = self._haos.compute_reward(
                final_cost,
                self._best_cost,
                self._last_cost,
                self._haos_config.rewards,
                is_deferred=True,
            )
            contributing = self._contributing_tags_from_sp_routes(sp_result)
            self._haos.update_deferred(contributing, iteration, deferred)

            add_post_standard_improvement_routes_to_pool(
                self._pool,
                self._instance,
                sp_result,
                final_seqs,
                iteration,
                selection,
            )
            self._update_best(final_seqs, iteration, "standard_ails")
            self._last_cost = final_cost
        else:
            if not bg_improved:
                self._update_best(bg_seqs, iteration, "bg_ails")
            self._last_cost = bg_cost

        self._haos.update_final(selection, iteration)
        self._logger.log_haos_roll(
            iteration,
            selection,
            is_spsc,
            joint_prob,
            levels=self._haos.state_dict(),
        )

        rejected, replaced = self._pool.get_iter_counters()
        self._pool.reset_iter_counters()
        entries = self._pool.routes()
        diversity_avg = (
            sum(e.diversity_rank_score for e in entries) / len(entries) if entries else 0.0
        )
        quality_avg = (
            sum(e.quality_rank_score for e in entries) / len(entries) if entries else 0.0
        )
        self._logger.log_summary(
            iteration,
            iter_cost,
            self._best_cost,
            self._last_cost,
            self._no_improve_count,
            pool_size=self._pool.size(),
            pool_diversity_avg=diversity_avg,
            pool_quality_avg=quality_avg,
            duplicates_rejected=rejected,
            duplicates_replaced=replaced,
        )

    def _should_run_sync_spsc(self, iteration: int) -> bool:
        """Sync always uses iteration trigger (warmup + sp_interval)."""
        if self._sp_sc_mode != "sync":
            return False
        return should_run_sp_sc(
            iteration,
            self._config.warmup_iterations,
            self._config.sp_interval,
        )

    def _apply_async_spsc_result(self, result: SpScResult, iteration: int) -> None:
        """Apply a drained async SC/SP (+ AILS) result against the live incumbent."""
        applied_ts = time.time()
        drain_latency = (
            applied_ts - result.ready_ts if result.ready_ts else None
        )
        if drain_latency is not None:
            self._drain_latencies.append(drain_latency)
        self._spsc_invocations += 1

        log_iter = int(result.snapshot_iteration)
        metrics = result.metrics or {}
        sp_elapsed = float(metrics.get("total_wall_s") or 0.0)
        sp_budget = metrics.get("sp_time_limit")
        if sp_budget is None:
            sp_budget = self._config.sp_time_limit
        mode_done = "SP" if result.used_sp else "SC"
        self._logger.log_phase_done(
            log_iter,
            1,
            2,
            "sp_sc",
            sp_elapsed,
            float(sp_budget),
            {
                "sc_or_sp": mode_done,
                "pool_size": metrics.get("pool_size"),
                "avg_coverage": metrics.get("avg_coverage"),
                "min_coverage": metrics.get(
                    "min_coverage", self._config.min_coverage
                ),
            },
            tag="ASYNC 1/2",
        )

        ails_elapsed = metrics.get("ails_wall_s")
        if ails_elapsed is not None:
            std_budget = metrics.get("std_improve_limit")
            if std_budget is None:
                std_budget = self._config.standard_improvement_time_limit
            self._logger.log_phase_done(
                log_iter,
                2,
                2,
                "standard_ails",
                float(ails_elapsed),
                float(std_budget),
                tag="ASYNC 2/2",
            )

        if result.lp_weights:
            self._manager.update_scores_after_solve(self._pool, result.lp_weights)

        reason = result.reason
        adopted = False
        if result.final_routes and result.final_cost is not None:
            prev_best = self._best_cost
            # Deferred HAOS from pre-AILS SP routes
            deferred = self._haos.compute_reward(
                float(result.final_cost),
                self._best_cost,
                self._last_cost,
                self._haos_config.rewards,
                is_deferred=True,
            )
            sp_routes = result.sp_routes or []
            if sp_routes:
                contributing = self._contributing_tags_from_sp_routes(sp_routes)
                self._haos.update_deferred(contributing, iteration, deferred)
            placeholder = (
                self._selection_from_sp_routes(sp_routes, log_iter)
                if sp_routes
                else self._selection_from_sp_routes(result.final_routes, log_iter)
            )
            add_post_standard_improvement_routes_to_pool(
                self._pool,
                self._instance,
                sp_routes,
                result.final_routes,
                log_iter,
                placeholder,
            )
            if float(result.final_cost) < prev_best:
                self._update_best(result.final_routes, log_iter, "standard_ails")
                adopted = True
                self._spsc_adopted += 1
                self._log_improvement(
                    log_iter, float(result.final_cost), result.final_routes, "standard_ails"
                )
                reason = "adopted"
            else:
                reason = reason or "worse_than_incumbent"
            self._last_cost = float(result.final_cost)
        elif reason == "mip_no_solution":
            pass
        else:
            reason = reason or "empty_solution"

        # Structured-only apply event (console uses ASYNC 1/2 + 2/2 above).
        self._logger._emit_json(  # noqa: SLF001
            {
                "type": "spsc_apply",
                "iteration": log_iter,
                "apply_iteration": iteration,
                "snapshot_iteration": log_iter,
                "job_id": result.job_id,
                "adopted": adopted,
                "reason": reason,
                "snapshot_best_cost": result.snapshot_best_cost,
                "live_best_cost": self._best_cost,
                "final_cost": result.final_cost,
                "result_ready_ts": result.ready_ts,
                "applied_ts": applied_ts,
                "drain_latency_s": drain_latency,
                "metrics": result.metrics,
                "used_sp": result.used_sp,
            }
        )

    def _selection_from_sp_routes(
        self, sp_routes: list[Route], iteration: int
    ) -> HAOSSelection:
        """Build a HAOSSelection for pool tagging without rolling the HAOS RNG."""
        for r in sp_routes:
            tag = self._pool.get_haos_tag(r)
            if tag is not None:
                return HAOSSelection(
                    k=tag.k,
                    lambda_demand=tag.lambda_demand,
                    paradigm=tag.paradigm,
                    method=tag.method,
                    solver=tag.solver,
                    k_index=0,
                    lambda_index=0,
                    paradigm_index=0,
                    method_index=0,
                    solver_index=0,
                )
        _ = iteration
        return HAOSSelection(
            k=1,
            lambda_demand=0.0,
            paradigm="vertex",
            method="kmeans",
            solver="ails2",
            k_index=0,
            lambda_index=0,
            paradigm_index=0,
            method_index=0,
            solver_index=0,
        )

    def _ails2_solver(self) -> Ails2Solver:
        """AILS-II with JVM flags; ActiveProcessorCount from cores.sp when set."""
        if self._cores.from_cores_block and self._cores.sp > 0:
            return Ails2Solver(active_processor_count=self._cores.sp)
        return Ails2Solver()

    def _solve_subclusters_parallel(
        self,
        partition: list[list[int]],
        selection: HAOSSelection,
        iteration: int,
        n_clusters: int,
    ) -> tuple[list[list[Route]], int]:
        seed = self._solver_seed + iteration * 10007
        dri_cpus = self._cores.dri_cpus
        request = n_clusters
        if self._cores.from_cores_block:
            # Never request more DRI slots than the reserved dri budget when
            # CoreManager is shared; still ask for n_clusters and let CM grant.
            pass
        if self._core_manager is not None:
            n_workers = self._core_manager.acquire(self._instance_id, request)
            try:
                return solve_subclusters_parallel(
                    self._instance,
                    partition,
                    selection.solver,
                    self._config.subcluster_time_per_customer,
                    n_workers,
                    seed,
                    dri_cpus=dri_cpus,
                )
            finally:
                self._core_manager.release(self._instance_id, n_workers)
        n_workers = min(self._cores.dri, n_clusters)
        return solve_subclusters_parallel(
            self._instance,
            partition,
            selection.solver,
            self._config.subcluster_time_per_customer,
            n_workers,
            seed,
            dri_cpus=dri_cpus,
        )

    def _skip_iteration_subcluster_failure(
        self,
        iteration: int,
        selection: HAOSSelection,
        is_spsc: bool,
        total_phases: int,
        joint_prob: float,
        elapsed: float,
        max_budget: float,
        cluster_sizes: list[int],
        exc: SubclusterSolveError,
    ) -> None:
        self._logger.log_iteration_skipped(
            iteration,
            "route",
            str(exc),
            elapsed=elapsed,
            budget=max_budget,
        )
        self._logger.log_phase_done(
            iteration,
            2,
            total_phases,
            "route",
            elapsed,
            max_budget,
            {"solver": selection.solver, "k": selection.k, "failed": True},
            cluster_sizes=cluster_sizes,
        )
        imm = self._haos.compute_reward(
            None,
            self._best_cost,
            self._last_cost,
            self._haos_config.rewards,
            is_deferred=False,
        )
        self._haos.update_immediate(selection, iteration, imm)
        self._haos.update_final(selection, iteration)
        self._no_improve_count += 1
        self._logger.log_haos_roll(
            iteration,
            selection,
            is_spsc,
            joint_prob,
            levels=self._haos.state_dict(),
        )
        rejected, replaced = self._pool.get_iter_counters()
        self._pool.reset_iter_counters()
        entries = self._pool.routes()
        diversity_avg = (
            sum(e.diversity_rank_score for e in entries) / len(entries) if entries else 0.0
        )
        quality_avg = (
            sum(e.quality_rank_score for e in entries) / len(entries) if entries else 0.0
        )
        iter_cost = self._last_cost if math.isfinite(self._last_cost) else self._best_cost
        self._logger.log_summary(
            iteration,
            iter_cost,
            self._best_cost,
            self._last_cost,
            self._no_improve_count,
            pool_size=self._pool.size(),
            pool_diversity_avg=diversity_avg,
            pool_quality_avg=quality_avg,
            duplicates_rejected=rejected,
            duplicates_replaced=replaced,
        )

    def _log_improvement(
        self,
        iteration: int,
        cost: float,
        routes: list[Route],
        phase_name: str,
    ) -> None:
        self._logger.log_improve(iteration, cost, phase_name)
        if self._logger.beats_bks(cost):
            self._logger.log_new_bks(
                iteration,
                cost,
                phase_name,
                routes,
                self._instance,
            )

    def _contributing_tags_from_sp_routes(self, sp_routes: list[Route]) -> list[HAOSTag]:
        tags: list[HAOSTag] = []
        for r in sp_routes:
            tag = self._pool.get_haos_tag(r)
            if tag is not None:
                tags.append(tag)
        return tags

    def _update_best(self, solution: list[Route], iteration: int, phase_name: str) -> None:
        """Track global best S*, mark elite pool rows, and stagnation.

        Callers must ensure ``solution`` routes are already present in the pool
        (with correct HAOS tags) before invoking this on an improvement.
        """
        recomputed = float(sum(self._instance.route_cost(list(r)) for r in solution))
        if self._initial_cost is None and math.isfinite(recomputed):
            self._initial_cost = recomputed

        if recomputed < self._best_cost:
            self._best_solution = [list(r) for r in solution]
            self._best_cost = recomputed
            self._no_improve_count = 0
            self._last_improve_phase = phase_name
            self._pool.set_elite(self._best_solution)
            self._snapshot_writer.write_best_solution(
                self._best_solution,
                self._best_cost,
                iteration,
                phase_name,
                self._instance,
            )
        else:
            self._no_improve_count += 1

    def _finalize(self) -> None:
        """Persist HAOS state, best .sol, and final log block."""
        discarded = False
        if self._async_ctrl is not None:
            self._async_ctrl.shutdown()
            discarded = self._async_ctrl.discarded_in_flight_at_shutdown
            self._logger._emit(  # noqa: SLF001
                None,
                "FINAL",
                (
                    f"async SP/SC totals invocations={self._async_ctrl.invocation_count} "
                    f"adopted={self._spsc_adopted} "
                    f"skipped={self._async_ctrl.skipped_trigger_count} "
                    f"discarded_in_flight={discarded} "
                    f"sp_busy_frac={self._async_ctrl.sp_busy_fraction()}"
                ),
                json_event={
                    "type": "spsc_run_totals",
                    "invocation_count": self._async_ctrl.invocation_count,
                    "adopted_count": self._spsc_adopted,
                    "skipped_trigger_count": self._async_ctrl.skipped_trigger_count,
                    "discarded_in_flight_at_shutdown": discarded,
                    "batch_rounds_max": max(self._batch_rounds_hist)
                    if self._batch_rounds_hist
                    else None,
                    "batch_rounds_mean": (
                        sum(self._batch_rounds_hist) / len(self._batch_rounds_hist)
                        if self._batch_rounds_hist
                        else None
                    ),
                    "drain_latency_max": max(self._drain_latencies)
                    if self._drain_latencies
                    else None,
                    "drain_latency_mean": (
                        sum(self._drain_latencies) / len(self._drain_latencies)
                        if self._drain_latencies
                        else None
                    ),
                    "sp_core_busy_fraction": self._async_ctrl.sp_busy_fraction(),
                    "cores": {
                        "total": self._cores.total,
                        "dri": self._cores.dri,
                        "sp": self._cores.sp,
                        "cpu_list": self._cores.cpu_list,
                        "from_cores_block": self._cores.from_cores_block,
                    },
                    "sp_sc_mode": self._sp_sc_mode,
                    "sp_sc_trigger": self._sp_sc_trigger,
                    "jvm_flags": self._ails2_solver().jvm_flags,
                },
            )
        self._snapshot_writer.stop()
        out = self.run_dir
        out.mkdir(parents=True, exist_ok=True)
        save_weights(
            self._haos,
            out / "haos_weights_final.json",
            self._instance.name,
            self._iterations_completed,
        )
        if self._best_solution is not None:
            written_cost = float(
                sum(self._instance.route_cost(r) for r in self._best_solution)
            )
            write_sol(self._best_solution, written_cost, out / f"{self._instance.name}.sol")

        elapsed = time.perf_counter() - self._start_time
        self._logger.log_final(
            self._iterations_completed,
            elapsed,
            self._best_cost,
            self._stopped_by_no_improve,
            self._stopped_by_time,
            self._best_solution or [],
            self._pool,
            self._instance,
            cancelled=self._cancelled,
        )
        self._write_status(
            "cancelled" if self._cancelled else "finished",
            stop_reason=self._stop_reason(),
            elapsed=round(elapsed, 1),
            iterations=self._iterations_completed,
        )

        if self._config.run_analysis:
            from drispi.pipeline.analysis import load_jsonl, run_auto_analysis

            lines = load_jsonl(self.run_dir)
            run_auto_analysis(self.run_dir, lines, self._bks_cost)

    def _stop_reason(self) -> str | None:
        if self._cancelled:
            return "cancelled"
        if self._stopped_by_time:
            return "time_limit"
        if self._stopped_by_no_improve:
            return "no_improve"
        return None

    def _write_status(self, status: str, **extra: object) -> None:
        """Persist run status for the dashboard (running / finished / cancelled)."""
        payload: dict[str, object] = {
            "status": status,
            "updated_at": local_now().isoformat(timespec="seconds"),
            **extra,
        }
        path = self.run_dir / "status.json"
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
