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
    _route_cluster_ids,
    run_bg_ails_improve,
    run_bg_ails_perturb,
    run_standard_improvement,
)
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.core_manager import CoreManager
from drispi.pipeline.logger import PipelineLogger
from drispi.pipeline.snapshot import SnapshotWriter
from drispi.pipeline.subproblem import solve_subclusters_parallel, SubclusterWallTimeoutError
from drispi.route_pool.coverage import coverage_counts
from drispi.route_pool.manager import RoutePoolManager
from drispi.route_pool.pool import RoutePool
from drispi.route_pool.post_sp_improvement import add_post_standard_improvement_routes_to_pool
from drispi.sp.policy import should_run_sp_sc
from drispi.sp.solver import run_sp_sc
from drispi.utils.io import write_sol

_PHASE_SNAPSHOT_NAMES: dict[int, str] = {
    1: "dissim+cluster",
    2: "subclusters",
    3: "bg_ails",
    4: "sp_sc",
    5: "standard_ails",
}


def make_run_label(instance_name: str, *, now: datetime | None = None) -> str:
    """Build timestamped run directory label: ``<instance>_<MMDD_HHMM>``."""
    t = now or datetime.now()
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
    ) -> None:
        self._instance = instance
        self._config = config
        self._bks_cost = bks_cost
        self._core_manager = core_manager
        self._instance_id = instance_id if instance_id is not None else instance.name

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
        selection = self._haos.select(iteration, self._rng)
        best_routes = self._best_solution
        selection = self._haos.coerce_vertex_when_no_routes(
            selection,
            best_solution_available=best_routes is not None and len(best_routes) > 0,
            rng=self._rng,
        )
        if selection.paradigm == "route" and best_routes is not None:
            selection = self._haos.cap_k_for_route_clustering(selection, len(best_routes))
        is_spsc = should_run_sp_sc(
            iteration,
            self._config.warmup_iterations,
            self._config.sp_interval,
        )
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
            cluster_routes = self._solve_subclusters_parallel(
                partition,
                selection,
                iteration,
                n_clusters,
            )
        except SubclusterWallTimeoutError as exc:
            self._skip_iteration_subcluster_timeout(
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
        self._logger.log_phase_done(
            iteration,
            2,
            total_phases,
            "route",
            time.perf_counter() - t0,
            max_budget,
            {"solver": selection.solver, "k": selection.k},
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
        perturbed_sol, perturbed_indices = run_bg_ails_perturb(
            self._instance,
            combined_sol,
            dissim,
            partition,
            boundary_threshold=self._config.bg_ails_boundary_threshold,
            small_cluster_cap=self._config.bg_ails_small_cluster_cap,
            small_cluster_alpha=self._config.bg_ails_small_cluster_alpha,
            seed=bg_seed,
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

        t0 = time.perf_counter()
        bg_solution = run_bg_ails_improve(
            self._instance,
            perturbed_sol,
            partition,
            self._config.bg_ails_initial_omega,
            time_limit=self._config.bg_ails_time_limit,
            seed=bg_seed,
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
            self._config.bg_ails_time_limit,
            bg_cost_before=bg_cost_before,
            bg_cost_after=bg_cost,
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

        sp_result: list[Route] | None = None
        used_sp = False
        if is_spsc:
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
            )
            sp_elapsed = time.perf_counter() - t0
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

            prev_best = self._best_cost
            if final_cost < prev_best:
                add_post_standard_improvement_routes_to_pool(
                    self._pool,
                    self._instance,
                    sp_result,
                    final_seqs,
                    final_cost,
                    prev_best,
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

    def _solve_subclusters_parallel(
        self,
        partition: list[list[int]],
        selection: HAOSSelection,
        iteration: int,
        n_clusters: int,
    ) -> list[list[Route]]:
        seed = self._solver_seed + iteration * 10007
        if self._core_manager is not None:
            n_workers = self._core_manager.acquire(self._instance_id, n_clusters)
            try:
                return solve_subclusters_parallel(
                    self._instance,
                    partition,
                    selection.solver,
                    self._config.subcluster_time_per_customer,
                    n_workers,
                    seed,
                )
            finally:
                self._core_manager.release(self._instance_id, n_workers)
        n_workers = min(self._config.n_workers, n_clusters)
        return solve_subclusters_parallel(
            self._instance,
            partition,
            selection.solver,
            self._config.subcluster_time_per_customer,
            n_workers,
            seed,
        )

    def _skip_iteration_subcluster_timeout(
        self,
        iteration: int,
        selection: HAOSSelection,
        is_spsc: bool,
        total_phases: int,
        joint_prob: float,
        elapsed: float,
        max_budget: float,
        cluster_sizes: list[int],
        exc: SubclusterWallTimeoutError,
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
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
        path = self.run_dir / "status.json"
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
