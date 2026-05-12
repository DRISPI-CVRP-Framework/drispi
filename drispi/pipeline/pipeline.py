"""
Full DRISPI orchestration: HAOS-driven decomposition, parallel cluster solves,
boundary-guided improvement, route pool maintenance, SP/SC, and post-MIP AILS.

This module is the single narrative entry point for how subsystems connect.
"""

from __future__ import annotations

import math
import random
import time

from drispi.clustering.dissimilarity import compute_dissimilarity_matrix
from drispi.clustering.interface import cluster_instance
from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.core.types import Route
from drispi.haos.haos import HAOS
from drispi.haos.tag import HAOSTag
from drispi.haos.weights_io import save_weights
from drispi.improvement.bg_ails import run_bg_ails, run_standard_improvement
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.subproblem import solve_subclusters_parallel
from drispi.route_pool.coverage import coverage_counts
from drispi.route_pool.manager import RoutePoolManager
from drispi.route_pool.pool import RoutePool
from drispi.route_pool.post_sp_improvement import add_post_standard_improvement_routes_to_pool
from drispi.sp.policy import should_run_sp_sc
from drispi.sp.solver import run_sp_sc
from drispi.utils.io import write_sol


def _seqs_to_solution_routes(instance: CVRPInstance, seqs: list[list[int]]) -> list[SolutionRoute]:
    """Wrap plain customer sequences as ``SolutionRoute`` for Java-backed improvement."""
    return [SolutionRoute(customers=list(s), cost=instance.route_cost(s)) for s in seqs]


def _solution_routes_to_seqs(routes: list[SolutionRoute]) -> list[list[int]]:
    """Flatten ``SolutionRoute`` objects back to customer ID lists for the pool / SP."""
    return [list(r.customers) for r in routes]


class DRISPIPipeline:
    """
    End-to-end DRISPI loop: operator selection, clustering, subcluster routing,
    BG-AILS, pool + SP/SC scheduling, standard AILS after MIP, and HAOS rewards.
    """

    def __init__(self, instance: CVRPInstance, config: DRISPIConfig) -> None:
        self._instance = instance
        self._config = config
        self._rng = random.Random(config.seed)
        self._haos = HAOS(config.haos_config, instance, rng=self._rng)
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

    def _sp_sc_coverage_summary(self) -> str:
        """Minimum pool-route multiplicity vs ``min_coverage`` (SP gate), as ``x/y``."""
        req = self._config.min_coverage
        counts = coverage_counts(self._pool, self._instance)
        min_obs = min(counts.values()) if counts else 0
        return f"current_coverage={min_obs}/{req}"

    def run(self) -> list[Route]:
        """
        Run until wall time or stagnation limits, then persist HAOS weights and
        the best solution on disk.
        """
        # --- Outer loop: each pass is one HAOS-guided decomposition iteration ---
        self._start_time = time.perf_counter()
        iteration = 0
        while True:
            elapsed = time.perf_counter() - self._start_time
            if iteration > 0 and elapsed >= self._config.time_limit:
                break
            if iteration > 0 and self._no_improve_count >= self._config.max_no_improve:
                break

            self._run_iteration(iteration)
            iteration += 1

        self._iterations_completed = iteration
        self._finalize()
        if self._best_solution is None:
            return []
        return self._best_solution

    def _run_iteration(self, iteration: int) -> None:
        # ------------------------------------------------------------------
        # Step 1 — HAOS: pick operators for this iteration (k, λ, paradigm,
        # clustering method, and subcluster solver). Same selection drives tags
        # for pool entries and traceability for improvement-phase routes.
        # ------------------------------------------------------------------
        selection = self._haos.select(iteration, self._rng)
        print(self._haos.format_operator_roll(iteration, selection), flush=True)
        angular_offset = self._rng.uniform(0.0, 2.0 * math.pi)

        _t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 2 — Geometry + demand dissimilarity for clustering / BG-AILS.
        # (cluster_instance recomputes internally; we keep D for boundary ranks.)
        # ------------------------------------------------------------------
        dissim = compute_dissimilarity_matrix(
            self._instance,
            selection.lambda_demand,
            angular_offset,
        )

        # ------------------------------------------------------------------
        # Step 3 — Partition customers. Route-based methods need pool routes;
        # if the pool is still empty, fall back to vertex clustering for this
        # call only (HAOS selection is unchanged for rewards).
        # ------------------------------------------------------------------
        route_export = self._pool.as_route_pool()
        if selection.paradigm == "route" and len(route_export) == 0:
            paradigm_used = "vertex"
            routes_for_cluster: list[Route] | None = None
        else:
            paradigm_used = selection.paradigm
            routes_for_cluster = route_export if paradigm_used == "route" else None

        partition = cluster_instance(
            self._instance,
            paradigm=paradigm_used,
            method=selection.method,
            k=selection.k,
            routes=routes_for_cluster,
            lambda_demand=selection.lambda_demand,
            angular_offset=angular_offset,
            seed=self._rng.randint(0, 2**31 - 1),
        )
        print(
            f"[it {iteration}] step dissim_cluster={time.perf_counter() - _t0:.2f}s",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Step 4 — Independent CVRP solve per cluster (worker picks solver by name).
        # ------------------------------------------------------------------
        cluster_sizes = [len(c) for c in partition]
        max_budget = max(
            (max(10.0, float(sz) * self._config.subcluster_time_per_customer) for sz in cluster_sizes),
            default=0.0,
        )
        _t0 = time.perf_counter()
        cluster_routes = solve_subclusters_parallel(
            self._instance,
            partition,
            selection.solver,
            self._config.subcluster_time_per_customer,
            self._config.n_workers,
            self._config.seed + iteration * 10007,
        )
        print(
            f"[it {iteration}] step subclusters_done={time.perf_counter() - _t0:.2f}s "
            f"solver={selection.solver!r} k={selection.k} "
            f"cluster_sizes={cluster_sizes} n_workers={self._config.n_workers} "
            f"max_budget={max_budget:.1f}s",
            flush=True,
        )

        # ------------------------------------------------------------------
        # Step 5 — Merge cluster routes into one multi-vehicle solution (parent IDs).
        # ------------------------------------------------------------------
        combined_seqs = [seq for group in cluster_routes for seq in group]
        combined_sol = _seqs_to_solution_routes(self._instance, combined_seqs)

        # ------------------------------------------------------------------
        # Step 6 — Boundary-guided AILS on the full instance using partition + D.
        # ------------------------------------------------------------------
        _t0 = time.perf_counter()
        bg_solution = run_bg_ails(
            self._instance,
            combined_sol,
            dissim,
            partition,
            self._config.bg_ails_initial_omega,
            time_limit=self._config.bg_ails_time_limit,
            boundary_threshold=self._config.bg_ails_boundary_threshold,
            seed=self._rng.randint(0, 2**31 - 1),
        )
        print(
            f"[it {iteration}] step bg_ails_done={time.perf_counter() - _t0:.2f}s "
            f"time_limit={self._config.bg_ails_time_limit:.1f}s",
            flush=True,
        )
        bg_seqs = _solution_routes_to_seqs(bg_solution)
        bg_cost = sum(self._instance.route_cost(r) for r in bg_seqs)

        # ------------------------------------------------------------------
        # Step 7 — Immediate HAOS reward from BG-AILS cost vs global / last iter.
        # ------------------------------------------------------------------
        imm = self._haos.compute_reward(
            bg_cost,
            self._best_cost,
            self._last_cost,
            self._config.haos_config.rewards,
            is_deferred=False,
        )
        self._haos.update_immediate(selection, iteration, imm)

        # ------------------------------------------------------------------
        # Step 8 — Memorize BG routes in the pool with the iteration operator tag.
        # ------------------------------------------------------------------
        op_tag = selection.to_tag(iteration)
        for seq in bg_seqs:
            self._pool.add(seq, self._instance.route_cost(seq), haos_tag=op_tag)

        # ------------------------------------------------------------------
        # Step 9 — Evict low-value routes if the pool exceeds max size / fitness.
        # ------------------------------------------------------------------
        self._manager.maybe_evict(self._pool)

        # ------------------------------------------------------------------
        # Step 10–12 — Optional SP/SC + standard AILS on the MIP solution, then
        # deferred HAOS credit to routes that actually appeared in the pool before
        # SP. Post-improvement pool inserts only happen on a new global best
        # (see route_pool.post_sp_improvement).
        # ------------------------------------------------------------------
        if not should_run_sp_sc(
            iteration,
            self._config.warmup_iterations,
            self._config.sp_interval,
        ):
            sp_result, used_sp = None, False
        else:
            _t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - _t0
            mode_done = "SP" if used_sp else "SC"
            print(
                f"[it {iteration}] step sp_sc_done={elapsed:.2f}s mode={mode_done} "
                f"pool_size={self._pool.size()} {self._sp_sc_coverage_summary()} "
                f"routes_out={len(sp_result) if sp_result else 0} "
                f"time_limit={self._config.sp_time_limit:.1f}s",
                flush=True,
            )

        if sp_result is not None and len(sp_result) > 0:
            sp_input_sol = _seqs_to_solution_routes(self._instance, sp_result)
            _t0 = time.perf_counter()
            final_sol = run_standard_improvement(
                self._instance,
                sp_input_sol,
                self._config.standard_improvement_time_limit,
                seed=self._rng.randint(0, 2**31 - 1),
            )
            print(
                f"[it {iteration}] step standard_ails_done={time.perf_counter() - _t0:.2f}s "
                f"time_limit={self._config.standard_improvement_time_limit:.1f}s",
                flush=True,
            )
            final_seqs = _solution_routes_to_seqs(final_sol)
            final_cost = sum(self._instance.route_cost(r) for r in final_seqs)

            deferred = self._haos.compute_reward(
                final_cost,
                self._best_cost,
                self._last_cost,
                self._config.haos_config.rewards,
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
            self._update_best(final_seqs, iteration)
            self._last_cost = final_cost
        else:
            # No SP/SC this iteration: global tracking follows BG-AILS only.
            self._update_best(bg_seqs, iteration)
            self._last_cost = bg_cost

        # ------------------------------------------------------------------
        # Step 13 — Fold immediate + deferred rewards into roulette weights.
        # ------------------------------------------------------------------
        self._haos.update_final(selection, iteration)

    def _contributing_tags_from_sp_routes(self, sp_routes: list[Route]) -> list[HAOSTag]:
        tags: list[HAOSTag] = []
        for r in sp_routes:
            tag = self._pool.get_haos_tag(r)
            if tag is not None:
                tags.append(tag)
        return tags

    def _update_best(self, solution: list[Route], iteration: int) -> None:
        """Track global best S*, elite pool rows, stagnation, and a simple progress line."""
        recomputed = float(sum(self._instance.route_cost(list(r)) for r in solution))
        if self._initial_cost is None and math.isfinite(recomputed):
            self._initial_cost = recomputed

        if recomputed < self._best_cost:
            self._best_solution = [list(r) for r in solution]
            self._best_cost = recomputed
            self._no_improve_count = 0
            for r in self._best_solution:
                self._pool.add(r, self._instance.route_cost(r))
            self._pool.set_elite(self._best_solution)
        else:
            self._no_improve_count += 1

        gap_best = 0.0
        if self._initial_cost is not None and self._initial_cost > 0:
            if math.isfinite(self._best_cost):
                gap_best = (self._best_cost - self._initial_cost) / self._initial_cost * 100.0
        print(
            f"[it {iteration}] best={self._best_cost:.4f} "
            f"gap_vs_start={gap_best:.2f}% "
            f"no_improve={self._no_improve_count}"
        )

    def _finalize(self) -> None:
        """Persist HAOS state and best .sol once the main loop terminates."""
        out = self._config.output_dir
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
        print(
            f"Finished {self._iterations_completed} iterations in {elapsed:.1f}s; "
            f"best cost {self._best_cost}"
        )
