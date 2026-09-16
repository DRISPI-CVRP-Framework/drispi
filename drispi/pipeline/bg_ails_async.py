"""Long-lived async BG-AILS worker process and main-side controller.

Option 3': the worker owns a startup copy of the instance and rebuilds the
dissimilarity matrix per job from ``(instance, lambda_demand, angular_offset)``.
Each job ships only the routed solution, the partition, and scalars (~KBs);
the ~n^2 float64 matrix never crosses the process boundary.

Single-slot handoff: at most one job is ever in flight. The pipeline blocks
until the worker is free, consume-drains the previous result, then launches the
next job — bounding BG lag to exactly one iteration (required for HAOS decay
policy (b) to stay arm-unbiased).

The worker drops all references to the dissimilarity matrix after perturb and
before launching the AILS-II JVM so the numpy peak and the JVM heap never
overlap in the same process.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any

from drispi.clustering.dissimilarity import compute_dissimilarity_matrix
from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.haos.tag import HAOSTag
from drispi.improvement.bg_ails import run_bg_ails_improve, run_bg_ails_perturb
from drispi.pipeline.cores import affinity_logging_enabled, set_affinity
from drispi.solvers.ails2 import Ails2Solver

LOGGER = logging.getLogger(__name__)

_SHUTDOWN: Any = None


@dataclass(frozen=True)
class BgAilsJob:
    """One perturb+improve job for the BG worker.

    All fields are copies / scalars built on the main thread before enqueue —
    never live references. The dissimilarity matrix is intentionally absent:
    the worker rebuilds it from its startup instance plus ``lambda_demand`` /
    ``angular_offset`` (bit-identical to the main-thread computation).
    """

    job_id: int
    launch_iteration: int
    combined_seqs: list[list[int]]
    partition: list[list[int]]
    lambda_demand: float
    angular_offset: float
    initial_omega: float
    time_limit: float
    seed: int
    boundary_threshold: float
    small_cluster_cap: int
    small_cluster_alpha: float
    pair_selection: str
    n_chains_mode: str
    unique_first_routes: bool
    boundary_mask_first_ls: bool
    op_tag: HAOSTag
    enqueue_ts: float


@dataclass
class BgAilsResult:
    """Worker result drained (consume-once) by the main process."""

    job_id: int
    launch_iteration: int
    bg_seqs: list[list[int]] | None
    bg_cost: float | None
    cost_before: float
    perturbed_route_indices: list[int]
    op_tag: HAOSTag
    reason: str  # "ok" | "error:<ExceptionName>"
    perturb_wall_s: float
    improve_wall_s: float
    total_wall_s: float
    ready_ts: float
    enqueue_ts: float
    # Budget the job ran with (echoed from BgAilsJob.time_limit; 0.0 when the
    # worker died before a job could be attributed).
    time_limit: float = 0.0
    error: str | None = None


def _process_job(job: BgAilsJob, *, instance: CVRPInstance, xmx: str) -> BgAilsResult:
    """Rebuild dissim, perturb, free the matrix, then improve (JVM)."""
    t0 = time.perf_counter()

    combined_sol = [
        SolutionRoute(customers=list(s), cost=instance.route_cost(s))
        for s in job.combined_seqs
    ]
    cost_before = float(sum(r.cost for r in combined_sol))

    dissim = compute_dissimilarity_matrix(
        instance, job.lambda_demand, job.angular_offset
    )
    perturbed_sol, perturbed_indices, ranks_hat = run_bg_ails_perturb(
        instance,
        combined_sol,
        dissim,
        job.partition,
        boundary_threshold=job.boundary_threshold,
        small_cluster_cap=job.small_cluster_cap,
        small_cluster_alpha=job.small_cluster_alpha,
        seed=job.seed,
        pair_selection=job.pair_selection,  # type: ignore[arg-type]
        n_chains_mode=job.n_chains_mode,  # type: ignore[arg-type]
        unique_first_routes=job.unique_first_routes,
    )
    # Free the O(n^2) matrix before the JVM starts so numpy and -Xmx peaks
    # never overlap inside this process.
    del dissim
    perturb_wall_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    bg_solution = run_bg_ails_improve(
        instance,
        perturbed_sol,
        job.partition,
        job.initial_omega,
        time_limit=job.time_limit,
        seed=job.seed,
        solver=Ails2Solver(active_processor_count=1, xmx=xmx),
        ranks_hat=ranks_hat,
        boundary_threshold=job.boundary_threshold,
        boundary_mask_first_ls=job.boundary_mask_first_ls,
    )
    improve_wall_s = time.perf_counter() - t1

    bg_seqs = [list(r.customers) for r in bg_solution]
    bg_cost = float(sum(instance.route_cost(s) for s in bg_seqs))

    return BgAilsResult(
        job_id=job.job_id,
        launch_iteration=job.launch_iteration,
        bg_seqs=bg_seqs,
        bg_cost=bg_cost,
        cost_before=cost_before,
        perturbed_route_indices=perturbed_indices,
        op_tag=job.op_tag,
        reason="ok",
        perturb_wall_s=perturb_wall_s,
        improve_wall_s=improve_wall_s,
        total_wall_s=time.perf_counter() - t0,
        ready_ts=time.time(),
        enqueue_ts=job.enqueue_ts,
        time_limit=job.time_limit,
    )


def bg_ails_worker_main(
    job_queue: Queue,
    result_queue: Queue,
    instance: CVRPInstance,
    bg_cpus: list[int] | None,
    xmx: str = "4g",
) -> None:
    """
    Long-lived BG worker target.

    Becomes its own process group leader so a controller discard can kill the
    whole group (worker + any in-flight AILS-II JVM child) in one signal.
    Owns the startup instance copy; loops on ``job_queue`` until the shutdown
    sentinel (``None``).
    """
    try:
        os.setpgrp()
    except (AttributeError, OSError):  # non-POSIX or already leader
        pass

    if bg_cpus:
        set_affinity(list(bg_cpus), label=f"bg-worker pid={os.getpid()}")
    elif affinity_logging_enabled():
        LOGGER.info("bg-worker: no bg_cpus provided; leaving OS affinity unchanged")

    # Warm the distance matrix once at startup.
    _ = instance.distance_matrix

    while True:
        job = job_queue.get()
        if job is _SHUTDOWN:
            LOGGER.info("bg-worker: shutdown sentinel received")
            break
        if not isinstance(job, BgAilsJob):
            LOGGER.error("bg-worker: unexpected job type %r; skipping", type(job))
            continue
        try:
            result = _process_job(job, instance=instance, xmx=xmx)
        except Exception as exc:  # noqa: BLE001 — deliver failure to main
            LOGGER.exception("bg-worker: job_id=%s failed: %s", job.job_id, exc)
            result = BgAilsResult(
                job_id=job.job_id,
                launch_iteration=job.launch_iteration,
                bg_seqs=None,
                bg_cost=None,
                cost_before=0.0,
                perturbed_route_indices=[],
                op_tag=job.op_tag,
                reason=f"error:{type(exc).__name__}",
                perturb_wall_s=0.0,
                improve_wall_s=0.0,
                total_wall_s=0.0,
                ready_ts=time.time(),
                enqueue_ts=job.enqueue_ts,
                time_limit=job.time_limit,
                error=str(exc),
            )
        result_queue.put(result)


class AsyncBgAilsController:
    """Main-side controller for the single-slot BG-AILS worker.

    Pipeline API:

    * ``start(instance)`` — spawn the daemon worker once (startup instance copy).
    * ``launch(...)`` — enqueue one job; raises if a job is already in flight
      (callers must block+drain first — single-slot by construction).
    * ``poll_result()`` — nonblocking consume-once drain.
    * ``wait_result(...)`` — blocking consume-once drain (the handoff block),
      with worker-liveness detection so a crashed worker surfaces as a
      ``worker_died`` result instead of a hang.
    * ``shutdown(discard=...)`` — sentinel + join; ``discard=True`` kills the
      whole process group (worker + JVM child) without waiting for the job.
    """

    def __init__(
        self,
        *,
        bg_cpus: list[int] | None,
        xmx: str = "4g",
        join_timeout_s: float = 30.0,
    ) -> None:
        self.bg_cpus = list(bg_cpus) if bg_cpus is not None else None
        self.xmx = str(xmx)
        self.join_timeout_s = float(join_timeout_s)

        self._job_queue: Queue | None = None
        self._result_queue: Queue | None = None
        self._process: Process | None = None
        self._started = False
        self._in_flight = False
        self._next_job_id = 1

        # Instrumentation
        self.invocation_count = 0
        self.discarded_in_flight_at_shutdown = False

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def start(self, instance: CVRPInstance) -> None:
        """Spawn the long-lived daemon worker holding a startup instance copy."""
        if self._started:
            return
        self._job_queue = Queue()
        self._result_queue = Queue()
        self._process = Process(
            target=bg_ails_worker_main,
            args=(self._job_queue, self._result_queue, instance, self.bg_cpus, self.xmx),
            name="drispi-bg-ails-worker",
            daemon=True,
        )
        self._process.start()
        self._started = True
        LOGGER.info(
            "AsyncBgAilsController: started worker pid=%s bg_cpus=%s xmx=%s",
            self._process.pid,
            self.bg_cpus,
            self.xmx,
        )

    def launch(
        self,
        *,
        launch_iteration: int,
        combined_seqs: list[list[int]],
        partition: list[list[int]],
        lambda_demand: float,
        angular_offset: float,
        initial_omega: float,
        time_limit: float,
        seed: int,
        boundary_threshold: float,
        small_cluster_cap: int,
        small_cluster_alpha: float,
        pair_selection: str,
        n_chains_mode: str,
        op_tag: HAOSTag,
        unique_first_routes: bool = True,
        boundary_mask_first_ls: bool = True,
    ) -> int:
        """Enqueue one job (deep-copied payload). Raises when a job is in flight."""
        if not self._started or self._job_queue is None:
            raise RuntimeError("BG worker not started")
        if self._in_flight:
            raise RuntimeError(
                "BG job already in flight — callers must block+drain before launch"
            )
        job = BgAilsJob(
            job_id=self._next_job_id,
            launch_iteration=int(launch_iteration),
            combined_seqs=[list(s) for s in combined_seqs],
            partition=[list(g) for g in partition],
            lambda_demand=float(lambda_demand),
            angular_offset=float(angular_offset),
            initial_omega=float(initial_omega),
            time_limit=float(time_limit),
            seed=int(seed),
            boundary_threshold=float(boundary_threshold),
            small_cluster_cap=int(small_cluster_cap),
            small_cluster_alpha=float(small_cluster_alpha),
            pair_selection=str(pair_selection),
            n_chains_mode=str(n_chains_mode),
            unique_first_routes=bool(unique_first_routes),
            boundary_mask_first_ls=bool(boundary_mask_first_ls),
            op_tag=op_tag,
            enqueue_ts=time.time(),
        )
        self._next_job_id += 1
        self._job_queue.put(job)
        self._in_flight = True
        self.invocation_count += 1
        return job.job_id

    def _worker_died_result(self) -> BgAilsResult:
        self._in_flight = False
        return BgAilsResult(
            job_id=-1,
            launch_iteration=-1,
            bg_seqs=None,
            bg_cost=None,
            cost_before=0.0,
            perturbed_route_indices=[],
            op_tag=HAOSTag(
                k=0,
                lambda_demand=0.0,
                paradigm="vertex",
                method="",
                solver="",
                iteration=-1,
            ),
            reason="worker_died",
            perturb_wall_s=0.0,
            improve_wall_s=0.0,
            total_wall_s=0.0,
            ready_ts=time.time(),
            enqueue_ts=0.0,
            error="BG worker process exited without delivering a result",
        )

    def poll_result(self) -> BgAilsResult | None:
        """Nonblocking consume-once drain of at most one result."""
        if self._result_queue is None:
            return None
        try:
            result = self._result_queue.get_nowait()
        except queue.Empty:
            if (
                self._in_flight
                and self._process is not None
                and not self._process.is_alive()
            ):
                return self._worker_died_result()
            return None
        if not isinstance(result, BgAilsResult):
            LOGGER.error("AsyncBgAilsController: unexpected result type %r", type(result))
            return None
        self._in_flight = False
        return result

    def wait_result(self, *, poll_interval_s: float = 0.1) -> BgAilsResult | None:
        """Blocking consume-once drain: wait until the in-flight job delivers.

        Returns ``None`` immediately when nothing is in flight. Detects worker
        death (queue stays empty, process gone) and returns a ``worker_died``
        result instead of hanging.
        """
        if not self._in_flight:
            return None
        assert self._result_queue is not None
        while True:
            try:
                result = self._result_queue.get(timeout=poll_interval_s)
            except queue.Empty:
                if self._process is not None and not self._process.is_alive():
                    return self._worker_died_result()
                continue
            if not isinstance(result, BgAilsResult):
                LOGGER.error(
                    "AsyncBgAilsController: unexpected result type %r", type(result)
                )
                continue
            self._in_flight = False
            return result

    def shutdown(self, *, discard: bool = False) -> None:
        """Stop the worker.

        ``discard=True`` (wall-clock cap with a job in flight): SIGKILL the
        worker's process group so the AILS-II JVM child dies with it — the
        in-flight result is deliberately dropped (``discarded_at_cap``).
        Otherwise: sentinel + join, terminate only if the join times out.
        """
        if not self._started:
            return
        self.discarded_in_flight_at_shutdown = self._in_flight

        proc = self._process
        if discard and self._in_flight and proc is not None and proc.pid is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.terminate()
            proc.join(timeout=5.0)
        else:
            if self._job_queue is not None:
                try:
                    self._job_queue.put(_SHUTDOWN)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "AsyncBgAilsController: failed to send shutdown: %s", exc
                    )
            if proc is not None and proc.is_alive():
                proc.join(timeout=self.join_timeout_s)
                if proc.is_alive():
                    LOGGER.warning(
                        "AsyncBgAilsController: worker alive after %.1fs; terminating",
                        self.join_timeout_s,
                    )
                    proc.terminate()
                    proc.join(timeout=5.0)

        self._in_flight = False
        self._started = False
        LOGGER.info(
            "AsyncBgAilsController: shutdown complete; "
            "discarded_in_flight_at_shutdown=%s invocation_count=%d",
            self.discarded_in_flight_at_shutdown,
            self.invocation_count,
        )


__all__ = [
    "AsyncBgAilsController",
    "BgAilsJob",
    "BgAilsResult",
    "bg_ails_worker_main",
]
