"""Long-lived async SC/SP worker process and main-side controller.

The worker owns Gurobi ``build_and_solve`` and post-SP/SC
``run_standard_improvement`` on the reserved SP CPU set. The main process
only pickles RoutePool snapshots to bytes (sync), enqueues jobs, and drains
results at iteration start.

See the async SC/SP design: start-anchored triggers, ``overlap_policy``
``skip`` | ``queue_latest``, and ``mip_no_solution`` (no stale-best fallback).
"""

from __future__ import annotations

import logging
import os
import pickle
import queue
import subprocess
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Literal

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route as SolutionRoute
from drispi.core.types import Route
from drispi.improvement.bg_ails import run_standard_improvement
from drispi.pipeline.cores import affinity_logging_enabled, set_affinity
from drispi.route_pool.pool import RoutePool
from drispi.solvers.ails2 import Ails2Solver
from drispi.sp.deduplicate import remove_duplicates
from drispi.sp.model import SolveMetrics, build_and_solve
from drispi.sp.policy import should_run_sp_sc, should_use_sp

LOGGER = logging.getLogger(__name__)

_SHUTDOWN: Any = None

OverlapPolicy = Literal["skip", "queue_latest"]
TriggerMode = Literal["iteration", "wallclock"]


@dataclass(frozen=True)
class SpScJob:
    """One SC/SP (+ optional standard improvement) job for the SP worker.

    ``pool_bytes`` must already be a ``pickle.dumps`` snapshot — never put a
    live ``RoutePool`` on the queue (feeder-thread pickling would race).
    """

    job_id: int
    pool_bytes: bytes
    best_solution: list[Route]
    best_cost_at_snapshot: float
    instance: CVRPInstance
    time_limit: float
    mip_gap: float
    min_coverage: int
    std_improve_limit: float
    seed: int
    enqueue_ts: float


@dataclass
class SpScResult:
    """Worker result drained by the main process at iteration start."""

    job_id: int
    used_sp: bool
    sp_routes: list[Route] | None
    final_routes: list[Route] | None
    final_cost: float | None
    lp_weights: dict[frozenset[int], float] | None
    metrics: dict[str, Any]
    snapshot_best_cost: float
    reason: str
    ready_ts: float
    enqueue_ts: float | None = None


@dataclass
class TriggerOutcome:
    """Return value of :meth:`AsyncSpScController.maybe_trigger`."""

    action: Literal[
        "enqueued",
        "queued_latest",
        "skipped_busy",
        "not_due",
        "not_started",
    ]
    job_id: int | None = None
    serialize_s: float | None = None
    payload_bytes: int | None = None
    enqueue_ts: float | None = None


def _read_cpus_allowed_list(pid: int) -> str | None:
    """Return ``Cpus_allowed_list`` from ``/proc/<pid>/status``, if available."""
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return None


def _probe_java_child_affinity(*, ails_apc: int, xmx: str) -> None:
    """Spawn a short-lived ``java … -version`` and log its CPU affinity once."""
    cmd = [
        "java",
        "-XX:+UseSerialGC",
        f"-XX:ActiveProcessorCount={int(ails_apc)}",
        f"-Xmx{xmx}",
        "-version",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        LOGGER.warning("java-child-affinity-probe: failed to spawn java: %s", exc)
        return
    # Give the JVM a moment to start so /proc is populated.
    time.sleep(0.05)
    allowed = _read_cpus_allowed_list(proc.pid)
    try:
        taskset = subprocess.run(
            ["taskset", "-p", str(proc.pid)],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        taskset_out = (taskset.stdout or taskset.stderr or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        taskset_out = ""
    LOGGER.info(
        "java-child-affinity-probe: pid=%s Cpus_allowed_list=%s taskset=%s",
        proc.pid,
        allowed,
        taskset_out or "(unavailable)",
    )
    try:
        proc.communicate(timeout=30.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5.0)


def _metrics_dict(
    metrics: SolveMetrics | None,
    *,
    ails_wall_s: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if metrics is not None:
        out["lp_wall_s"] = metrics.lp_wall_s
        out["mip_wall_s"] = metrics.mip_wall_s
        out["total_wall_s"] = metrics.total_wall_s
        out["mip_gap"] = metrics.mip_gap
        out["objective"] = metrics.objective
        out["timed_out"] = metrics.timed_out
        out["sol_count"] = metrics.sol_count
    if ails_wall_s is not None:
        out["ails_wall_s"] = ails_wall_s
    return out


def _seqs_to_solution_routes(
    instance: CVRPInstance, seqs: list[Route]
) -> list[SolutionRoute]:
    return [SolutionRoute(customers=list(s), cost=instance.route_cost(s)) for s in seqs]


def _solution_routes_to_seqs(routes: list[SolutionRoute]) -> list[Route]:
    return [list(r.customers) for r in routes]


def _empty_result(
    job: SpScJob,
    *,
    used_sp: bool,
    reason: str,
    lp_weights: dict[frozenset[int], float] | None = None,
    metrics: dict[str, Any] | None = None,
) -> SpScResult:
    return SpScResult(
        job_id=job.job_id,
        used_sp=used_sp,
        sp_routes=None,
        final_routes=None,
        final_cost=None,
        lp_weights=lp_weights,
        metrics=metrics or {},
        snapshot_best_cost=job.best_cost_at_snapshot,
        reason=reason,
        ready_ts=time.time(),
        enqueue_ts=job.enqueue_ts,
    )


def _process_job(
    job: SpScJob,
    *,
    gurobi_threads: int,
    ails_apc: int,
    xmx: str,
) -> SpScResult:
    """Unpickle pool, solve SC/SP, optionally run standard improvement."""
    pool = pickle.loads(job.pool_bytes)
    if not isinstance(pool, RoutePool):
        raise TypeError(f"Expected RoutePool in job payload, got {type(pool)!r}")

    instance = job.instance
    # Materialize distance matrix in the worker before Gurobi / AILS.
    _ = instance.distance_matrix

    use_sp = should_use_sp(pool, instance, job.min_coverage)
    lp_weights, raw_solution, timed_out, sol_count, metrics = build_and_solve(
        pool,
        instance,
        use_sp,
        time_limit=job.time_limit,
        mip_gap=job.mip_gap,
        threads=gurobi_threads,
    )

    if timed_out and sol_count == 0:
        return _empty_result(
            job,
            used_sp=use_sp,
            reason="mip_no_solution",
            lp_weights=lp_weights,
            metrics=_metrics_dict(metrics),
        )

    base: list[Route] = [list(r) for r in raw_solution]
    if use_sp:
        sp_routes = base
    else:
        sp_routes = remove_duplicates(base, instance)

    if not sp_routes:
        return _empty_result(
            job,
            used_sp=use_sp,
            reason="empty_sp_routes",
            lp_weights=lp_weights,
            metrics=_metrics_dict(metrics),
        )

    solver = Ails2Solver(active_processor_count=ails_apc, xmx=xmx)
    sp_input = _seqs_to_solution_routes(instance, sp_routes)
    t_ails0 = time.perf_counter()
    final_sol = run_standard_improvement(
        instance,
        sp_input,
        job.std_improve_limit,
        seed=job.seed,
        solver=solver,
    )
    ails_wall_s = time.perf_counter() - t_ails0
    final_seqs = _solution_routes_to_seqs(final_sol)
    final_cost = float(sum(instance.route_cost(r) for r in final_seqs))

    return SpScResult(
        job_id=job.job_id,
        used_sp=use_sp,
        sp_routes=sp_routes,
        final_routes=final_seqs,
        final_cost=final_cost,
        lp_weights=lp_weights,
        metrics=_metrics_dict(metrics, ails_wall_s=ails_wall_s),
        snapshot_best_cost=job.best_cost_at_snapshot,
        reason="ok",
        ready_ts=time.time(),
        enqueue_ts=job.enqueue_ts,
    )


def sp_sc_worker_main(
    job_queue: Queue,
    result_queue: Queue,
    sp_cpus: list[int] | None,
    gurobi_threads: int,
    ails_apc: int,
    xmx: str = "4g",
) -> None:
    """
    Long-lived SP worker target.

    Pins to ``sp_cpus`` (when provided), probes java-child affinity once, then
    loops on ``job_queue`` until a shutdown sentinel (``None``).
    """
    if sp_cpus:
        set_affinity(list(sp_cpus), label=f"sp-worker pid={os.getpid()}")
    elif affinity_logging_enabled():
        LOGGER.info("sp-worker: no sp_cpus provided; leaving OS affinity unchanged")

    if affinity_logging_enabled():
        _probe_java_child_affinity(ails_apc=ails_apc, xmx=xmx)

    while True:
        job = job_queue.get()
        if job is _SHUTDOWN:
            LOGGER.info("sp-worker: shutdown sentinel received")
            break
        if not isinstance(job, SpScJob):
            LOGGER.error("sp-worker: unexpected job type %r; skipping", type(job))
            continue
        try:
            result = _process_job(
                job,
                gurobi_threads=gurobi_threads,
                ails_apc=ails_apc,
                xmx=xmx,
            )
        except Exception as exc:  # noqa: BLE001 — deliver failure to main
            LOGGER.exception("sp-worker: job_id=%s failed: %s", job.job_id, exc)
            result = _empty_result(
                job,
                used_sp=False,
                reason=f"error:{type(exc).__name__}",
                metrics={"error": str(exc)},
            )
        result_queue.put(result)


class AsyncSpScController:
    """Main-side controller for the async SC/SP worker.

    Pipeline API (call sites to wire later in ``pipeline.py``):

    * ``start()`` — spawn the daemon worker once.
    * ``maybe_trigger(...)`` — start-anchored enqueue with skip / queue_latest.
    * ``poll_result(nonblocking=True)`` — drain at iteration start only.
    * ``shutdown()`` — sentinel + join + optional terminate; sets
      ``discarded_in_flight_at_shutdown``.
    """

    def __init__(
        self,
        *,
        sp_cpus: list[int] | None,
        gurobi_threads: int,
        ails_apc: int,
        xmx: str = "4g",
        overlap_policy: OverlapPolicy = "skip",
        trigger: TriggerMode = "wallclock",
        interval_minutes: float = 20.0,
        warmup_iterations: int = 10,
        sp_interval: int = 3,
        join_timeout_s: float = 30.0,
    ) -> None:
        if overlap_policy not in ("skip", "queue_latest"):
            raise ValueError(f"Invalid overlap_policy: {overlap_policy!r}")
        if trigger not in ("iteration", "wallclock"):
            raise ValueError(f"Invalid trigger: {trigger!r}")

        self.sp_cpus = list(sp_cpus) if sp_cpus is not None else None
        self.gurobi_threads = int(gurobi_threads)
        self.ails_apc = int(ails_apc)
        self.xmx = str(xmx)
        self.overlap_policy: OverlapPolicy = overlap_policy
        self.trigger: TriggerMode = trigger
        self.interval_minutes = float(interval_minutes)
        self.warmup_iterations = int(warmup_iterations)
        self.sp_interval = int(sp_interval)
        self.join_timeout_s = float(join_timeout_s)

        self._job_queue: Queue | None = None
        self._result_queue: Queue | None = None
        self._process: Process | None = None

        self._next_job_id = 1
        self._in_flight = False
        self._pending: SpScJob | None = None
        self._next_due_ts: float | None = None  # wallclock; None => due immediately
        self._started = False

        # Instrumentation (read by pipeline / logger)
        self.skipped_trigger_count = 0
        self.discarded_in_flight_at_shutdown = False
        self.invocation_count = 0
        self.last_serialize_s: float | None = None
        self.last_payload_bytes: int | None = None
        self.last_enqueue_ts: float | None = None
        self._busy_started_ts: float | None = None
        self._busy_accumulated_s = 0.0
        self._run_started_ts: float | None = None

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def sp_busy_fraction(self, *, now: float | None = None) -> float | None:
        """Fraction of wall time since ``start()`` that a job was in flight."""
        if self._run_started_ts is None:
            return None
        t = time.time() if now is None else now
        elapsed = max(t - self._run_started_ts, 1e-9)
        busy = self._busy_accumulated_s
        if self._in_flight and self._busy_started_ts is not None:
            busy += t - self._busy_started_ts
        return busy / elapsed

    def start(self) -> None:
        """Spawn the long-lived daemon worker process."""
        if self._started:
            return
        self._job_queue = Queue()
        self._result_queue = Queue()
        self._process = Process(
            target=sp_sc_worker_main,
            args=(
                self._job_queue,
                self._result_queue,
                self.sp_cpus,
                self.gurobi_threads,
                self.ails_apc,
                self.xmx,
            ),
            name="drispi-sp-sc-worker",
            daemon=True,
        )
        self._process.start()
        self._started = True
        self._run_started_ts = time.time()
        LOGGER.info(
            "AsyncSpScController: started worker pid=%s sp_cpus=%s "
            "gurobi_threads=%s ails_apc=%s xmx=%s",
            self._process.pid,
            self.sp_cpus,
            self.gurobi_threads,
            self.ails_apc,
            self.xmx,
        )

    def _is_due(self, *, iteration: int | None, now: float) -> bool:
        if self.trigger == "iteration":
            if iteration is None:
                raise ValueError("iteration is required when trigger='iteration'")
            return should_run_sp_sc(
                iteration, self.warmup_iterations, self.sp_interval
            )
        # wallclock, start-anchored
        if self._next_due_ts is None:
            return True
        return now >= self._next_due_ts

    def _advance_due(self, enqueue_ts: float) -> None:
        if self.trigger == "wallclock":
            self._next_due_ts = enqueue_ts + self.interval_minutes * 60.0

    def _put_job(self, job: SpScJob) -> None:
        assert self._job_queue is not None
        self._job_queue.put(job)
        self._in_flight = True
        self._busy_started_ts = time.time()
        self.invocation_count += 1
        self.last_enqueue_ts = job.enqueue_ts

    def _build_job(
        self,
        *,
        pool: RoutePool,
        best_solution: list[Route],
        best_cost: float,
        instance: CVRPInstance,
        time_limit: float,
        mip_gap: float,
        min_coverage: int,
        std_improve_limit: float,
        seed: int,
        now: float,
    ) -> tuple[SpScJob, float, int]:
        t0 = time.perf_counter()
        pool_bytes = pickle.dumps(pool, protocol=pickle.HIGHEST_PROTOCOL)
        serialize_s = time.perf_counter() - t0
        payload_bytes = len(pool_bytes)
        self.last_serialize_s = serialize_s
        self.last_payload_bytes = payload_bytes

        job_id = self._next_job_id
        self._next_job_id += 1
        job = SpScJob(
            job_id=job_id,
            pool_bytes=pool_bytes,
            best_solution=[list(r) for r in best_solution],
            best_cost_at_snapshot=float(best_cost),
            instance=instance,
            time_limit=float(time_limit),
            mip_gap=float(mip_gap),
            min_coverage=int(min_coverage),
            std_improve_limit=float(std_improve_limit),
            seed=int(seed),
            enqueue_ts=now,
        )
        return job, serialize_s, payload_bytes

    def maybe_trigger(
        self,
        *,
        pool: RoutePool,
        best_solution: list[Route],
        best_cost: float,
        instance: CVRPInstance,
        time_limit: float,
        mip_gap: float,
        min_coverage: int,
        std_improve_limit: float,
        seed: int,
        iteration: int | None = None,
        now: float | None = None,
    ) -> TriggerOutcome:
        """
        Possibly enqueue a pool snapshot for the SP worker.

        Start-anchored: after an enqueue (or a busy skip under wallclock), the
        next wall-clock due time is ``enqueue_ts + interval_minutes``.

        Overlap: ``skip`` increments ``skipped_trigger_count``; ``queue_latest``
        keeps at most one pending job released when the in-flight result is polled.
        """
        if not self._started:
            return TriggerOutcome(action="not_started")

        t = time.time() if now is None else now
        if not self._is_due(iteration=iteration, now=t):
            return TriggerOutcome(action="not_due")

        job, serialize_s, payload_bytes = self._build_job(
            pool=pool,
            best_solution=best_solution,
            best_cost=best_cost,
            instance=instance,
            time_limit=time_limit,
            mip_gap=mip_gap,
            min_coverage=min_coverage,
            std_improve_limit=std_improve_limit,
            seed=seed,
            now=t,
        )

        if not self._in_flight:
            self._put_job(job)
            self._advance_due(t)
            return TriggerOutcome(
                action="enqueued",
                job_id=job.job_id,
                serialize_s=serialize_s,
                payload_bytes=payload_bytes,
                enqueue_ts=t,
            )

        # Busy: overlap policy
        self._advance_due(t)
        if self.overlap_policy == "skip":
            self.skipped_trigger_count += 1
            LOGGER.info(
                "AsyncSpScController: skipped trigger (busy); "
                "skipped_trigger_count=%d job_would_be=%d",
                self.skipped_trigger_count,
                job.job_id,
            )
            return TriggerOutcome(
                action="skipped_busy",
                job_id=job.job_id,
                serialize_s=serialize_s,
                payload_bytes=payload_bytes,
                enqueue_ts=t,
            )

        # queue_latest: replace any pending blob
        self._pending = job
        LOGGER.info(
            "AsyncSpScController: queued_latest job_id=%d (in-flight still busy)",
            job.job_id,
        )
        return TriggerOutcome(
            action="queued_latest",
            job_id=job.job_id,
            serialize_s=serialize_s,
            payload_bytes=payload_bytes,
            enqueue_ts=t,
        )

    def _mark_idle(self) -> None:
        if self._busy_started_ts is not None:
            self._busy_accumulated_s += time.time() - self._busy_started_ts
        self._busy_started_ts = None
        self._in_flight = False

    def _release_pending(self) -> None:
        if self._pending is None:
            return
        pending = self._pending
        self._pending = None
        self._put_job(pending)
        LOGGER.info(
            "AsyncSpScController: released pending job_id=%d after drain",
            pending.job_id,
        )

    def poll_result(self, *, nonblocking: bool = True) -> SpScResult | None:
        """Fetch one result from the worker. Non-blocking by default."""
        if self._result_queue is None:
            return None
        try:
            if nonblocking:
                result = self._result_queue.get_nowait()
            else:
                result = self._result_queue.get()
        except queue.Empty:
            return None

        if not isinstance(result, SpScResult):
            LOGGER.error(
                "AsyncSpScController: unexpected result type %r", type(result)
            )
            return None

        self._mark_idle()
        self._release_pending()
        return result

    def shutdown(self) -> None:
        """Send shutdown sentinel, join with timeout, terminate if needed."""
        if not self._started:
            return

        discarded = self._in_flight or self._pending is not None
        self.discarded_in_flight_at_shutdown = discarded
        # Structural hook for a future final-flush trigger (not implemented).
        self._pending = None

        if self._job_queue is not None:
            try:
                self._job_queue.put(_SHUTDOWN)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("AsyncSpScController: failed to send shutdown: %s", exc)

        proc = self._process
        if proc is not None and proc.is_alive():
            proc.join(timeout=self.join_timeout_s)
            if proc.is_alive():
                LOGGER.warning(
                    "AsyncSpScController: worker still alive after %.1fs; terminating",
                    self.join_timeout_s,
                )
                proc.terminate()
                proc.join(timeout=5.0)

        if self._in_flight:
            self._mark_idle()

        self._started = False
        LOGGER.info(
            "AsyncSpScController: shutdown complete; "
            "discarded_in_flight_at_shutdown=%s skipped_trigger_count=%d "
            "invocation_count=%d",
            self.discarded_in_flight_at_shutdown,
            self.skipped_trigger_count,
            self.invocation_count,
        )


__all__ = [
    "AsyncSpScController",
    "SpScJob",
    "SpScResult",
    "TriggerOutcome",
    "sp_sc_worker_main",
]
