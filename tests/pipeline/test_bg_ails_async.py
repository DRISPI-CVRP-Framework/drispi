"""Async BG-AILS: controller single-slot/consume-once semantics and the
pipeline handoff (register -> block/drain -> launch -> late apply) with a fake
in-process controller."""

from __future__ import annotations

import time
from multiprocessing import Queue
from pathlib import Path
from unittest.mock import patch

import pytest

from drispi.core.instance import CVRPInstance
from drispi.haos.tag import HAOSTag
from drispi.pipeline.bg_ails_async import AsyncBgAilsController, BgAilsResult
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.pipeline import DRISPIPipeline


def _dummy_tag(iteration: int = 0) -> HAOSTag:
    return HAOSTag(
        k=2,
        lambda_demand=0.0,
        paradigm="vertex",
        method="kmeans",
        solver="ails2",
        iteration=iteration,
    )


def _result(job_id: int = 1, launch_iteration: int = 0, **overrides) -> BgAilsResult:
    kwargs = dict(
        job_id=job_id,
        launch_iteration=launch_iteration,
        bg_seqs=[[2, 3], [4, 5]],
        bg_cost=10.0,
        cost_before=12.0,
        perturbed_route_indices=[0],
        op_tag=_dummy_tag(launch_iteration),
        reason="ok",
        perturb_wall_s=0.1,
        improve_wall_s=0.2,
        total_wall_s=0.3,
        ready_ts=time.time(),
        enqueue_ts=time.time() - 0.3,
    )
    kwargs.update(overrides)
    return BgAilsResult(**kwargs)


# ── Controller unit tests (no real worker process) ──────────────────────


def test_poll_result_is_consume_once() -> None:
    ctrl = AsyncBgAilsController(bg_cpus=None)
    ctrl._result_queue = Queue()
    ctrl._in_flight = True
    ctrl._result_queue.put(_result())
    time.sleep(0.05)  # let the queue feeder thread deliver

    first = ctrl.poll_result()
    assert first is not None and first.job_id == 1
    assert not ctrl.in_flight
    # Back-to-back drain must be a no-op.
    assert ctrl.poll_result() is None


def test_launch_raises_when_job_in_flight() -> None:
    ctrl = AsyncBgAilsController(bg_cpus=None)
    ctrl._started = True
    ctrl._job_queue = Queue()
    ctrl._in_flight = True
    with pytest.raises(RuntimeError, match="in flight"):
        ctrl.launch(
            launch_iteration=1,
            combined_seqs=[[2, 3]],
            partition=[[2, 3]],
            lambda_demand=0.0,
            angular_offset=0.0,
            initial_omega=0.8,
            time_limit=60.0,
            seed=1,
            boundary_threshold=0.5,
            small_cluster_cap=20,
            small_cluster_alpha=0.5,
            pair_selection="greedy",
            n_chains_mode="k",
            op_tag=_dummy_tag(1),
        )


def test_wait_result_returns_none_when_idle() -> None:
    ctrl = AsyncBgAilsController(bg_cpus=None)
    assert ctrl.wait_result() is None


# ── Pipeline integration with a fake in-process controller ──────────────


class _FakeBgCtrl:
    """Single-slot controller that 'finishes' each job instantly.

    ``launch`` synthesizes the result via ``improve_fn(combined_seqs)`` and
    parks it in the slot; the pipeline's next drain (iteration start or
    pre-launch block) consumes it — one-iteration lag like the real worker.
    """

    improve_fn = staticmethod(lambda seqs: [list(s) for s in seqs])

    def __init__(self, *, bg_cpus=None, xmx="4g", **_kw) -> None:
        self.bg_cpus = bg_cpus
        self.instance: CVRPInstance | None = None
        self.launched_jobs: list[dict] = []
        self._slot: BgAilsResult | None = None
        self._next_job_id = 1
        self.discarded_in_flight_at_shutdown = False
        self.shutdown_calls: list[bool] = []

    @property
    def in_flight(self) -> bool:
        return self._slot is not None

    def start(self, instance: CVRPInstance) -> None:
        self.instance = instance

    def launch(self, **kw) -> int:
        if self._slot is not None:
            raise RuntimeError("BG job already in flight")
        assert self.instance is not None
        self.launched_jobs.append(kw)
        bg_seqs = type(self).improve_fn(kw["combined_seqs"])
        bg_cost = float(sum(self.instance.route_cost(s) for s in bg_seqs))
        cost_before = float(
            sum(self.instance.route_cost(s) for s in kw["combined_seqs"])
        )
        self._slot = BgAilsResult(
            job_id=self._next_job_id,
            launch_iteration=kw["launch_iteration"],
            bg_seqs=bg_seqs,
            bg_cost=bg_cost,
            cost_before=cost_before,
            perturbed_route_indices=[],
            op_tag=kw["op_tag"],
            reason="ok",
            perturb_wall_s=0.0,
            improve_wall_s=0.0,
            total_wall_s=0.0,
            ready_ts=time.time(),
            enqueue_ts=time.time(),
        )
        self._next_job_id += 1
        return self._slot.job_id

    def poll_result(self) -> BgAilsResult | None:
        result, self._slot = self._slot, None
        return result

    def wait_result(self, **_kw) -> BgAilsResult | None:
        return self.poll_result()

    def shutdown(self, *, discard: bool = False) -> None:
        self.shutdown_calls.append(discard)
        self.discarded_in_flight_at_shutdown = self._slot is not None
        self._slot = None


def _fake_cluster_instance(instance, paradigm, method, k, routes=None, **kwargs):
    del paradigm, method, routes, kwargs
    customers = list(instance.customers)
    k = max(1, k)
    chunk = (len(customers) + k - 1) // k
    return [
        customers[i * chunk : min((i + 1) * chunk, len(customers))]
        for i in range(k)
        if customers[i * chunk : min((i + 1) * chunk, len(customers))]
    ]


def _fake_cluster_routes(instance, partition, *args, **kwargs):
    del args, kwargs
    return [[[c] for c in group] for group in partition], 1


def _async_cfg(tmp_path: Path, **overrides) -> DRISPIConfig:
    base = dict(
        time_limit=1e9,
        max_no_improve=1000,
        output_dir=tmp_path,
        haos_warmup=0,
        warmup_iterations=1000,
        sp_interval=1000,
        bg_ails_mode="async",
        cores_total=8,
        cores_dri=6,
        cores_bg=1,
        cores_sp=1,
        decomp_k_min_routes_per_cluster=0,
        decomp_k_min_arm_spacing=1,
    )
    base.update(overrides)
    return DRISPIConfig(**base)


def _run_async_iterations(
    instance_12: CVRPInstance, cfg: DRISPIConfig, n_iterations: int
) -> DRISPIPipeline:
    with patch("drispi.pipeline.pipeline.AsyncBgAilsController", _FakeBgCtrl):
        pipe = DRISPIPipeline(instance_12, cfg, bks_cost=None)
        pipe._start_time = time.perf_counter()
        with (
            patch(
                "drispi.pipeline.pipeline.cluster_instance",
                side_effect=_fake_cluster_instance,
            ),
            patch(
                "drispi.pipeline.pipeline.solve_subclusters_parallel",
                side_effect=_fake_cluster_routes,
            ),
            patch(
                "drispi.pipeline.pipeline.run_sp_sc",
                side_effect=lambda *a, **k: (None, False),
            ),
        ):
            for it in range(n_iterations):
                pipe._run_iteration(it)
    return pipe


def test_async_mode_launches_with_one_iteration_lag(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    cfg = _async_cfg(tmp_path)
    pipe = _run_async_iterations(instance_12, cfg, 3)
    ctrl = pipe._bg_ctrl
    assert isinstance(ctrl, _FakeBgCtrl)

    # One launch per iteration; BG(2) still parked in the slot (lag == 1).
    assert pipe._bg_launched == 3
    assert [j["launch_iteration"] for j in ctrl.launched_jobs] == [0, 1, 2]
    assert pipe._bg_applied == 2
    assert ctrl.in_flight

    # Applied BG routes carry the launch iteration's HAOS tag in the pool.
    applied_tag = ctrl.launched_jobs[0]["op_tag"]
    assert applied_tag.iteration == 0
    # Only the still-in-flight iteration remains pending in the registry.
    assert set(pipe._haos._pending_selections) == {2}


def test_async_apply_adopts_improvement_against_live_incumbent(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    cheap = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]

    class _Improving(_FakeBgCtrl):
        improve_fn = staticmethod(lambda seqs: [list(r) for r in cheap])

    cfg = _async_cfg(tmp_path)
    with patch("drispi.pipeline.pipeline.AsyncBgAilsController", _Improving):
        pipe = DRISPIPipeline(instance_12, cfg, bks_cost=None)
        pipe._start_time = time.perf_counter()
        with (
            patch(
                "drispi.pipeline.pipeline.cluster_instance",
                side_effect=_fake_cluster_instance,
            ),
            patch(
                "drispi.pipeline.pipeline.solve_subclusters_parallel",
                side_effect=_fake_cluster_routes,
            ),
            patch(
                "drispi.pipeline.pipeline.run_sp_sc",
                side_effect=lambda *a, **k: (None, False),
            ),
        ):
            pipe._run_iteration(0)
            best_after_0 = pipe._best_cost
            pipe._run_iteration(1)  # drains + applies BG(0)

    cheap_cost = float(sum(instance_12.route_cost(r) for r in cheap))
    # Iteration 0 adopted only the DR combined solution (BG not yet back).
    assert best_after_0 > cheap_cost
    assert pipe._best_cost == cheap_cost
    assert pipe._bg_adopted == 1
    # _last_cost tracks the most recent producer: iteration 1's DR candidate,
    # applied after the BG(0) drain (fakes make DR cost identical across
    # iterations, so it equals iteration 0's DR cost).
    assert pipe._last_cost == best_after_0


def test_async_launch_skipped_at_wall_clock_cap_clears_pending(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    # Remaining time (~5s) is far below the BG budget (min 60s) -> skip launch.
    cfg = _async_cfg(tmp_path, time_limit=5.0)
    pipe = _run_async_iterations(instance_12, cfg, 1)
    ctrl = pipe._bg_ctrl
    assert isinstance(ctrl, _FakeBgCtrl)

    assert pipe._bg_launched == 0
    assert pipe._bg_launch_skipped_cap == 1
    assert ctrl.launched_jobs == []
    # Pending selection for the skipped iteration was flushed with no credit.
    assert pipe._haos._pending_selections == {}


def test_async_finalize_discards_in_flight_and_clears_pending(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    cfg = _async_cfg(tmp_path)
    pipe = _run_async_iterations(instance_12, cfg, 1)
    ctrl = pipe._bg_ctrl
    assert isinstance(ctrl, _FakeBgCtrl)
    assert ctrl.in_flight

    pipe._iterations_completed = 1
    pipe._finalize()
    # Always-discard at cap: shutdown(discard=True) and pending flushed.
    assert ctrl.shutdown_calls == [True]
    assert pipe._haos._pending_selections == {}


def test_async_crash_clears_pending_and_fails_at_threshold(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    cfg = _async_cfg(tmp_path)
    with patch("drispi.pipeline.pipeline.AsyncBgAilsController", _FakeBgCtrl):
        pipe = DRISPIPipeline(instance_12, cfg, bks_cost=None)
    pipe._haos.register_selection(0, pipe._haos.select(0, pipe._rng))
    pipe._bg_inflight_iteration = 0

    failed = _result(
        launch_iteration=0,
        bg_seqs=None,
        bg_cost=None,
        reason="error:RuntimeError",
        error="boom",
    )
    with pytest.raises(RuntimeError, match="crashed"):
        pipe._handle_bg_result(failed, apply_iteration=1, drain_point="iteration_start")
    assert pipe._haos._pending_selections == {}
    assert pipe._bg_crash_count == 1


def test_sync_and_async_share_rng_stream_order(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    """The angular_offset pin: both modes draw HAOS selections, the offset, and
    the clustering seed at identical RNG stream positions (verified during
    warmup, before any apply divergence can shift HAOS weights)."""
    from drispi.core.solution import Route as SolutionRoute

    captured: dict[str, list] = {"sync": [], "async": []}

    def _capturing_cluster(arm):
        def inner(instance, paradigm, method, k, routes=None, **kwargs):
            captured[arm].append(
                (paradigm, method, k, kwargs["angular_offset"], kwargs["seed"])
            )
            return _fake_cluster_instance(instance, paradigm, method, k, routes)

        return inner

    def _fake_perturb(instance, solution, *args, **kwargs):
        del args, kwargs
        import numpy as np

        return solution, [], np.ones(instance.n_customers, dtype=np.float64)

    def _fake_improve(instance, perturbed, *args, **kwargs):
        del instance, args, kwargs
        return perturbed

    def _fake_dissim(instance, lambda_demand=0.0, angular_offset=0.0):
        del lambda_demand, angular_offset
        import numpy as np

        return np.zeros((instance.n_customers, instance.n_customers))

    for arm, mode in (("sync", "sync"), ("async", "async")):
        cfg = _async_cfg(tmp_path / arm, bg_ails_mode=mode, haos_warmup=1000)
        with patch("drispi.pipeline.pipeline.AsyncBgAilsController", _FakeBgCtrl):
            pipe = DRISPIPipeline(instance_12, cfg, bks_cost=None)
            pipe._start_time = time.perf_counter()
            with (
                patch(
                    "drispi.pipeline.pipeline.cluster_instance",
                    side_effect=_capturing_cluster(arm),
                ),
                patch(
                    "drispi.pipeline.pipeline.solve_subclusters_parallel",
                    side_effect=_fake_cluster_routes,
                ),
                patch(
                    "drispi.pipeline.pipeline.compute_dissimilarity_matrix",
                    side_effect=_fake_dissim,
                ),
                patch(
                    "drispi.pipeline.pipeline.run_bg_ails_perturb",
                    side_effect=_fake_perturb,
                ),
                patch(
                    "drispi.pipeline.pipeline.run_bg_ails_improve",
                    side_effect=_fake_improve,
                ),
                patch(
                    "drispi.pipeline.pipeline.run_sp_sc",
                    side_effect=lambda *a, **k: (None, False),
                ),
            ):
                for it in range(3):
                    pipe._run_iteration(it)

    assert captured["sync"] == captured["async"]


def test_invalid_bg_ails_mode_rejected(instance_12: CVRPInstance, tmp_path: Path) -> None:
    cfg = DRISPIConfig(output_dir=tmp_path, bg_ails_mode="bogus")
    with pytest.raises(ValueError, match="Invalid bg_ails_mode"):
        DRISPIPipeline(instance_12, cfg, bks_cost=None)
