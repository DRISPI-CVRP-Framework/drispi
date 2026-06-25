"""Phase snapshot writer for the real-time dashboard (background thread)."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from drispi.core.instance import CVRPInstance
from drispi.core.types import Route
from drispi.utils.time import local_now

_SENTINEL: object = object()


def _instance_coords(instance: CVRPInstance) -> tuple[list[int], list[list[float]], list[float]]:
    customer_ids = list(instance.customers)
    customers = [
        [float(instance.coordinates[cid][0]), float(instance.coordinates[cid][1])]
        for cid in customer_ids
    ]
    depot = [float(instance.depot[0]), float(instance.depot[1])]
    return customer_ids, customers, depot


class SnapshotWriter:
    """
    Writes phase snapshot files to disk from a background thread.
    All write operations are non-blocking from the pipeline's perspective.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._snapshots_dir = run_dir / "snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[str, Any] | object] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def write_phase(
        self,
        iteration: int,
        phase_num: int,
        phase_name: str,
        is_spsc_iter: bool,
        total_phases: int,
        instance: CVRPInstance,
        *,
        cluster_assignments: dict[int, int] | None = None,
        routes: list[Route] | None = None,
        route_cluster_ids: list[int] | None = None,
        perturbed_route_indices: list[int] | None = None,
        changed_route_indices: list[int] | None = None,
        selected_route_indices: list[int] | None = None,
        phase_tag: str | None = None,
    ) -> None:
        """Enqueue a snapshot for background writing. Returns immediately."""
        customer_ids, customers, depot = _instance_coords(instance)
        cluster_json: dict[str, int] | None = None
        if cluster_assignments is not None:
            cluster_json = {str(k): v for k, v in cluster_assignments.items()}

        payload: dict[str, Any] = {
            "iteration": iteration,
            "phase_num": phase_num,
            "phase_name": phase_name,
            "is_spsc_iter": is_spsc_iter,
            "total_phases": total_phases,
            "timestamp": local_now().strftime("%H:%M:%S"),
            "depot": depot,
            "customer_ids": customer_ids,
            "customers": customers,
            "cluster_assignments": cluster_json,
            "routes": [list(r) for r in routes] if routes is not None else None,
            "route_cluster_ids": route_cluster_ids,
            "perturbed_route_indices": perturbed_route_indices,
            "changed_route_indices": changed_route_indices,
            "selected_route_indices": selected_route_indices,
        }
        if phase_tag is not None:
            payload["bg_stage"] = phase_tag
        self._queue.put(("phase", (iteration, phase_num, phase_tag, payload)))

    def write_best_solution(
        self,
        routes: list[Route],
        cost: float,
        iteration: int,
        phase_name: str,
        instance: CVRPInstance,
    ) -> None:
        """Enqueue best_solution.json for background writing."""
        customer_ids, customers, depot = _instance_coords(instance)
        payload: dict[str, Any] = {
            "cost": cost,
            "iteration_found": iteration,
            "phase_found": phase_name,
            "depot": depot,
            "customer_ids": customer_ids,
            "customers": customers,
            "routes": [list(r) for r in routes],
        }
        self._queue.put(("best", payload))

    def cleanup_old_snapshots(self, current_iteration: int) -> None:
        """Delete snapshot files from iterations <= current_iteration - 5."""
        self._queue.put(("cleanup", current_iteration))

    def stop(self) -> None:
        """Flush the queue and stop the background thread gracefully."""
        self._queue.put(_SENTINEL)
        self._thread.join()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            if not isinstance(item, tuple):
                continue
            kind = item[0]
            if kind == "phase":
                _, args = item
                iteration, phase_num, phase_tag, payload = args
                self._write_phase_atomic(iteration, phase_num, phase_tag, payload)
            elif kind == "best":
                _, payload = item
                self._write_best_atomic(payload)
            elif kind == "cleanup":
                _, current_iteration = item
                self._do_cleanup(current_iteration)

    def _write_phase_atomic(
        self,
        iteration: int,
        phase_num: int,
        phase_tag: str | None,
        payload: dict[str, Any],
    ) -> None:
        if phase_tag:
            final_path = self._snapshots_dir / f"iter_{iteration}_phase_{phase_num}_{phase_tag}.json"
        else:
            final_path = self._snapshots_dir / f"iter_{iteration}_phase_{phase_num}.json"
        tmp_path = final_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp_path.rename(final_path)

    def _write_best_atomic(self, payload: dict[str, Any]) -> None:
        final_path = self._run_dir / "best_solution.json"
        tmp_path = final_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp_path.rename(final_path)

    def _do_cleanup(self, current_iteration: int) -> None:
        cutoff = current_iteration - 5
        if cutoff < 0:
            return
        for path in self._snapshots_dir.glob("iter_*_phase_*.json"):
            parts = path.stem.split("_")
            if len(parts) < 2:
                continue
            try:
                iter_num = int(parts[1])
            except ValueError:
                continue
            if iter_num <= cutoff:
                path.unlink(missing_ok=True)
