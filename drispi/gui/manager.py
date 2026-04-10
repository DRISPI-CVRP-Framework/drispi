from __future__ import annotations
import queue
import threading
from typing import TYPE_CHECKING

from .events import GUIEvent, EventKind

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance


class GUIManager:
    """
    Algorithm-facing interface to the visualizer. Thread-safe.

    All on_*() methods may be called from any thread. The Tkinter
    window runs on whichever thread calls mainloop() — in practice
    that should always be the main thread.

    Typical entry-point wiring:

        gui = GUIManager(instance, enabled=args.gui)
        worker = threading.Thread(target=run, args=(instance, gui), daemon=True)
        worker.start()
        gui.mainloop()          # blocks main thread; returns when DONE received

    When enabled=False every method is a no-op, so the algorithm code
    needs no conditional guards.
    """

    def __init__(self, instance: CVRPInstance, enabled: bool = True) -> None:
        self._enabled = enabled
        self._instance = instance
        self._queue: queue.Queue[GUIEvent] = queue.Queue()

    # ------------------------------------------------------------------
    # Main-thread interface
    # ------------------------------------------------------------------

    def mainloop(self) -> None:
        """Open the Tkinter window and block until the run finishes or
        the user closes the window."""
        if not self._enabled:
            return
        from .window import DRISPIVisualizer
        viz = DRISPIVisualizer(self._instance, self._queue)
        viz.mainloop()

    # ------------------------------------------------------------------
    # Algorithm-thread interface
    # ------------------------------------------------------------------

    def post(self, event: GUIEvent) -> None:
        """Low-level enqueue. Prefer the typed on_*() helpers below."""
        if self._enabled:
            self._queue.put_nowait(event)

    def on_iter_start(
        self,
        iter_n: int,
        n_clusters: int,
        clustering: str,
        solver: str,
        weights: dict[str, float],
    ) -> None:
        """Call at the very beginning of each main loop iteration.

        weights: mapping of operator key → current weight,
                 e.g. {"kmeans+filo": 0.31, "angular+filo2": 0.24, ...}
        """
        self.post(GUIEvent(EventKind.ITER_START, {
            "iter":       iter_n,
            "n_clusters": n_clusters,
            "clustering": clustering,
            "solver":     solver,
            "weights":    weights,
        }))

    def on_decomposed(self, assignments: list[int]) -> None:
        """Call after clustering is complete.

        assignments[i] = cluster_id for customer i (0-indexed, depot excluded).
        Length must equal instance.n_customers.
        """
        self.post(GUIEvent(EventKind.DECOMPOSED, {
            "assignments": assignments,
        }))

    def on_cluster_solved(
        self,
        cluster_id: int,
        n_clusters: int,
        routes: list[list[int]],
    ) -> None:
        """Call each time a single cluster's routing is finished.

        routes: list of routes for this cluster only; each route is a
                list of customer indices (0-indexed, depot excluded).
        """
        self.post(GUIEvent(EventKind.CLUSTER_SOLVED, {
            "cluster_id": cluster_id,
            "n_clusters": n_clusters,
            "routes":     routes,
        }))

    def on_ls1_done(self, modified_customers: set[int]) -> None:
        """Call after the first local-search / BG-AILS phase.

        modified_customers: union of all customer indices that appeared
        in any route that was changed by an accepted LS move.
        """
        self.post(GUIEvent(EventKind.LS1_DONE, {
            "modified_customers": modified_customers,
        }))

    def on_sc_done(
        self,
        routes: list[list[int]],
        cost: float,
        pool_stats: dict | None = None,
    ) -> None:
        """Call after set covering / set partitioning produces a solution.

        pool_stats (optional): dict with keys matching PoolPanel.STAT_KEYS,
        e.g. {"pool_size": 14832, "unique_routes": 12401, ...}
        """
        self.post(GUIEvent(EventKind.SC_DONE, {
            "routes":     routes,
            "cost":       cost,
            "pool_stats": pool_stats or {},
        }))

    def on_ls2_done(self, modified_customers: set[int]) -> None:
        """Call after the second local-search phase (post-SC improvement)."""
        self.post(GUIEvent(EventKind.LS2_DONE, {
            "modified_customers": modified_customers,
        }))

    def on_best_updated(self, cost: float, routes: list[list[int]]) -> None:
        """Call whenever a new global best solution is found.

        Triggers: trajectory dot, best-solution panel redraw.
        """
        self.post(GUIEvent(EventKind.BEST_UPDATED, {
            "cost":   cost,
            "routes": routes,
        }))

    def on_done(self) -> None:
        """Call once when the algorithm finishes (time limit or convergence)."""
        self.post(GUIEvent(EventKind.DONE, {}))
