from __future__ import annotations
import queue
import tkinter as tk
from typing import TYPE_CHECKING

from .events import GUIEvent, EventKind
from .panels.weights import WeightsPanel
from .panels.trajectory import TrajectoryPanel
from .panels.pool import PoolPanel
from .panels.iteration import IterationPanel
from .panels.solution import SolutionPanel

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance

POLL_MS  = 100    # queue polling interval in ms
BG_DARK  = "#111111"
BG_PANEL = "#1a1a1a"


class DRISPIVisualizer:
    """
    Main Tkinter window for the DRISPI run monitor.

    Window layout:
    ┌──────────────────────────────────────────────────────┐
    │  header: instance · iter · status · elapsed · best  │
    ├───────────────────┬──────────────────────────────────┤
    │  WeightsPanel     │  TrajectoryPanel                 │
    ├────────┬──────────┴──────────┬───────────────────────┤
    │  Pool  │  IterationPanel     │  SolutionPanel        │
    └────────┴─────────────────────┴───────────────────────┘

    Rendering contract:
    - Only Tkinter calls happen here (main thread).
    - Algorithm thread communicates exclusively via queue.Queue.
    - Window close hides the window but does not stop the algorithm.
    """

    def __init__(
        self,
        instance: CVRPInstance,
        event_queue: queue.Queue[GUIEvent],
    ) -> None:
        self._instance    = instance
        self._queue       = event_queue
        self._alive       = True   # False after DONE received

        self._root = tk.Tk()
        self._root.title("DRISPI run monitor")
        self._root.configure(bg=BG_DARK)
        self._root.minsize(900, 620)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._schedule_poll()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        root = self._root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=0)   # header
        root.rowconfigure(1, weight=0)   # row 1
        root.rowconfigure(2, weight=1)   # row 2

        self._build_header(root)

        # --- Row 1: weights + trajectory ---
        row1 = tk.Frame(root, bg=BG_DARK)
        row1.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        row1.columnconfigure(0, weight=1)
        row1.columnconfigure(1, weight=2)

        self._weights_panel    = WeightsPanel(row1)
        self._trajectory_panel = TrajectoryPanel(row1)

        self._weights_panel.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self._trajectory_panel.frame.grid(row=0, column=1, sticky="nsew")

        # --- Row 2: pool + iteration viz + best solution ---
        row2 = tk.Frame(root, bg=BG_DARK)
        row2.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=2)
        row2.columnconfigure(2, weight=2)
        row2.rowconfigure(0, weight=1)

        self._pool_panel      = PoolPanel(row2)
        self._iteration_panel = IterationPanel(row2, self._instance)
        self._solution_panel  = SolutionPanel(row2, self._instance)

        self._pool_panel.frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self._iteration_panel.frame.grid(row=0, column=1, sticky="nsew", padx=(0, 5))
        self._solution_panel.frame.grid(row=0, column=2, sticky="nsew")

    def _build_header(self, root: tk.Widget) -> None:
        hdr = tk.Frame(root, bg=BG_DARK)
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        hdr.columnconfigure(1, weight=1)

        tk.Label(
            hdr, text="DRISPI run monitor",
            font=("Helvetica", 13, "bold"), fg="#dddddd", bg=BG_DARK,
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            hdr, text=getattr(self._instance, "name", ""),
            font=("Helvetica", 11), fg="#666666", bg=BG_DARK,
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        right = tk.Frame(hdr, bg=BG_DARK)
        right.grid(row=0, column=2, sticky="e")

        self._lbl_iter   = tk.Label(right, text="Iter —",  font=("Helvetica", 11), fg="#666666", bg=BG_DARK)
        self._lbl_status = tk.Label(right, text="Idle",    font=("Helvetica", 10, "bold"), fg="#4caf50", bg=BG_DARK)
        self._lbl_best   = tk.Label(right, text="Best: —", font=("Helvetica", 11), fg="#666666", bg=BG_DARK)

        for lbl in (self._lbl_iter, self._lbl_status, self._lbl_best):
            lbl.pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _schedule_poll(self) -> None:
        self._root.after(POLL_MS, self._poll)

    def _poll(self) -> None:
        try:
            while True:
                self._dispatch(self._queue.get_nowait())
        except queue.Empty:
            pass
        if self._alive:
            self._schedule_poll()

    def _dispatch(self, event: GUIEvent) -> None:
        k, d = event.kind, event.data
        if   k == EventKind.ITER_START:     self._on_iter_start(d)
        elif k == EventKind.DECOMPOSED:     self._on_decomposed(d)
        elif k == EventKind.CLUSTER_SOLVED: self._on_cluster_solved(d)
        elif k == EventKind.LS1_DONE:       self._on_ls_done(d, phase=1)
        elif k == EventKind.SC_DONE:        self._on_sc_done(d)
        elif k == EventKind.LS2_DONE:       self._on_ls_done(d, phase=2)
        elif k == EventKind.BEST_UPDATED:   self._on_best_updated(d)
        elif k == EventKind.DONE:           self._on_done()

    # ------------------------------------------------------------------
    # Per-event handlers (delegate to panels)
    # ------------------------------------------------------------------

    def _on_iter_start(self, d: dict) -> None:
        self._lbl_iter.config(text=f"Iter {d['iter']}")
        self._lbl_status.config(text="Running", fg="#4caf50")
        self._weights_panel.update_weights(d["weights"])
        self._iteration_panel.reset()

    def _on_decomposed(self, d: dict) -> None:
        self._iteration_panel.show_decomposed(d["assignments"])

    def _on_cluster_solved(self, d: dict) -> None:
        self._iteration_panel.show_cluster_solved(
            d["cluster_id"], d["n_clusters"], d["routes"]
        )

    def _on_ls_done(self, d: dict, phase: int) -> None:
        self._iteration_panel.show_ls_done(d["modified_customers"], phase)

    def _on_sc_done(self, d: dict) -> None:
        self._iteration_panel.show_sc_done(d["routes"])
        if d.get("pool_stats"):
            self._pool_panel.update(d["pool_stats"])

    def _on_best_updated(self, d: dict) -> None:
        self._lbl_best.config(text=f"Best: {d['cost']:,.0f}")
        self._trajectory_panel.add_point(d["cost"])
        self._solution_panel.show_solution(d["routes"], d["cost"])

    def _on_done(self) -> None:
        self._lbl_status.config(text="Done", fg="#2196f3")
        self._alive = False

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        """Hide the window; algorithm continues unaffected."""
        self._root.withdraw()

    def mainloop(self) -> None:
        self._root.mainloop()
