from __future__ import annotations
import tkinter as tk

BG        = "#1a1a1a"
CANVAS_BG = "#080808"
DOT_R     = 3    # dot radius in px
BSF_COLOR = "#5499E0"


class TrajectoryPanel:
    """
    Scatter plot of solution costs at trajectory events.

    Points are added only on BEST_UPDATED (post-LS1, post-SC, post-LS2).
    The best-so-far step line is extended incrementally — no full redraw.

    Color encoding:
        dot color = improvement magnitude relative to previous best
        yellow (#EF9F27) = large improvement
        green  (#1D9E75) = small / no improvement
        (interpolate between the two based on Δcost / initial_cost)

    Rendering is fully additive: dots and BSF-line segments are only
    ever *added*, never removed. Canvas item count stays low throughout
    a 10-hour run (one dot + one line segment per BEST_UPDATED event).
    """

    PAD = {"l": 40, "r": 12, "t": 8, "b": 20}

    def __init__(self, parent: tk.Widget) -> None:
        self.frame = tk.Frame(parent, bg=BG, bd=0)

        header = tk.Frame(self.frame, bg=BG)
        header.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(header, text="Search trajectory", font=("Helvetica", 10),
                 fg="#666666", bg=BG, anchor="w").pack(side="left")
        tk.Label(header, text="cost / event",       font=("Helvetica", 9),
                 fg="#444444", bg=BG).pack(side="right")

        self._canvas = tk.Canvas(
            self.frame, bg=CANVAS_BG, bd=0, highlightthickness=0, height=170,
        )
        self._canvas.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._canvas.bind("<Configure>", self._on_resize)

        # state
        self._points:      list[float] = []
        self._best_so_far: float       = float("inf")
        self._prev_best:   float       = float("inf")
        self._cost_min:    float       = float("inf")
        self._cost_max:    float       = float("-inf")
        self._bsf_end_x:   float | None = None  # x pixel of last BSF update
        self._bsf_end_y:   float | None = None
        self._canvas_w:    int = 1
        self._canvas_h:    int = 1

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_point(self, cost: float) -> None:
        """
        Add one trajectory point and extend the best-so-far line.

        Called from window._on_best_updated() on the main thread.
        """
        self._prev_best = self._best_so_far
        self._points.append(cost)

        if cost < self._cost_min:
            self._cost_min = cost
        if cost > self._cost_max:
            self._cost_max = cost
        if cost < self._best_so_far:
            self._best_so_far = cost

        # TODO: implement incremental draw
        #
        # Steps:
        #   1. Map (point_index, cost) → (px, py) using _to_canvas()
        #   2. Compute dot color from improvement magnitude:
        #        delta = max(0, self._prev_best - cost)
        #        rel   = delta / max(1, self._cost_max - self._cost_min)
        #        color = interpolate amber→teal by rel
        #   3. self._canvas.create_oval(px-DOT_R, py-DOT_R, px+DOT_R, py+DOT_R, ...)
        #   4. If BSF improved, extend blue step line from (_bsf_end_x, _bsf_end_y)
        #      to (px, bsf_py) then (px, bsf_py) — store new endpoint
        #   5. On first call: draw axis labels (min/max cost, event count)
        #      These only need drawing once; track with a flag.
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_resize(self, event: tk.Event) -> None:
        self._canvas_w = event.width
        self._canvas_h = event.height
        # TODO: on resize, full redraw from self._points
        #       (rare event, full redraw acceptable)

    def _to_canvas(self, index: int, cost: float) -> tuple[float, float]:
        """Map (event_index, cost) → canvas (px, py)."""
        p = self.PAD
        cw = self._canvas_w - p["l"] - p["r"]
        ch = self._canvas_h - p["t"] - p["b"]
        n  = max(1, len(self._points) - 1)
        span = max(1.0, self._cost_max - self._cost_min)
        px = p["l"] + index / n * cw
        py = p["t"] + ch - (cost - self._cost_min) / span * ch
        return px, py
