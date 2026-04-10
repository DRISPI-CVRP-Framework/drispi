from __future__ import annotations
import tkinter as tk

BG = "#1a1a1a"

# One color per operator slot (up to 8 combinations)
OPERATOR_COLORS = [
    "#378ADD", "#639922", "#EF9F27",
    "#D85A30", "#7F77DD", "#1D9E75",
    "#D4537E", "#888780",
]

BAR_H       = 10    # bar fill height in px
ROW_H       = 22    # vertical spacing per row
LABEL_W     = 110   # fixed-width label column
VAL_W       = 36    # fixed-width value column
BAR_MAX_W   = 160   # max bar fill width in px


class WeightsPanel:
    """
    Horizontal bar chart of current operator selection weights.

    Each bar represents one (clustering_method, solver) combination.
    Bar width is proportional to normalized weight; bars are sorted
    descending so the dominant operator is always at the top.

    update_weights() is the only public method called from the outside.
    """

    def __init__(self, parent: tk.Widget) -> None:
        self.frame = tk.Frame(parent, bg=BG, bd=0)

        tk.Label(
            self.frame, text="Operator weights",
            font=("Helvetica", 10), fg="#666666", bg=BG, anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 4))

        self._canvas = tk.Canvas(
            self.frame, bg=BG, bd=0, highlightthickness=0, height=200,
        )
        self._canvas.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # canvas item ids, keyed by operator label string
        self._bar_ids:   dict[str, int] = {}
        self._label_ids: dict[str, int] = {}
        self._val_ids:   dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_weights(self, weights: dict[str, float]) -> None:
        """
        Redraw bars to reflect the current weight distribution.

        weights: {"kmeans+filo": 0.31, "angular+filo2": 0.24, ...}
        Keys are operator labels; values are already normalized (sum ≈ 1).
        """
        if not weights:
            return
        sorted_ops = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        self._canvas.delete("all")
        self._bar_ids.clear()
        self._label_ids.clear()
        self._val_ids.clear()

        # TODO: draw each row
        #
        # Layout per row (y = row_index * ROW_H + ROW_H // 2):
        #   x=0..LABEL_W        → right-aligned label text (ts, fg="#888888")
        #   x=LABEL_W+4         → bar track (rectangle, height=BAR_H, fill="#2a2a2a")
        #   x=LABEL_W+4         → bar fill  (rectangle, width=weight*BAR_MAX_W,
        #                                    fill=OPERATOR_COLORS[row_index % len])
        #   x=LABEL_W+BAR_MAX_W+8 → value text "{weight:.0%}" (ts, fg="#cccccc")
        #
        # Store canvas item ids so update_weights() can itemconfig instead of
        # deleting and recreating on subsequent calls (avoids flicker).
        pass

    # ------------------------------------------------------------------
    # Internal helpers (add as needed during implementation)
    # ------------------------------------------------------------------
