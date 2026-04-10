from __future__ import annotations
import tkinter as tk

BG = "#1a1a1a"

# Order and display labels for route pool statistics.
# Keys must match what GUIManager.on_sc_done() passes in pool_stats.
STAT_KEYS: list[tuple[str, str]] = [
    ("pool_size",       "Pool size"),
    ("unique_routes",   "Unique routes"),
    ("avg_route_len",   "Avg route len"),
    ("n_clusters",      "Clusters / iter"),
    ("last_sp_time_s",  "Last SP time"),
    ("ls_hits",         "LS hits"),
    ("no_imp_streak",   "No-imp streak"),
]


class PoolPanel:
    """
    Displays route pool summary statistics as a two-column label list.

    No canvas needed — pure Tkinter labels. update() is called from
    window._on_sc_done() whenever the SC produces a new solution.
    """

    def __init__(self, parent: tk.Widget) -> None:
        self.frame = tk.Frame(parent, bg=BG, bd=0)

        tk.Label(
            self.frame, text="Route pool",
            font=("Helvetica", 10), fg="#666666", bg=BG, anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 4))

        inner = tk.Frame(self.frame, bg=BG)
        inner.pack(fill="x", padx=10, pady=(0, 8))

        self._vars: dict[str, tk.StringVar] = {}
        for key, label in STAT_KEYS:
            row = tk.Frame(inner, bg=BG)
            row.pack(fill="x", pady=1)
            tk.Frame(row, bg="#2a2a2a", height=1).pack(fill="x")  # divider

            tk.Label(
                row, text=label, font=("Helvetica", 10),
                fg="#555555", bg=BG, anchor="w",
            ).pack(side="left")

            var = tk.StringVar(value="—")
            self._vars[key] = var
            tk.Label(
                row, textvariable=var, font=("Helvetica", 10, "bold"),
                fg="#bbbbbb", bg=BG, anchor="e",
            ).pack(side="right")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, stats: dict) -> None:
        """
        Update displayed values from a stats dict.

        Expected keys (all optional — missing keys keep their previous value):
            pool_size       int
            unique_routes   int
            avg_route_len   float
            n_clusters      int
            last_sp_time_s  float   (seconds, displayed as "4.2 s")
            ls_hits         int
            no_imp_streak   str     (e.g. "4 / 50")
        """
        for key, _ in STAT_KEYS:
            if key not in stats:
                continue
            val = stats[key]
            if key == "last_sp_time_s":
                self._vars[key].set(f"{val:.1f} s")
            elif isinstance(val, float):
                self._vars[key].set(f"{val:.1f}")
            elif isinstance(val, int):
                self._vars[key].set(f"{val:,}")
            else:
                self._vars[key].set(str(val))
