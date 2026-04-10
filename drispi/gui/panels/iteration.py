from __future__ import annotations
import tkinter as tk
from typing import TYPE_CHECKING

try:
    from PIL import Image, ImageDraw, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ._coords import CoordMapper

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance

BG        = "#1a1a1a"
CANVAS_BG = "#080808"

# RGB tuples for up to 12 clusters (extend if needed)
CLUSTER_COLORS: list[tuple[int, int, int]] = [
    (55,  138, 221),   # blue
    (99,  153,  34),   # green
    (239, 159,  39),   # amber
    (216,  90,  48),   # coral
    (127, 119, 221),   # purple
    (29,  158, 117),   # teal
    (212,  83, 126),   # pink
    (136, 135, 128),   # gray
    (231,  76,  60),   # red
    (52,  152, 219),   # light blue
    (241, 196,  15),   # yellow
    (46,  204, 113),   # emerald
]

PENDING_ALPHA      = 30    # 0–255, dimness for unsolved clusters
DOT_RADIUS         = 2     # customer dot radius in pixels
DEPOT_HALF         = 5     # depot square half-size in pixels
LS_HIGHLIGHT_COLOR = (255, 255, 255)   # white overlay for LS-modified customers
SC_DONE_COLOR      = (29,  158, 117)   # teal for SP/SC solution
ROUTE_ALPHA        = 140   # 0–255 opacity for route lines on Canvas


class IterationPanel:
    """
    Visualizes the current iteration state phase by phase.

    Rendering layers:
        1. PIL Image (bottom) — customer dots, colored by cluster.
           Regenerated as a flat RGBA image; blit as a single PhotoImage.
           Handles 10k customers at < 5ms.

        2. tk.Canvas lines (top) — route lines per cluster.
           Added incrementally as clusters are solved; cleared on reset().

    Phase sequence (called from window.py):
        reset()               → blank canvas, awaiting decomposition
        show_decomposed()     → dots colored by cluster, no routes
        show_cluster_solved() → add route lines for the finished cluster;
                                dots for pending clusters remain dim
        show_ls_done()        → overlay modified customers in white
        show_sc_done()        → all routes in SC_DONE_COLOR, dots uniform
    """

    def __init__(self, parent: tk.Widget, instance: CVRPInstance) -> None:
        self._instance = instance
        self._mapper   = CoordMapper(instance)
        self.frame     = tk.Frame(parent, bg=BG, bd=0)

        # --- header ---
        header = tk.Frame(self.frame, bg=BG)
        header.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(header, text="Current iteration", font=("Helvetica", 10),
                 fg="#666666", bg=BG, anchor="w").pack(side="left")
        self._lbl_phase = tk.Label(header, text="Idle",
                                   font=("Helvetica", 9, "bold"),
                                   fg="#2196f3", bg=BG)
        self._lbl_phase.pack(side="right")

        # --- canvas ---
        self._canvas = tk.Canvas(
            self.frame, bg=CANVAS_BG, bd=0, highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._canvas.bind("<Configure>", self._on_resize)

        # internal state
        self._photo:           ImageTk.PhotoImage | None = None
        self._canvas_img_id:   int | None = None
        self._route_line_ids:  list[int] = []
        self._assignments:     list[int] = []      # customer_idx → cluster_id
        self._solved_clusters: set[int] = set()    # clusters with routes drawn
        self._canvas_w:        int = 1
        self._canvas_h:        int = 1

    # ------------------------------------------------------------------
    # Public phase methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear to blank state at the start of a new iteration."""
        self._assignments     = []
        self._solved_clusters = set()
        self._clear_routes()
        self._lbl_phase.config(text="Decomposing…")
        self._draw_blank()

    def show_decomposed(self, assignments: list[int]) -> None:
        """Color every customer by its cluster; no routes yet."""
        self._assignments = assignments
        self._solved_clusters = set()
        self._clear_routes()
        self._lbl_phase.config(text="Routing…")
        self._redraw_dots()

    def show_cluster_solved(
        self,
        cluster_id: int,
        n_clusters: int,
        routes: list[list[int]],
    ) -> None:
        """
        Add route lines for the newly solved cluster.
        Pending clusters remain visually dimmed.
        """
        self._solved_clusters.add(cluster_id)
        self._lbl_phase.config(text=f"Routing {len(self._solved_clusters)} / {n_clusters}")
        # Redraw dots so solved clusters appear at full brightness
        self._redraw_dots()
        self._draw_routes(routes, color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)])

    def show_ls_done(self, modified_customers: set[int], phase: int) -> None:
        """
        Overlay modified customers in LS_HIGHLIGHT_COLOR.
        All other customers keep their cluster color.
        """
        self._lbl_phase.config(text=f"LS{phase} done")
        # TODO: create a second PIL RGBA layer (transparent background)
        #       draw LS_HIGHLIGHT_COLOR dots for each idx in modified_customers
        #       composite over the base image and blit
        pass

    def show_sc_done(self, routes: list[list[int]]) -> None:
        """Show the complete SP/SC solution: all routes in SC_DONE_COLOR."""
        self._lbl_phase.config(text="SP/SC done")
        self._clear_routes()
        # TODO: redraw all customer dots in SC_DONE_COLOR (uniform)
        self._draw_routes(routes, color=SC_DONE_COLOR)

    # ------------------------------------------------------------------
    # Rendering internals
    # ------------------------------------------------------------------

    def _on_resize(self, event: tk.Event) -> None:
        self._canvas_w = event.width
        self._canvas_h = event.height
        self._mapper.update(event.width, event.height)
        self._redraw_dots()

    def _draw_blank(self) -> None:
        """Blit a solid black image (pre-run state)."""
        # TODO: create solid CANVAS_BG Image, convert, blit
        pass

    def _redraw_dots(self) -> None:
        """
        Regenerate the PIL dot layer from self._assignments.

        Algorithm:
            img = Image.new("RGB", (canvas_w, canvas_h), CANVAS_BG)
            draw = ImageDraw.Draw(img)
            for i, cluster_id in enumerate(self._assignments):
                px, py = self._mapper(instance.nodes[i+1].x, instance.nodes[i+1].y)
                solved = cluster_id in self._solved_clusters
                color  = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
                alpha  = 255 if solved (or no assignments yet) else PENDING_ALPHA
                # blend color toward background by alpha
                blended = tuple(int(c * alpha/255 + 8 * (1 - alpha/255)) for c in color)
                r = DOT_RADIUS
                draw.ellipse([px-r, py-r, px+r, py+r], fill=blended)
            # draw depot as white square
            dpx, dpy = self._mapper(depot.x, depot.y)
            draw.rectangle([dpx-DEPOT_HALF, dpy-DEPOT_HALF,
                            dpx+DEPOT_HALF, dpy+DEPOT_HALF], fill=(255,255,255))
            self._blit(img)
        """
        if not PIL_AVAILABLE or self._canvas_w < 2 or self._canvas_h < 2:
            return
        # TODO: implement as described above
        pass

    def _draw_routes(
        self,
        routes: list[list[int]],
        color: tuple[int, int, int],
    ) -> None:
        """
        Draw route polylines on the Canvas layer.

        Each route: depot → customer[0] → … → customer[-1] → depot.
        Stores all created canvas item ids in self._route_line_ids for later cleanup.

        color: RGB tuple, will be converted to hex for Canvas.
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        """
        # TODO: implement
        # nodes = self._instance.nodes
        # depot = nodes[0]
        # for route in routes:
        #     coords = [self._mapper(depot.x, depot.y)]
        #     for idx in route:
        #         n = nodes[idx + 1]   # +1 because depot is index 0
        #         coords.append(self._mapper(n.x, n.y))
        #     coords.append(self._mapper(depot.x, depot.y))
        #     flat = [c for xy in coords for c in xy]
        #     lid = self._canvas.create_line(*flat, fill=hex_color, width=1, smooth=False)
        #     self._route_line_ids.append(lid)
        pass

    def _blit(self, img: "Image.Image") -> None:
        """Convert PIL Image to PhotoImage and place on canvas."""
        self._photo = ImageTk.PhotoImage(img)
        if self._canvas_img_id is None:
            self._canvas_img_id = self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self._canvas.itemconfig(self._canvas_img_id, image=self._photo)
        self._canvas.tag_lower(self._canvas_img_id)   # keep dots below route lines

    def _clear_routes(self) -> None:
        for lid in self._route_line_ids:
            self._canvas.delete(lid)
        self._route_line_ids.clear()
