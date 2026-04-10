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

SOLUTION_COLOR = (29,  158, 117)   # teal — all routes same color
DOT_COLOR      = (29,  158, 117)   # same teal for customer dots
DEPOT_COLOR    = (255, 255, 255)
DOT_RADIUS     = 2
DEPOT_HALF     = 5


class SolutionPanel:
    """
    Displays the current best solution.

    All routes are drawn in SOLUTION_COLOR (no cluster distinction).
    Only updated on BEST_UPDATED events — typically infrequent.

    Rendering approach is identical to IterationPanel:
        PIL Image → PhotoImage  (customer dots)
        Canvas lines on top     (route polylines)
    """

    def __init__(self, parent: tk.Widget, instance: CVRPInstance) -> None:
        self._instance = instance
        self._mapper   = CoordMapper(instance)
        self.frame     = tk.Frame(parent, bg=BG, bd=0)

        # --- header ---
        header = tk.Frame(self.frame, bg=BG)
        header.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(header, text="Best solution", font=("Helvetica", 10),
                 fg="#666666", bg=BG, anchor="w").pack(side="left")
        self._lbl_cost = tk.Label(header, text="—", font=("Helvetica", 9),
                                  fg="#444444", bg=BG)
        self._lbl_cost.pack(side="right")

        # --- canvas ---
        self._canvas = tk.Canvas(
            self.frame, bg=CANVAS_BG, bd=0, highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._canvas.bind("<Configure>", self._on_resize)

        # internal state
        self._photo:          ImageTk.PhotoImage | None = None
        self._canvas_img_id:  int | None = None
        self._route_line_ids: list[int] = []
        self._current_routes: list[list[int]] = []
        self._canvas_w:       int = 1
        self._canvas_h:       int = 1

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def show_solution(self, routes: list[list[int]], cost: float) -> None:
        """
        Render the new best solution. Called only on BEST_UPDATED.

        Full redraw each time — acceptable because BEST_UPDATED is infrequent.
        """
        self._current_routes = routes
        self._lbl_cost.config(text=f"{cost:,.0f}")
        self._clear_routes()
        self._draw_dots()
        self._draw_routes(routes)

    # ------------------------------------------------------------------
    # Rendering internals
    # ------------------------------------------------------------------

    def _on_resize(self, event: tk.Event) -> None:
        self._canvas_w = event.width
        self._canvas_h = event.height
        self._mapper.update(event.width, event.height)
        if self._current_routes:
            self._clear_routes()
            self._draw_dots()
            self._draw_routes(self._current_routes)

    def _draw_dots(self) -> None:
        """
        Draw all customer dots in DOT_COLOR via PIL, plus white depot square.

        TODO: implement using the same PIL pattern as IterationPanel._redraw_dots():
            img = Image.new("RGB", (canvas_w, canvas_h), CANVAS_BG)
            draw = ImageDraw.Draw(img)
            for node in instance.nodes[1:]:       # skip depot
                px, py = self._mapper(node.x, node.y)
                r = DOT_RADIUS
                draw.ellipse([px-r, py-r, px+r, py+r], fill=DOT_COLOR)
            depot = instance.nodes[0]
            dpx, dpy = self._mapper(depot.x, depot.y)
            draw.rectangle([dpx-DEPOT_HALF, dpy-DEPOT_HALF,
                            dpx+DEPOT_HALF, dpy+DEPOT_HALF], fill=DEPOT_COLOR)
            self._blit(img)
        """
        if not PIL_AVAILABLE or self._canvas_w < 2 or self._canvas_h < 2:
            return
        # TODO: implement
        pass

    def _draw_routes(self, routes: list[list[int]]) -> None:
        """
        Draw all route polylines in SOLUTION_COLOR on the Canvas layer.

        TODO: same implementation as IterationPanel._draw_routes() but with
        SOLUTION_COLOR hardcoded — no per-cluster color selection needed.
        """
        # TODO: implement
        pass

    def _blit(self, img: "Image.Image") -> None:
        self._photo = ImageTk.PhotoImage(img)
        if self._canvas_img_id is None:
            self._canvas_img_id = self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self._canvas.itemconfig(self._canvas_img_id, image=self._photo)
        self._canvas.tag_lower(self._canvas_img_id)

    def _clear_routes(self) -> None:
        for lid in self._route_line_ids:
            self._canvas.delete(lid)
        self._route_line_ids.clear()
