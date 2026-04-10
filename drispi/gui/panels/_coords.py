from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drispi.core.instance import CVRPInstance

PAD = 12   # pixel padding on each edge of the canvas


class CoordMapper:
    """
    Maps CVRPInstance world coordinates → canvas pixel coordinates.

    Call update() whenever the canvas is resized. Both IterationPanel
    and SolutionPanel hold a CoordMapper instance and share the same
    logic without duplication.

    Usage:
        mapper = CoordMapper(instance)
        mapper.update(canvas_width, canvas_height)
        px, py = mapper(node.x, node.y)
    """

    def __init__(self, instance: CVRPInstance) -> None:
        nodes = instance.nodes          # list of Node with .x and .y
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        self._xmin = min(xs)
        self._xmax = max(xs)
        self._ymin = min(ys)
        self._ymax = max(ys)
        self._cw: int = 1
        self._ch: int = 1

    def update(self, canvas_w: int, canvas_h: int) -> None:
        """Recalculate scaling factors after a canvas resize."""
        self._cw = max(1, canvas_w - 2 * PAD)
        self._ch = max(1, canvas_h - 2 * PAD)

    def __call__(self, x: float, y: float) -> tuple[int, int]:
        """Return (px, py) canvas pixel coordinates for world (x, y)."""
        px = int(PAD + (x - self._xmin) / max(1e-9, self._xmax - self._xmin) * self._cw)
        # y-axis is flipped: higher world-y → lower canvas-y
        py = int(PAD + (self._ymax - y) / max(1e-9, self._ymax - self._ymin) * self._ch)
        return px, py
