#!/usr/bin/env python3
"""Visualize a CVRP instance and a solution on the same matplotlib canvas.

Adapted from the XL-challenge solution plots: white background, a yellow
square for the depot, one colour per route, and no legs from or to the depot.

Default pair: XL-n3888-k1010 and the Lagrange benchmark seed-11 solution,
whose cost of 1,880,215 beats the initial BKS of 1,880,368.

Usage:
    python scripts/generate_instance_solution_plot.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
INSTANCES_DIR = ROOT / "data" / "instances" / "xl"
DEFAULT_INSTANCE = "XL-n3888-k1010"
DEFAULT_SOL = (
    ROOT
    / "data/results/lagrange_benchmark/X-n3888-k1010/seed11"
    / "X-n3888-k1010_0917_0825/X-n3888-k1010.sol"
)
OUTPUT_DIR = ROOT / "thesis" / "figures"

Route = List[int]
Coord = Tuple[float, float]
ROUTE_LINE_RE = re.compile(r"Route #\d+:\s*(.*)")


def _parse_instance_coords(path: Path) -> Dict[int, Coord]:
    """Parse NODE_COORD_SECTION of a CVRPLib .vrp instance."""
    coords: Dict[int, Coord] = {}
    in_coords = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            if not in_coords:
                continue
            if line.startswith("DEMAND_SECTION") or line.startswith("DEPOT_SECTION"):
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue
            coords[idx] = (x, y)

    if not coords:
        raise ValueError(f"No coordinates parsed from {path}")
    return coords


def _parse_solution_routes(path: Path) -> List[Route]:
    """Parse `Route #i:` lines. Drop depot visits (node 0)."""
    routes: List[Route] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = ROUTE_LINE_RE.match(line.strip())
            if not match:
                continue
            tail = match.group(1).strip()
            if not tail:
                continue
            route: List[int] = []
            for tok in tail.split():
                try:
                    node = int(tok)
                except ValueError:
                    continue
                if node == 0:
                    continue
                route.append(node)
            if route:
                routes.append(route)
    if not routes:
        raise ValueError(f"No routes found in solution {path}")
    return routes


_PALETTE: List[Tuple[float, float, float]] = [
    (0.90, 0.10, 0.10),  # red
    (0.10, 0.40, 0.95),  # blue
    (0.10, 0.75, 0.10),  # green
    (0.95, 0.75, 0.10),  # yellow
    (0.95, 0.50, 0.05),  # orange
    (0.60, 0.20, 0.80),  # purple
    (0.10, 0.80, 0.80),  # cyan
    (0.95, 0.10, 0.60),  # magenta
    (0.90, 0.30, 0.30),  # soft red
    (0.30, 0.60, 0.95),  # sky blue
    (0.30, 0.80, 0.50),  # mint
    (0.95, 0.75, 0.35),  # warm yellow
    (0.80, 0.40, 0.90),  # lavender
    (0.40, 0.85, 0.80),  # aqua
    (0.95, 0.55, 0.55),  # coral
    (0.55, 0.75, 0.40),  # olive green
    (0.95, 0.80, 0.55),  # sand
    (0.75, 0.60, 0.95),  # lilac
    (0.65, 0.45, 0.30),  # brown
    (0.55, 0.35, 0.20),  # dark brown
    (0.80, 0.60, 0.45),  # light brown
    (0.70, 0.55, 0.35),  # ochre
    (0.35, 0.35, 0.35),  # dark grey
    (0.55, 0.55, 0.55),  # medium grey
    (0.75, 0.75, 0.75),  # light grey
    (0.60, 0.65, 0.70),  # bluish grey
]


def _generate_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """`n` RGB colours, each taken from the fixed palette, cycling if needed."""
    if n <= 0:
        return []
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal", adjustable="box")
    ticks = list(range(0, 1001, 200))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(
        which="both",
        linestyle="--",
        linewidth=0.4,
        color="lightgray",
        alpha=0.9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def _plot_depot(ax: Axes, coords: Mapping[int, Coord], depot_index: int = 1) -> None:
    if depot_index not in coords:
        return
    dx, dy = coords[depot_index]
    ax.scatter(
        [dx],
        [dy],
        s=80,
        marker="s",
        edgecolor="black",
        facecolor="yellow",
        linewidth=1.0,
        zorder=5,
    )


def _plot_unsolved(ax: Axes, coords: Mapping[int, Coord], depot_index: int = 1) -> None:
    _style_axes(ax)
    xs = [x for idx, (x, _y) in coords.items() if idx != depot_index]
    ys = [y for idx, (_x, y) in coords.items() if idx != depot_index]
    ax.scatter(xs, ys, s=6, color=(0.62, 0.62, 0.62), alpha=0.9)
    _plot_depot(ax, coords, depot_index)


def _plot_solution(
    ax: Axes,
    coords: Mapping[int, Coord],
    routes: Sequence[Route],
    depot_index: int = 1,
) -> None:
    _style_axes(ax)
    colors = _generate_distinct_colors(len(routes))

    for route, color in zip(routes, colors):
        if len(route) == 1:
            x, y = coords[route[0] + 1]
            ax.scatter(x, y, s=8, color=color, alpha=0.9)
            continue

        xs: List[float] = []
        ys: List[float] = []
        for sol_node in route:
            coord_idx = sol_node + 1
            if coord_idx not in coords:
                continue
            x, y = coords[coord_idx]
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, "-", linewidth=0.8, color=color, alpha=0.9)
        ax.scatter(xs, ys, s=6, color=color, alpha=0.9)

    _plot_depot(ax, coords, depot_index)


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout(pad=0.05)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--sol", type=Path, default=DEFAULT_SOL)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    instance_path = INSTANCES_DIR / f"{args.instance}.vrp"
    if not instance_path.is_file():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    if not args.sol.is_file():
        raise FileNotFoundError(f"Solution file not found: {args.sol}")

    coords = _parse_instance_coords(instance_path)
    routes = _parse_solution_routes(args.sol)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.instance
    for line in instance_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("NAME"):
            stem = line.split(":", 1)[1].strip()
            break
    unsolved_path = args.out_dir / f"{stem}_unsolved.png"
    solved_path = args.out_dir / f"{stem}_solved.png"

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor("white")
    _plot_unsolved(ax, coords)
    _save(fig, unsolved_path)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor("white")
    _plot_solution(ax, coords, routes)
    _save(fig, solved_path)

    print(f"{stem}: {len(routes)} routes")
    print(unsolved_path)
    print(solved_path)


if __name__ == "__main__":
    main()
