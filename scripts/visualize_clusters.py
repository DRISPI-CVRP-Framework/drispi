"""Generate cluster visualizations for one instance and method set."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import typer

from drispi.clustering.interface import ROUTE_METHODS, VERTEX_METHODS, cluster_instance
from drispi.core.instance import CVRPInstance
from drispi.utils.io import read_sol

app = typer.Typer(help="Plot clustering outputs on top of instance geometry.")


def _draw_clusters(
    instance: CVRPInstance,
    groups: list[list[int]],
    title: str,
    output_path: Path,
) -> None:
    depot_x, depot_y = instance.depot
    plt.figure(figsize=(8, 6))
    plt.scatter([depot_x], [depot_y], c="black", marker="s", s=80, label="Depot")

    cmap = plt.cm.get_cmap("tab20", max(1, len(groups)))
    for cluster_idx, group in enumerate(groups):
        if not group:
            continue
        xs = [instance.coordinates[cid][0] for cid in group]
        ys = [instance.coordinates[cid][1] for cid in group]
        plt.scatter(xs, ys, s=24, color=cmap(cluster_idx), label=f"C{cluster_idx}")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


@app.command()
def main(
    instance: str = typer.Option("X-n101-k25", help="VRPLIB instance name."),
    paradigm: str = typer.Option("vertex", help="vertex or route."),
    methods: str = typer.Option(
        "all",
        help="Comma-separated methods, or 'all'.",
    ),
    k: int = typer.Option(8, help="Number of clusters."),
    seed: int = typer.Option(42, help="Random seed."),
    lambda_demand: float = typer.Option(0.2, help="Demand-scaling coefficient."),
    angular_offset: float = typer.Option(0.7, help="Angular offset in radians."),
    routes_sol: Path | None = typer.Option(
        None,
        help="Path to .sol file used only for route paradigm.",
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/clustering"),
        help="Directory where PNGs are saved.",
    ),
) -> None:
    """Cluster one instance and save one PNG per selected method."""
    cvrp = CVRPInstance.from_vrplib(instance)
    if paradigm not in {"vertex", "route"}:
        raise typer.BadParameter("paradigm must be either 'vertex' or 'route'.")

    available = sorted(VERTEX_METHODS if paradigm == "vertex" else ROUTE_METHODS)
    selected = available if methods == "all" else [m.strip() for m in methods.split(",") if m.strip()]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise typer.BadParameter(f"Unknown methods for paradigm='{paradigm}': {unknown}")

    routes: list[list[int]] | None = None
    if paradigm == "route":
        if routes_sol is None:
            raise typer.BadParameter("route paradigm requires --routes-sol.")
        routes, _ = read_sol(routes_sol)

    for method in selected:
        try:
            groups = cluster_instance(
                instance=cvrp,
                paradigm=paradigm,
                method=method,
                k=k,
                routes=routes,
                lambda_demand=lambda_demand,
                angular_offset=angular_offset,
                seed=seed,
            )
        except Exception as exc:  # pragma: no cover - interactive utility
            typer.echo(f"[skip] {method}: {exc}")
            continue

        out_file = output_dir / f"{instance}_{paradigm}_{method}_k{k}_seed{seed}.png"
        _draw_clusters(cvrp, groups, f"{instance} | {paradigm}:{method} | k={k}", out_file)
        typer.echo(f"[ok] {method} -> {out_file}")


if __name__ == "__main__":
    app()
