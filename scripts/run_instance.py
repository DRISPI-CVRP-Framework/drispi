"""CLI: run DRISPI on a single instance file."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Run DRISPI on one CVRP instance.")


@app.command()
def main(
    instance: Path = typer.Argument(..., exists=True, readable=True),
    config: Path = typer.Option(Path("configs/default.yaml"), exists=True),
) -> None:
    """Load ``instance`` and execute the pipeline."""
    # TODO: load config YAML, CVRPInstance.from_vrplib, DRISPIPipeline.run
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
