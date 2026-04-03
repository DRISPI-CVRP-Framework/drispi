"""CLI: batch benchmark over instances under data/instances."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Batch-run DRISPI on many instances.")


@app.command()
def main(
    instances_dir: Path = typer.Option(Path("data/instances")),
    config: Path = typer.Option(Path("configs/default.yaml"), exists=True),
) -> None:
    """Scan ``instances_dir`` and run the pipeline for each file."""
    # TODO: glob instances, aggregate metrics, write results/
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
