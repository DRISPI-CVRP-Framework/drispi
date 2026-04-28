"""CLI: initialize HAOS weight snapshot files."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Initialize default HAOS weight JSON for experiments.")


@app.command()
def main(
    output: Path = typer.Option(Path("data/weights.json"), help="Output JSON path"),
) -> None:
    """Write a default weight snapshot to ``output``."""
    # TODO: build HierarchicalHAOS snapshot and serialize JSON
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
