"""CLI: compare solutions to best-known scores in data/bks."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Evaluate solution quality vs BKS tables.")


@app.command()
def main(
    results: Path = typer.Option(Path("results"), help="Directory with run outputs"),
    bks: Path = typer.Option(Path("data/bks"), help="Best-known solution files"),
) -> None:
    """Load results and BKS files; print gaps."""
    # TODO: parse JSON results and BKS format
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
