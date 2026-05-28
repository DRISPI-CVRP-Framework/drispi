"""Tests for BKS resolution helper."""

from __future__ import annotations

from pathlib import Path

from drispi.utils.metrics import resolve_bks_cost


def test_resolve_bks_override() -> None:
    assert resolve_bks_cost("XL-n1", bks_override=99.0, bks_file=Path("x")) == 99.0


def test_resolve_bks_from_file(tmp_path: Path) -> None:
    path = tmp_path / "bks.json"
    path.write_text('{"inst-a": 42.0}', encoding="utf-8")
    assert resolve_bks_cost("inst-a", bks_file=path) == 42.0
    assert resolve_bks_cost("missing", bks_file=path) is None


def test_resolve_bks_missing_file() -> None:
    assert resolve_bks_cost("x", bks_file=Path("/nonexistent/bks.json")) is None
