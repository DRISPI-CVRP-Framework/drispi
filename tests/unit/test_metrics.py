"""Tests for BKS metric helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drispi.utils.metrics import gap_to_bks, read_bks


def test_gap_to_bks_zero_when_equal() -> None:
    bks = {"XL-n1048-k237": 380107.0}
    assert gap_to_bks(380107.0, "XL-n1048-k237", bks) == 0.0


def test_gap_to_bks_positive_percentage() -> None:
    bks = {"XL-n1048-k237": 100.0}
    assert gap_to_bks(105.0, "XL-n1048-k237", bks) == 5.0


def test_gap_to_bks_unknown_instance_raises() -> None:
    with pytest.raises(KeyError):
        gap_to_bks(100.0, "UNKNOWN", {"XL-n1048-k237": 100.0})


def test_read_bks_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_bks(tmp_path / "missing.json")


def test_read_bks_returns_dict_for_valid_json(tmp_path: Path) -> None:
    data = {"XL-a": 10, "XL-b": 20.5}
    path = tmp_path / "bks.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    out = read_bks(path)

    assert out == {"XL-a": 10.0, "XL-b": 20.5}
