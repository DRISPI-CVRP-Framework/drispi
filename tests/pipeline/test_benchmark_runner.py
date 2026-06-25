"""Tests for scripts/run_benchmark.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.core_manager import CoreManager

import scripts.run_benchmark as benchmark


def test_dry_run_prints_plan_and_exits(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    inst = tmp_path / "X-n10-k3.vrp"
    inst.write_text("NAME: X-n10-k3\n", encoding="utf-8")
    argv = [
        "run_benchmark.py",
        str(tmp_path / "*.vrp"),
        "--total-cores",
        "8",
        "--cores-per-instance",
        "4",
        "--dry-run",
    ]
    with patch("scripts.run_benchmark.sys.argv", argv):
        benchmark.main()
    out = capsys.readouterr().out
    assert "Discovered 1 instance" in out
    assert "Parallelism: 2 instance" in out
    assert "Estimated wall time" in out


def test_dry_run_multiple_explicit_paths(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    inst_a = tmp_path / "X-n10-k3.vrp"
    inst_b = tmp_path / "X-n11-k3.vrp"
    inst_a.write_text("NAME: X-n10-k3\n", encoding="utf-8")
    inst_b.write_text("NAME: X-n11-k3\n", encoding="utf-8")
    argv = [
        "run_benchmark.py",
        str(inst_a),
        str(inst_b),
        "--total-cores",
        "8",
        "--cores-per-instance",
        "4",
        "--dry-run",
    ]
    with patch("scripts.run_benchmark.sys.argv", argv):
        benchmark.main()
    out = capsys.readouterr().out
    assert "Discovered 2 instance(s)" in out
    assert "X-n10-k3" in out
    assert "X-n11-k3" in out


    mock_pipeline = MagicMock()
    mock_pipeline.best_cost = 100.0
    mock_pipeline.iterations_completed = 3
    mock_pipeline.stop_reason = "no_improve"
    mock_pipeline.run_dir = tmp_path / "run"

    with patch("scripts.run_benchmark.load_instance_from_vrp_path") as load_inst:
        load_inst.return_value = MagicMock(name="X-n10-k3", n_customers=10)
        with patch("scripts.run_benchmark.DRISPIPipeline", return_value=mock_pipeline):
            with patch("scripts.run_benchmark.make_run_label", return_value="run"):
                result = benchmark._run_single_instance(
                    tmp_path / "x.vrp",
                    DRISPIConfig(),
                    99.0,
                    CoreManager(4, 2),
                    "X-n10-k3",
                )
    assert result["status"] == "ok"
    assert result["best_cost"] == 100.0
    assert result["iterations"] == 3
    assert result["stop_reason"] == "no_improve"
    assert result["error"] is None


def test_run_single_instance_catches_errors(tmp_path: Path) -> None:
    with patch(
        "scripts.run_benchmark.load_instance_from_vrp_path",
        side_effect=RuntimeError("boom"),
    ):
        result = benchmark._run_single_instance(
            tmp_path / "bad.vrp",
            DRISPIConfig(),
            None,
            CoreManager(4, 2),
            "bad",
        )
    assert result["status"] == "error"
    assert result["error"] == "boom"


def test_progress_output(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark._print_progress(
        1,
        2,
        {
            "instance": "X-n10-k3",
            "status": "ok",
            "best_cost": 12345.0,
            "gap_pct": 0.5,
            "runtime_s": 60.0,
            "error": None,
        },
    )
    out = capsys.readouterr().out
    assert "X-n10-k3" in out
    assert "done" in out

    benchmark._print_progress(
        2,
        2,
        {"instance": "bad", "status": "error", "error": "fail"},
    )
    err_out = capsys.readouterr().out
    assert "ERROR: fail" in err_out
