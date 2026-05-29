"""Tests for dashboard data readers."""

from __future__ import annotations

import json
from pathlib import Path

from drispi.dashboard import data


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n",
        encoding="utf-8",
    )


def test_get_latest_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snapshots"
    snap.mkdir()
    (snap / "iter_1_phase_1.json").write_text("{}", encoding="utf-8")
    (snap / "iter_2_phase_3_pre.json").write_text(
        json.dumps({"iteration": 2, "phase_num": 3, "bg_stage": "pre"}), encoding="utf-8"
    )
    (snap / "iter_2_phase_3_post.json").write_text(
        json.dumps({"iteration": 2, "phase_num": 3, "bg_stage": "post"}), encoding="utf-8"
    )
    result = data.get_latest_snapshot(tmp_path)
    assert result is not None
    assert result["iteration"] == 2
    assert result["phase_num"] == 3
    assert result["bg_stage"] == "post"


def test_get_latest_snapshot_empty(tmp_path: Path) -> None:
    assert data.get_latest_snapshot(tmp_path) is None


def test_get_best_solution_missing(tmp_path: Path) -> None:
    assert data.get_best_solution(tmp_path) is None


def test_get_all_summary_lines(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "run.jsonl",
        [
            {"type": "init"},
            {"type": "summary", "iteration": 0},
            {"type": "phase_done"},
            {"type": "summary", "iteration": 1},
        ],
    )
    lines = data.get_all_summary_lines(tmp_path)
    assert len(lines) == 2
    assert all(l["type"] == "summary" for l in lines)


def test_get_run_metadata_missing(tmp_path: Path) -> None:
    assert data.get_run_metadata(tmp_path) is None


def test_get_latest_run_state_empty(tmp_path: Path) -> None:
    (tmp_path / "run.jsonl").write_text("", encoding="utf-8")
    assert data.get_latest_run_state(tmp_path) is None


def test_get_current_iteration_0_from_snapshots(tmp_path: Path) -> None:
    snap = tmp_path / "snapshots"
    snap.mkdir()
    (snap / "iter_0_phase_3_post.json").write_text(
        '{"iteration":0,"phase_num":3,"bg_stage":"post"}', encoding="utf-8"
    )
    _write_jsonl(
        tmp_path / "run.jsonl",
        [{"type": "summary", "iteration": 0}],
    )
    (snap / "iter_1_phase_1.json").write_text(
        '{"iteration":1,"phase_num":1}', encoding="utf-8"
    )
    assert data.get_current_iteration_0(tmp_path) == 1


def test_get_latest_snapshot_for_iteration(tmp_path: Path) -> None:
    snap = tmp_path / "snapshots"
    snap.mkdir()
    (snap / "iter_1_phase_1.json").write_text(
        '{"iteration":1,"phase_num":1,"timestamp":"10:00:00"}', encoding="utf-8"
    )
    (snap / "iter_1_phase_3_pre.json").write_text(
        '{"iteration":1,"phase_num":3,"bg_stage":"pre","timestamp":"10:01:00"}',
        encoding="utf-8",
    )
    (snap / "iter_1_phase_3_post.json").write_text(
        '{"iteration":1,"phase_num":3,"bg_stage":"post","timestamp":"10:02:00"}',
        encoding="utf-8",
    )
    (snap / "iter_2_phase_1.json").write_text(
        '{"iteration":2,"phase_num":1,"timestamp":"11:00:00"}', encoding="utf-8"
    )
    result = data.get_latest_snapshot_for_iteration(tmp_path, 1)
    assert result is not None
    assert result["bg_stage"] == "post"
    assert result["timestamp"] == "10:02:00"


def test_list_iteration_snapshots(tmp_path: Path) -> None:
    snap = tmp_path / "snapshots"
    snap.mkdir()
    (snap / "iter_2_phase_1.json").write_text('{"phase_num":1,"iteration":2}', encoding="utf-8")
    (snap / "iter_2_phase_3_pre.json").write_text(
        '{"phase_num":3,"iteration":2,"bg_stage":"pre"}', encoding="utf-8"
    )
    (snap / "iter_2_phase_3_post.json").write_text(
        '{"phase_num":3,"iteration":2,"bg_stage":"post"}', encoding="utf-8"
    )
    (snap / "iter_1_phase_2.json").write_text('{"phase_num":2,"iteration":1}', encoding="utf-8")
    snaps = data.list_iteration_snapshots(tmp_path, 2)
    assert len(snaps) == 3
    assert snaps[0]["phase_num"] == 1
    assert snaps[1]["bg_stage"] == "pre"
    assert snaps[2]["bg_stage"] == "post"


def test_get_haos_weights_latest_with_levels(tmp_path: Path) -> None:
    levels = {
        "level_1_k": {
            "values": [1, 2],
            "probabilities": [0.6, 0.4],
            "raw_weights": [1.0, 1.0],
        }
    }
    _write_jsonl(
        tmp_path / "run.jsonl",
        [
            {"type": "haos_roll", "iteration": 0, "k": 1},
            {"type": "haos_roll", "iteration": 0, "k": 1, "lambda_demand": 0.4, "levels": levels},
            {"type": "haos_roll", "iteration": 1, "k": 2},
            {
                "type": "haos_roll",
                "iteration": 1,
                "k": 2,
                "lambda_demand": 0.2,
                "paradigm": "vb",
                "method": "kmeans",
                "solver": "pyvrp",
                "levels": levels,
            },
        ],
    )
    result = data.get_haos_weights(tmp_path)
    assert result is not None
    assert result["k"] == 2
    assert "levels" in result
    assert result["levels"]["level_1_k"]["probabilities"] == [0.6, 0.4]
