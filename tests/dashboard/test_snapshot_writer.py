"""Tests for SnapshotWriter."""

from __future__ import annotations

import json
import time
from pathlib import Path

from drispi.core.instance import CVRPInstance
from drispi.pipeline.snapshot import SnapshotWriter


def _inst() -> CVRPInstance:
    return CVRPInstance(
        name="toy",
        n_customers=4,
        capacity=30,
        depot=(0.0, 0.0),
        customers=[2, 3, 4, 5],
        coordinates={
            1: (0.0, 0.0),
            2: (0.0, 1.0),
            3: (1.0, 0.0),
            4: (0.0, 2.0),
            5: (2.0, 0.0),
        },
        demands={1: 0, 2: 5, 3: 5, 4: 5, 5: 5},
    )


def test_write_phase_creates_file(tmp_path: Path) -> None:
    writer = SnapshotWriter(tmp_path)
    inst = _inst()
    writer.write_phase(
        3,
        1,
        "dissim+cluster",
        False,
        3,
        inst,
        cluster_assignments={2: 0, 3: 0, 4: 1, 5: 1},
    )
    writer.stop()
    path = tmp_path / "snapshots" / "iter_3_phase_1.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["iteration"] == 3
    assert data["phase_num"] == 1
    assert data["cluster_assignments"] == {"2": 0, "3": 0, "4": 1, "5": 1}
    assert "depot" in data
    assert "customer_ids" in data


def test_write_best_solution(tmp_path: Path) -> None:
    writer = SnapshotWriter(tmp_path)
    inst = _inst()
    writer.write_best_solution([[2, 3], [4, 5]], 100.0, 1, "bg_ails", inst)
    writer.stop()
    path = tmp_path / "best_solution.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cost"] == 100.0
    assert data["routes"] == [[2, 3], [4, 5]]


def test_cleanup_old_snapshots(tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(parents=True)
    for i in range(10):
        (snap_dir / f"iter_{i}_phase_1.json").write_text("{}", encoding="utf-8")
    writer = SnapshotWriter(tmp_path)
    writer.cleanup_old_snapshots(10)
    writer.stop()
    time.sleep(0.1)
    assert (snap_dir / "iter_4_phase_1.json").exists() is False
    assert (snap_dir / "iter_5_phase_1.json").exists() is False
    assert (snap_dir / "iter_6_phase_1.json").exists() is True
    assert (snap_dir / "iter_9_phase_1.json").exists() is True


def test_write_phase_non_blocking(tmp_path: Path) -> None:
    writer = SnapshotWriter(tmp_path)
    inst = _inst()
    writer.write_phase(1, 2, "subclusters", False, 3, inst, routes=[[2, 3]])
    path = tmp_path / "snapshots" / "iter_1_phase_2.json"
    assert not path.exists()
    writer.stop()
    assert path.is_file()
