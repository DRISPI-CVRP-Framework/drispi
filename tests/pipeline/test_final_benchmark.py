"""Tests for the final-benchmark campaign packing and config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drispi.pipeline.config_io import load_config

import scripts.run_final_benchmark as fb


def test_repo_final_benchmark_config_is_production_slice() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_config(repo_root / "configs" / "final_benchmark.yaml")
    assert loaded.bg_ails_mode == "async"
    assert loaded.sp_sc_mode == "async"
    assert loaded.sp_sc_trigger == "wallclock"
    assert loaded.cores_total == 8
    assert loaded.cores_dri == 6
    assert loaded.cores_bg == 1
    assert loaded.cores_sp == 1
    assert loaded.time_limit == 7200.0
    assert loaded.max_no_improve >= 10**9
    assert loaded.run_analysis is False
    assert loaded.bg_ails_pair_selection == "greedy"
    assert loaded.bg_ails_n_chains_mode == "k"
    assert loaded.bg_ails_initial_omega == 10.0
    assert loaded.bg_ails_divisor == 35.0
    assert loaded.sp_time_limit == 720.0


def test_build_jobs_orders_by_size_then_seed(tmp_path: Path) -> None:
    instances = [
        tmp_path / "XL-n10001-k1.vrp",
        tmp_path / "XL-n1048-k1.vrp",
        tmp_path / "XL-n2307-k1.vrp",
        tmp_path / "XL-n3975-k1.vrp",
        tmp_path / "XL-n500-k1.vrp",
        tmp_path / "XL-n800-k1.vrp",
        tmp_path / "XL-n900-k1.vrp",
        tmp_path / "XL-n950-k1.vrp",
    ]
    jobs = fb._build_jobs(
        instances,
        [100, 200, 300],
        cpu_base=0,
        cores_per_slice=8,
        slices_per_wave=4,
    )
    assert len(jobs) == 24
    assert max(j["wave"] for j in jobs) == 6
    wave1 = [j for j in jobs if j["wave"] == 1]
    assert [j["instance_stem"] for j in sorted(wave1, key=lambda x: x["slot"])] == [
        "XL-n500-k1",
        "XL-n800-k1",
        "XL-n900-k1",
        "XL-n950-k1",
    ]
    assert {j["seed"] for j in wave1} == {100}
    assert [j["cpus"] for j in sorted(wave1, key=lambda x: x["slot"])] == [
        "0-7",
        "8-15",
        "16-23",
        "24-31",
    ]
    stems_per_wave = []
    for w in range(1, 7):
        stems_per_wave.append([j["instance_stem"] for j in jobs if j["wave"] == w])
        assert len({j["instance_stem"] for j in jobs if j["wave"] == w}) == 4
    last = [j for j in jobs if j["wave"] == 6]
    assert {j["seed"] for j in last} == {300}
    assert "XL-n10001-k1" in {j["instance_stem"] for j in last}


def test_resume_skips_completed(tmp_path: Path) -> None:
    jsonl = tmp_path / "campaign.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "status": "ok",
                "instance_stem": "XL-n500-k1",
                "seed": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    done = fb._load_completed(jsonl)
    assert done == {("XL-n500-k1", 100)}
    failed_only = tmp_path / "failed.jsonl"
    failed_only.write_text(
        json.dumps({"status": "error", "instance_stem": "XL-n500-k1", "seed": 200})
        + "\n",
        encoding="utf-8",
    )
    assert fb._load_completed(failed_only) == set()


def test_dry_run_default_xl_count(capsys: pytest.CaptureFixture[str]) -> None:
    repo = Path(__file__).resolve().parents[2]
    xl = repo / "data" / "instances" / "xl"
    if not xl.is_dir() or len(list(xl.glob("*.vrp"))) != 100:
        pytest.skip("XL instance set not present")
    fb.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "100 instances × 3 seeds = 300 runs" in out
    assert "75 waves" in out
    assert "pending: 300" in out


def test_config_io_final_benchmark_inherits_seed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_config(repo_root / "configs" / "final_benchmark.yaml")
    assert loaded.seed == 42
