"""Tests for Screen B BG-AILS omega campaign packing and config."""

from __future__ import annotations

from pathlib import Path

import scripts.run_bg_ails_screen_b as screen_b

from drispi.pipeline.config_io import load_config


def test_repo_screen_b_config_freezes_greedy_k_async_sp_off() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_config(repo_root / "configs" / "bg_ails_screen_b.yaml")
    assert loaded.bg_ails_mode == "async"
    assert loaded.sp_sc_mode == "off"
    assert loaded.bg_ails_pair_selection == "greedy"
    assert loaded.bg_ails_n_chains_mode == "k"
    assert loaded.cores_total == 8
    assert loaded.cores_dri == 6
    assert loaded.cores_bg == 1
    assert loaded.cores_sp == 1
    assert loaded.run_analysis is False
    assert loaded.bg_ails_divisor == 35.0


def test_build_jobs_fills_four_slices_per_wave() -> None:
    instances = [
        Path("data/instances/xl/XL-n2307-k34.vrp"),
        Path("data/instances/xl/XL-n3975-k687.vrp"),
        Path("data/instances/xl/XL-n8389-k2028.vrp"),
    ]
    seeds = [42, 43, 44, 45]
    jobs = screen_b._build_jobs(
        instances,
        seeds,
        cpu_base=0,
        cores_per_slice=8,
        slices_per_wave=4,
    )
    assert len(jobs) == 36
    assert max(j["wave"] for j in jobs) == 9
    assert all(len([j for j in jobs if j["wave"] == w]) == 4 for w in range(1, 10))

    wave1 = sorted(
        [j for j in jobs if j["wave"] == 1], key=lambda j: j["slot"]
    )
    assert [j["cpus"] for j in wave1] == ["0-7", "8-15", "16-23", "24-31"]
    assert [j["variant"] for j in wave1] == ["omega_1", "omega_10", "omega_30", "omega_1"]
    assert [j["seed"] for j in wave1] == [42, 42, 42, 43]
    assert all(j["instance"] == "XL-n2307-k34" for j in wave1)

    last = [j for j in jobs if j["wave"] == 9]
    assert all(j["instance"] == "XL-n8389-k2028" for j in last)


def test_dry_run_prints_plan(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    argv = [
        "--dry-run",
        "--config",
        str(repo_root / "configs" / "bg_ails_screen_b.yaml"),
        "--cpu-base",
        "0",
        "--total-cores",
        "32",
        "--cores-per-slice",
        "8",
    ]
    screen_b.main(argv)
    out = capsys.readouterr().out
    assert "36 runs" in out
    assert "9 wave" in out
    assert "omega_1" in out
    assert "omega_30" in out
    assert "est. wall 18h" in out
    assert "pair_selection=greedy" in out
    assert "idle" not in out
