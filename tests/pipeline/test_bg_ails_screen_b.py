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


def test_build_jobs_packs_three_omegas_and_idles_fourth_slice() -> None:
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
    assert max(j["wave"] for j in jobs) == 12

    wave1 = [j for j in jobs if j["wave"] == 1]
    assert len(wave1) == 3
    assert {j["variant"] for j in wave1} == set(screen_b.VARIANT_ORDER)
    assert all(j["seed"] == 42 and j["instance"] == "XL-n2307-k34" for j in wave1)
    by_slot = sorted(wave1, key=lambda j: j["slot"])
    assert [j["variant"] for j in by_slot] == list(screen_b.VARIANT_ORDER)
    assert [j["cpus"] for j in by_slot] == ["0-7", "8-15", "16-23"]

    last = [j for j in jobs if j["wave"] == 12]
    assert all(j["seed"] == 45 and j["instance"] == "XL-n8389-k2028" for j in last)


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
    assert "12 wave" in out
    assert "1 idle" in out
    assert "omega_1" in out
    assert "omega_30" in out
    assert "est. wall 24h" in out
    assert "pair_selection=greedy" in out
