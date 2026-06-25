"""Tests for config_io YAML load/save/merge."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.config_io import (
    config_to_dict,
    load_config,
    merge_cli_overrides,
    save_config,
)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = DRISPIConfig(
        time_limit=100.0,
        seed=42,
        haos_k_candidates=[1, 2, 4],
        output_dir=tmp_path / "runs",
    )
    path = tmp_path / "cfg.yaml"
    save_config(original, path)
    loaded = load_config(path)
    assert loaded.time_limit == 100.0
    assert loaded.seed == 42
    assert loaded.haos_k_candidates == [1, 2, 4]
    assert loaded.output_dir == tmp_path / "runs"


def test_load_ignores_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 7\nunknown_field: 99\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_config(path)
    assert loaded.seed == 7
    assert any("unknown_field" in str(w.message) for w in caught)


def test_load_uses_defaults_for_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 99\n", encoding="utf-8")
    loaded = load_config(path)
    assert loaded.seed == 99
    assert loaded.time_limit == DRISPIConfig().time_limit
    assert loaded.n_workers == DRISPIConfig().n_workers


def test_merge_cli_overrides() -> None:
    config = DRISPIConfig()
    merged = merge_cli_overrides(config, {"seed": 555, "time_limit": 10.0})
    assert merged.seed == 555
    assert merged.time_limit == 10.0
    assert config.seed == 123


def test_merge_cli_overrides_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unknown config field"):
        merge_cli_overrides(DRISPIConfig(), {"not_a_field": 1})


def test_config_to_dict_sections() -> None:
    grouped = config_to_dict(DRISPIConfig())
    expected = {
        "stopping",
        "parallelism",
        "sp_sc",
        "subcluster",
        "bg_ails",
        "standard_improvement",
        "pool",
        "haos",
        "haos_rewards",
        "seed",
        "output",
    }
    assert set(grouped) == expected
    assert grouped["stopping"]["time_limit"] == 7200.0
    assert grouped["haos"]["haos_decay"] == 0.95
    assert grouped["sp_sc"]["min_coverage"] == 5


def test_profile_config_inherits_default_yaml(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "default.yaml").write_text(
        "seed: 42\nmax_no_improve: 100\noutput_dir: artifacts/runs\n",
        encoding="utf-8",
    )
    (configs_dir / "benchmark.yaml").write_text(
        "max_no_improve: 200\nrun_analysis: true\noutput_dir: benchmark\n",
        encoding="utf-8",
    )

    loaded = load_config(configs_dir / "benchmark.yaml")
    assert loaded.seed == 42
    assert loaded.max_no_improve == 200
    assert loaded.run_analysis is True
    assert loaded.output_dir == Path("benchmark")
    assert loaded.time_limit == DRISPIConfig().time_limit


def test_loading_default_yaml_does_not_double_merge(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "default.yaml").write_text("seed: 42\n", encoding="utf-8")

    loaded = load_config(configs_dir / "default.yaml")
    assert loaded.seed == 42


def test_repo_benchmark_yaml_inherits_seed_from_default() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_config(repo_root / "configs" / "benchmark.yaml")
    assert loaded.seed == 42
    assert loaded.max_no_improve == 200
