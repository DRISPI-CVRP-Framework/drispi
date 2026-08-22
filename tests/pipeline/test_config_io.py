"""Tests for config_io YAML load/save/merge."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from drispi.core.instance import CVRPInstance
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.config_io import (
    config_to_dict,
    load_config,
    merge_cli_overrides,
    save_config,
)
from drispi.pipeline.pipeline import _build_haos_config


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = DRISPIConfig(
        time_limit=100.0,
        seed=42,
        decomp_k_base_arms=[2, 4, 8],
        decomp_k_ext_per_1000=4,
        bg_ails_budget_margin=0.9,
        output_dir=tmp_path / "runs",
    )
    path = tmp_path / "cfg.yaml"
    save_config(original, path)
    loaded = load_config(path)
    assert loaded.time_limit == 100.0
    assert loaded.seed == 42
    assert loaded.decomp_k_base_arms == [2, 4, 8]
    assert loaded.decomp_k_ext_per_1000 == 4
    assert loaded.bg_ails_budget_margin == 0.9
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
        "cores",
        "sp_sc",
        "decomposition",
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
    assert grouped["sp_sc"]["sp_sc_mode"] == "sync"
    assert grouped["sp_sc"]["sp_sc_trigger"] == "iteration"
    assert grouped["cores"]["cores_total"] is None
    assert grouped["decomposition"]["decomp_k_base_arms"] == [2, 3, 4, 6, 8, 10, 12]
    assert grouped["decomposition"]["decomp_k_ext_per_1000"] == 4
    assert grouped["decomposition"]["subcluster_floor_s"] == 5.0
    assert grouped["bg_ails"]["bg_ails_budget_floor_s"] == 60.0
    assert grouped["bg_ails"]["bg_ails_budget_margin"] == 1.0
    assert grouped["bg_ails"]["bg_ails_wall_model_scale"] == 0.976
    assert grouped["bg_ails"]["bg_ails_wall_model_wave_exponent"] == -0.180
    assert "bg_ails_time_limit" not in grouped["bg_ails"]
    assert "bg_ails_divisor" not in grouped["bg_ails"]


def test_load_rejects_removed_bg_ails_time_limit(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("bg_ails_time_limit: 120.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bg_ails_time_limit was removed"):
        load_config(path)


def test_load_nested_sp_sc_unquoted_off_is_mode_off(tmp_path: Path) -> None:
    """YAML 1.1 parses bare ``off`` as false; loader must not re-enable SP/SC."""
    path = tmp_path / "cfg.yaml"
    path.write_text("sp_sc:\n  mode: off\n", encoding="utf-8")
    loaded = load_config(path)
    assert loaded.sp_sc_mode == "off"


def test_load_nested_bg_ails_block(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "bg_ails:\n"
        "  mode: async\n"
        "  crash_threshold: 2\n"
        "  budget:\n"
        "    floor_s: 90.0\n"
        "    margin: 0.9\n",
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.bg_ails_mode == "async"
    assert loaded.bg_ails_crash_threshold == 2
    assert loaded.bg_ails_budget_floor_s == 90.0
    assert loaded.bg_ails_budget_margin == 0.9
    # Untouched nested keys fall back to dataclass defaults.
    assert loaded.bg_ails_wall_model_scale == DRISPIConfig().bg_ails_wall_model_scale
    assert loaded.bg_ails_wall_model_wave_exponent == DRISPIConfig().bg_ails_wall_model_wave_exponent


def test_load_nested_bg_ails_rejects_time_limit(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("bg_ails:\n  time_limit: 120.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bg_ails.time_limit was removed"):
        load_config(path)


@pytest.mark.parametrize("key", ["min_budget", "divisor", "divisor_assumes_time_per_customer"])
def test_load_nested_bg_ails_rejects_removed_static_budget_keys(
    tmp_path: Path, key: str
) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(f"bg_ails:\n  {key}: 35.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="was removed"):
        load_config(path)


def test_load_rejects_removed_flat_keys(tmp_path: Path) -> None:
    for body, match in (
        ("subcluster_time_per_customer: 0.06\n", "subcluster_time_per_customer was removed"),
        ("haos_k_candidates: [1, 2]\n", "haos_k_candidates was removed"),
        ("bg_ails_divisor: 35.0\n", "was removed"),
    ):
        path = tmp_path / "cfg.yaml"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(path)


def test_load_nested_decomposition_block(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "decomposition:\n"
        "  k_domain:\n"
        "    base_arms: [2, 4, 8]\n"
        "    max_arms: 8\n"
        "    ext_per_1000: 4\n"
        "  subcluster_budget:\n"
        "    rate_s_per_customer: 0.05\n"
        "    floor_s: 4.0\n",
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.decomp_k_base_arms == [2, 4, 8]
    assert loaded.decomp_k_max_arms == 8
    assert loaded.decomp_k_ext_per_1000 == 4
    assert loaded.subcluster_rate_s_per_customer == 0.05
    assert loaded.subcluster_floor_s == 4.0


def test_load_nested_bg_ails_warns_on_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("bg_ails:\n  mode: sync\n  bogus: 1\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_config(path)
    assert loaded.bg_ails_mode == "sync"
    assert any("bogus" in str(w.message) for w in caught)


def test_load_nested_cores_bg(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "cores:\n  total: 8\n  dri: 6\n  bg: 1\n  sp: 1\n",
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.cores_total == 8
    assert loaded.cores_dri == 6
    assert loaded.cores_bg == 1
    assert loaded.cores_sp == 1


def test_save_and_load_round_trip_cores_bg(tmp_path: Path) -> None:
    original = DRISPIConfig(
        cores_total=8,
        cores_dri=6,
        cores_bg=1,
        cores_sp=1,
        bg_ails_mode="async",
        output_dir=tmp_path / "runs",
    )
    path = tmp_path / "cfg.yaml"
    save_config(original, path)
    loaded = load_config(path)
    assert loaded.cores_bg == 1
    assert loaded.cores_dri == 6
    assert loaded.bg_ails_mode == "async"


def test_repo_default_yaml_bg_ails_budget_keys() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_config(repo_root / "configs" / "default.yaml")
    assert loaded.bg_ails_budget_floor_s == 60.0
    assert loaded.bg_ails_budget_margin == 1.0
    assert loaded.bg_ails_wall_model_scale == 0.976
    assert loaded.bg_ails_wall_model_wave_exponent == -0.180
    assert loaded.subcluster_rate_s_per_customer == 0.06
    assert loaded.subcluster_floor_s == 5.0
    assert loaded.decomp_k_base_arms == [2, 3, 4, 6, 8, 10, 12]
    assert loaded.decomp_k_ext_per_1000 == 4
    assert loaded.decomp_k_min_routes_per_cluster == 8
    assert loaded.decomp_k_min_arm_spacing == 2


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


def test_repo_default_yaml_exposes_seed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_config(repo_root / "configs" / "default.yaml")
    assert loaded.seed == 42
    assert loaded.max_no_improve == 1000


def test_repo_async_example_yaml_is_six_plus_one_plus_one() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    loaded = load_config(repo_root / "configs" / "async_example.yaml")
    assert loaded.cores_total == 8
    assert loaded.cores_dri == 6
    assert loaded.cores_bg == 1
    assert loaded.cores_sp == 1
    assert loaded.bg_ails_mode == "async"
    assert loaded.sp_sc_mode == "async"
    assert loaded.sp_sc_trigger == "wallclock"
    assert loaded.bg_ails_pair_selection == "greedy"
    assert loaded.bg_ails_n_chains_mode == "k"
    assert loaded.bg_ails_initial_omega == 10.0
    assert loaded.bg_ails_budget_floor_s == 60.0


def test_load_rejects_removed_k_domain_keys(tmp_path: Path) -> None:
    for key in ("s_hi", "imbalance_c", "imbalance_beta", "k_lo"):
        path = tmp_path / "cfg.yaml"
        path.write_text(f"decomposition:\n  k_domain:\n    {key}: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="was removed"):
            load_config(path)


def test_load_rejects_removed_wall_model_keys(tmp_path: Path) -> None:
    for key in ("wall_model_slope", "wall_model_intercept"):
        path = tmp_path / "cfg.yaml"
        path.write_text(f"bg_ails:\n  budget:\n    {key}: 1.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="was removed"):
            load_config(path)


def test_load_rejects_non_predicted_dr_wall_budget_mode(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("bg_ails:\n  budget:\n    mode: lockstep\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predicted_dr_wall"):
        load_config(path)


def test_confirmation_profiles_differ_only_in_k_domain() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    configs = repo_root / "configs"
    cell0 = load_config(configs / "pilot_cell0prime.yaml")
    cell_c4 = load_config(configs / "pilot_cellC4.yaml")
    cell_c6 = load_config(configs / "pilot_cellC6.yaml")
    cell_thin = load_config(configs / "pilot_cellC6thin.yaml")
    assert cell0.decomp_k_base_arms == [1, 2, 3, 4, 6, 8, 10, 12]
    assert cell0.decomp_k_ext_per_1000 == 0
    assert cell_c4.decomp_k_base_arms == [2, 3, 4, 6, 8, 10, 12]
    assert cell_c4.decomp_k_ext_per_1000 == 4
    assert cell_c6.decomp_k_base_arms == [2, 3, 4, 6, 8, 10, 12]
    assert cell_c6.decomp_k_ext_per_1000 == 6
    assert cell_thin.decomp_k_base_arms == [2, 4, 8, 12]
    assert cell_thin.decomp_k_ext_per_1000 == 6
    for loaded in (cell0, cell_c4, cell_c6, cell_thin):
        assert loaded.bg_ails_budget_margin == 1.0
        assert loaded.cores_dri == 6
        assert loaded.bg_ails_mode == "async"
        assert loaded.sp_sc_mode == "async"
        assert loaded.decomp_k_min_routes_per_cluster == 8
        assert loaded.decomp_k_min_arm_spacing == 2


def test_build_haos_config_requires_k_domain_fields() -> None:
    with pytest.raises(ValueError, match="min_routes_per_cluster is required"):
        _build_haos_config(DRISPIConfig())
    with pytest.raises(ValueError, match="min_arm_spacing is required"):
        _build_haos_config(DRISPIConfig(decomp_k_min_routes_per_cluster=8))


def test_default_yaml_k_domain_reaches_compute_k_values() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_config(repo_root / "configs" / "default.yaml")
    ignored = [
        str(w.message)
        for w in caught
        if "Unknown decomposition.k_domain key ignored" in str(w.message)
    ]
    assert not ignored
    assert loaded.decomp_k_min_routes_per_cluster == 8
    assert loaded.decomp_k_min_arm_spacing == 2

    haos_config = _build_haos_config(loaded)
    expected = {
        "XL-n9571-k55": [2, 3, 4, 6, 8, 10, 12],
        "XL-n5902-k122": [2, 3, 4, 6, 8, 10, 12, 14],
        "XL-n4340-k148": [2, 3, 4, 6, 8, 10, 12, 15],
        "XL-n9784-k2774": [2, 3, 4, 6, 8, 10, 12, 18, 27, 40],
    }
    for name, domain in expected.items():
        instance = CVRPInstance(
            name=name,
            n_customers=1,
            capacity=100,
            depot=(0.0, 0.0),
            customers=[2],
            coordinates={1: (0.0, 0.0), 2: (1.0, 0.0)},
            demands={1: 0, 2: 1},
        )
        assert haos_config.compute_k_values(instance).domain == domain, name
