"""Tests for the post-run analysis module (Phases 9c/9d)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from drispi.pipeline.analysis import (
    HAOS_LEVELS,
    filter_by_type,
    generate_summary_report,
    get_haos_config_from_jsonl,
    load_jsonl,
    plot_cost_trajectory,
    plot_haos_weights,
    plot_improvement_sources,
    plot_pool_size_over_time,
    plot_runtime_split,
    run_auto_analysis,
)

HAOS_CONFIG = {
    "decay": 0.9,
    "haos_warmup": 2,
    "k_candidates": [2, 4, 8],
    "lambda_demand_values": [0.0, 0.5, 1.0],
    "paradigm_values": ["vertex", "route"],
    "vertex_method_values": ["kmeans", "fcm"],
    "route_method_values": ["kmeans", "agglomerative_avg"],
    "solver_values": ["pyvrp", "filo"],
}

_LEVEL_VALUES = {
    "level_1_k": [2, 4, 8],
    "level_2_lambda_demand": [0.0, 0.5, 1.0],
    "level_3_paradigm": ["vertex", "route"],
    "level_4a_vertex_method": ["kmeans", "fcm"],
    "level_4b_route_method": ["kmeans", "agglomerative_avg"],
    "level_5_solver": ["pyvrp", "filo"],
}


def _levels_snapshot() -> dict:
    levels = {}
    for key, values in _LEVEL_VALUES.items():
        n = len(values)
        levels[key] = {
            "values": list(values),
            "raw_weights": [1.0] * n,
            "probabilities": [1.0 / n] * n,
            "min_weight": 0.05,
        }
    return levels


def _synthetic_lines(n_iterations: int = 4) -> list[dict]:
    lines: list[dict] = [
        {
            "type": "init",
            "instance": "test-instance",
            "n_customers": 100,
            "config": {"time_limit": 60.0, "haos_config": HAOS_CONFIG},
            "bks": 1000.0,
        }
    ]
    for i in range(n_iterations):
        lines.append(
            {
                "type": "haos_roll",
                "iteration": i,
                "k": [2, 4, 8][i % 3],
                "lambda_demand": 0.5,
                "paradigm": "vb",
                "method": "kmeans",
                "solver": "pyvrp",
                "joint_prob": 0.01,
                "is_spsc": i % 2 == 1,
            }
        )
        lines.append(
            {
                "type": "phase_done",
                "iteration": i,
                "phase_num": 1,
                "total_phases": 5,
                "phase_name": "decompose",
                "elapsed": 0.5,
                "budget": None,
                "load_pct": None,
                "operator_info": {"lambda_demand": 0.5, "paradigm": "vertex", "method": "kmeans"},
            }
        )
        lines.append(
            {
                "type": "phase_done",
                "iteration": i,
                "phase_num": 2,
                "total_phases": 5,
                "phase_name": "route",
                "elapsed": 3.0,
                "budget": 10.0,
                "load_pct": 30.0,
                "operator_info": {"solver": "pyvrp", "k": [2, 4, 8][i % 3]},
                "cluster_sizes": [40, 35, 25],
            }
        )
        lines.append(
            {
                "type": "phase_done",
                "iteration": i,
                "phase_num": 3,
                "total_phases": 5,
                "phase_name": "bg_ails",
                "elapsed": 5.0,
                "budget": 60.0,
                "load_pct": 8.3,
                "operator_info": None,
                "bg_cost_before": 1200.0 - 10 * i,
                "bg_cost_after": 1150.0 - 10 * i,
            }
        )
        if i % 2 == 1:
            lines.append(
                {
                    "type": "phase_done",
                    "iteration": i,
                    "phase_num": 4,
                    "total_phases": 5,
                    "phase_name": "sp_sc",
                    "elapsed": 20.0 + 100.0 * (i % 2),
                    "budget": 100.0,
                    "load_pct": 20.0,
                    "operator_info": {"sc_or_sp": "SC" if i < 3 else "SP", "pool_size": 50},
                    "lp_fractionality": 0.3 + 0.1 * i,
                }
            )
            lines.append(
                {
                    "type": "phase_done",
                    "iteration": i,
                    "phase_num": 5,
                    "total_phases": 5,
                    "phase_name": "standard_ails",
                    "elapsed": 8.0,
                    "budget": 90.0,
                    "load_pct": 8.9,
                    "operator_info": None,
                }
            )
        if i in (1, 2):
            lines.append(
                {
                    "type": "improve",
                    "iteration": i,
                    "cost": 1100.0 - 50 * i,
                    "phase_name": "bg_ails" if i == 1 else "standard_ails",
                    "gap_to_bks": 5.0,
                }
            )
        lines.append(
            {
                "type": "summary",
                "iteration": i,
                "iter_cost": 1150.0 - 10 * i,
                "best_cost": 1100.0 - 10 * i,
                "delta_to_best_pct": 1.0,
                "delta_to_bks_pct": 10.0,
                "delta_best_to_bks_pct": 8.0,
                "no_improve": 0,
                "pool_size": 30 + 5 * i,
                "pool_diversity_avg": 0.6 + 0.05 * i,
                "pool_quality_avg": 0.4 + 0.02 * i,
                "duplicates_rejected": i,
                "duplicates_replaced": 1,
            }
        )
        roll_with_levels = {
            "type": "haos_roll",
            "iteration": i,
            "k": [2, 4, 8][i % 3],
            "lambda_demand": 0.5,
            "paradigm": "vb",
            "method": "kmeans",
            "solver": "pyvrp",
            "joint_prob": 0.01,
            "is_spsc": i % 2 == 1,
            "levels": _levels_snapshot(),
        }
        lines.append(roll_with_levels)
    lines.append(
        {
            "type": "final",
            "iterations": n_iterations,
            "elapsed": 120.0,
            "best_cost": 1100.0 - 10 * (n_iterations - 1),
            "stopped_by_no_improve": False,
            "stopped_by_time": True,
            "pool_tag_iterations": [0, 0, 1, 2, 3],
        }
    )
    return lines


@pytest.fixture
def synthetic_lines() -> list[dict]:
    return _synthetic_lines()


@pytest.fixture
def run_dir(tmp_path: Path, synthetic_lines: list[dict]) -> Path:
    jsonl = tmp_path / "run.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for line in synthetic_lines:
            fh.write(json.dumps(line) + "\n")
    return tmp_path


def test_load_jsonl_parses_valid_and_skips_malformed(tmp_path: Path) -> None:
    jsonl = tmp_path / "run.jsonl"
    jsonl.write_text(
        '{"type":"init","instance":"x"}\n'
        "not json at all\n"
        '{"type":"summary","iteration":0}\n'
        '{"broken": \n'
        "\n",
        encoding="utf-8",
    )
    lines = load_jsonl(tmp_path)
    assert len(lines) == 2
    assert lines[0]["type"] == "init"
    assert lines[1]["type"] == "summary"


def test_load_jsonl_missing_file(tmp_path: Path) -> None:
    assert load_jsonl(tmp_path) == []


def test_filter_by_type(synthetic_lines: list[dict]) -> None:
    summaries = filter_by_type(synthetic_lines, "summary")
    assert len(summaries) == 4
    assert all(line["type"] == "summary" for line in summaries)
    assert filter_by_type(synthetic_lines, "does_not_exist") == []


def test_get_haos_config_from_jsonl(synthetic_lines: list[dict]) -> None:
    cfg = get_haos_config_from_jsonl(synthetic_lines)
    assert cfg is not None
    assert cfg["k_candidates"] == [2, 4, 8]
    assert get_haos_config_from_jsonl([]) is None


def test_plot_haos_weights_creates_six_pngs(
    synthetic_lines: list[dict], tmp_path: Path
) -> None:
    plot_haos_weights(synthetic_lines, HAOS_CONFIG, tmp_path)
    expected = {
        "haos_l1_k.png",
        "haos_l2_lambda.png",
        "haos_l3_paradigm.png",
        "haos_l4a_vertex_method.png",
        "haos_l4b_route_method.png",
        "haos_l5_solver.png",
    }
    created = {p.name for p in tmp_path.glob("*.png")}
    assert expected == created


def test_plot_cost_trajectory_creates_png(
    synthetic_lines: list[dict], tmp_path: Path
) -> None:
    plot_cost_trajectory(synthetic_lines, 1000.0, tmp_path)
    assert (tmp_path / "cost_trajectory.png").is_file()


def test_plot_cost_trajectory_without_bks(
    synthetic_lines: list[dict], tmp_path: Path
) -> None:
    plot_cost_trajectory(synthetic_lines, None, tmp_path)
    assert (tmp_path / "cost_trajectory.png").is_file()


def test_plot_runtime_split_creates_png(
    synthetic_lines: list[dict], tmp_path: Path
) -> None:
    plot_runtime_split(synthetic_lines, tmp_path)
    assert (tmp_path / "runtime_split.png").is_file()


def test_plot_improvement_sources_creates_png(
    synthetic_lines: list[dict], tmp_path: Path
) -> None:
    plot_improvement_sources(synthetic_lines, tmp_path)
    assert (tmp_path / "improvement_sources.png").is_file()


def test_plot_pool_size_over_time_creates_png(
    synthetic_lines: list[dict], tmp_path: Path
) -> None:
    plot_pool_size_over_time(synthetic_lines, tmp_path)
    assert (tmp_path / "pool_size_over_time.png").is_file()


def test_generate_summary_report_contains_haos_imgs(
    synthetic_lines: list[dict], tmp_path: Path
) -> None:
    generate_summary_report(tmp_path, synthetic_lines, 1000.0)
    report = tmp_path / "analysis" / "summary_report.html"
    assert report.is_file()
    html = report.read_text(encoding="utf-8")
    for _, _, _, filename in HAOS_LEVELS:
        assert f'<img src="{filename}"' in html


def test_run_auto_analysis_creates_all_files(run_dir: Path) -> None:
    lines = load_jsonl(run_dir)
    run_auto_analysis(run_dir, lines, 1000.0)
    analysis_dir = run_dir / "analysis"
    assert analysis_dir.is_dir()
    expected = {
        "haos_l1_k.png",
        "haos_l2_lambda.png",
        "haos_l3_paradigm.png",
        "haos_l4a_vertex_method.png",
        "haos_l4b_route_method.png",
        "haos_l5_solver.png",
        "cost_trajectory.png",
        "runtime_split.png",
        "improvement_sources.png",
        "pool_size_over_time.png",
        "summary_report.html",
    }
    created = {p.name for p in analysis_dir.iterdir()}
    assert expected <= created


def test_run_auto_analysis_survives_chart_failure(
    run_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = load_jsonl(run_dir)
    with patch(
        "drispi.pipeline.analysis.plot_cost_trajectory",
        side_effect=RuntimeError("boom"),
    ):
        run_auto_analysis(run_dir, lines, 1000.0)
    analysis_dir = run_dir / "analysis"
    assert not (analysis_dir / "cost_trajectory.png").exists()
    assert (analysis_dir / "haos_l1_k.png").is_file()
    assert (analysis_dir / "runtime_split.png").is_file()
    assert (analysis_dir / "improvement_sources.png").is_file()
    assert (analysis_dir / "pool_size_over_time.png").is_file()
    assert (analysis_dir / "summary_report.html").is_file()
    captured = capsys.readouterr()
    assert "plot_cost_trajectory failed" in captured.err
