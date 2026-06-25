"""Tests for PipelineLogger structured output."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

from drispi.core.instance import CVRPInstance
from drispi.haos.haos import HAOSSelection
from drispi.haos.tag import HAOSTag
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.logger import (
    PipelineLogger,
    _colorize,
    _format_line,
    _format_method_short,
)
from drispi.route_pool.pool import RoutePool

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TAG_RE = re.compile(r"\| \[(.{9})\] ")


def _minimal_config(tmp_path: Path) -> DRISPIConfig:
    return DRISPIConfig(
        time_limit=60.0,
        max_no_improve=10,
        output_dir=tmp_path,
        seed=1,
    )


def _selection() -> HAOSSelection:
    return HAOSSelection(
        k=3,
        lambda_demand=0.5,
        paradigm="vertex",
        method="kmeans",
        solver="pyvrp",
        k_index=0,
        lambda_index=0,
        paradigm_index=0,
        method_index=0,
        solver_index=0,
    )


def test_iteration_display_starts_at_one(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    logger.log_improve(0, 100.0, "bg_ails")
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert "| i 1 |" in text


def test_haos_roll_uses_spaces_and_short_methods(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    assert _format_method_short("agglomerative_avg") == "ac_avg"
    assert _format_method_short("agglomerative_complete") == "ac_complete"
    assert _format_method_short("agglomerative_single") == "ac_single"
    assert _format_method_short("kmeans") == "kmeans"

    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    selection = HAOSSelection(
        k=10,
        lambda_demand=0.0,
        paradigm="vertex",
        method="agglomerative_avg",
        solver="filo2",
        k_index=0,
        lambda_index=0,
        paradigm_index=0,
        method_index=0,
        solver_index=0,
    )
    logger.log_haos_roll(0, selection, True, 0.000347)
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert "vb  ac_avg  filo2" in text
    assert "·" not in text
    assert "SP/SC ✓" in text
    assert "| i 1 |" in text


def test_phase_names_decompose_and_route(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    logger.log_phase_done(
        0,
        1,
        3,
        "decompose",
        1.47,
        None,
        {"lambda_demand": 0.0, "paradigm": "vertex", "method": "agglomerative_single"},
    )
    logger.log_phase_done(0, 2, 3, "route", 87.43, 66.6, {"solver": "filo2", "k": 10})
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert "decompose" in text
    assert "route  " in text or "| [Phase 2/3] route" in text
    assert "vb ac_single" in text
    assert "dissim+cluster" not in text
    assert "subclusters" not in text


def test_summary_shows_three_gaps(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, bks=31083.0, stream=stream)
    # iter=31712, S*=31338, BKS=31083 → matches iteration-3-style regression
    logger.log_summary(2, iter_cost=31712.0, best_cost=31338.0, last_cost=31712.0, no_improve=1)
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert "Δ_S*=+1.19%" in text
    assert "Δ_BKS=+2.02%" in text
    assert "Δ_S*_BKS=+0.82%" in text


def test_improve_logged_after_phase_done(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    logger.log_phase_done(0, 3, 3, "bg_ails", 60.0, 60.0)
    logger.log_improve(0, 100.0, "bg_ails")
    lines = (logger.run_dir / "run.log").read_text(encoding="utf-8").splitlines()
    phase_idx = next(i for i, line in enumerate(lines) if "bg_ails" in line and "Phase 3/3" in line)
    improve_idx = next(i for i, line in enumerate(lines) if "[ IMPROVE ]" in line)
    assert improve_idx > phase_idx


def test_sp_sc_phase_shows_avg_coverage(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    logger.log_phase_done(
        0,
        4,
        5,
        "sp_sc",
        216.49,
        300.0,
        {
            "sc_or_sp": "SC",
            "pool_size": 2277,
            "avg_coverage": 2.345,
            "min_coverage": 5,
        },
    )
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert "avg_coverage=2.35/5" in text
    assert "routes_selected" not in text
    assert "pool=2277" in text


def test_logger_creates_run_log_and_jsonl(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    logger.log_init(_minimal_config(tmp_path), instance_12)
    assert (logger.run_dir / "run.log").is_file()
    assert (logger.run_dir / "run.jsonl").is_file()


def test_run_log_has_no_ansi(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    logger.log_improve(0, 100.0, "bg_ails")
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert _ANSI_RE.search(text) is None


def test_terminal_has_ansi_for_improve_and_new_bks(
    instance_12: CVRPInstance, tmp_path: Path
) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, bks=200.0, stream=stream)
    logger.log_improve(1, 150.0, "bg_ails")
    logger.log_new_bks(
        1,
        90.0,
        "bg_ails",
        [[2, 3], [4, 5]],
        instance_12,
    )
    terminal = stream.getvalue()
    assert _ANSI_RE.search(terminal) is not None
    assert _colorize("x", "improve") in terminal or "\x1b[34m" in terminal
    assert "\x1b[32m" in terminal


def test_tag_fields_are_nine_chars(instance_12: CVRPInstance, tmp_path: Path) -> None:
    line = _format_line(0, " IMPROVE ", "test")
    match = _TAG_RE.search(line)
    assert match is not None
    assert len(match.group(1)) == 9


def test_log_haos_roll_levels_jsonl_only(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    levels = {"level_1_k": {"values": [1, 2], "probabilities": [0.5, 0.5], "raw_weights": [1.0, 1.0]}}
    logger.log_haos_roll(0, _selection(), False, 0.001, levels=levels)
    terminal = stream.getvalue()
    assert "level_1_k" not in terminal
    log_text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert "level_1_k" not in log_text
    events = [
        json.loads(line)
        for line in (logger.run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["levels"] == levels


def test_jsonl_valid_json_per_line(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    logger.log_init(_minimal_config(tmp_path), instance_12)
    logger.log_haos_roll(0, _selection(), False, 0.001)
    for raw in (logger.run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines():
        if raw.strip():
            json.loads(raw)


def test_log_improve_line_tag(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    logger.log_improve(2, 42.0, "bg_ails")
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert "[ IMPROVE ]" in text or "| [ IMPROVE ]" in text


def test_log_new_bks_writes_sol(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, bks=500.0, stream=stream)
    routes = [[2, 3, 4], [5, 6, 7]]
    logger.log_new_bks(0, 120.0, "standard_ails", routes, instance_12)
    bks_path = logger.run_dir / f"{instance_12.name}_BKS.sol"
    assert bks_path.is_file()


def test_log_final_haos_summaries(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    pool = RoutePool()
    pool.add([2, 3], 10.0, haos_tag=HAOSTag(3, 0.5, "vertex", "kmeans", "pyvrp", 0, False))
    pool.add([4, 5], 11.0, haos_tag=HAOSTag(4, 0.2, "route", "spectral", "filo", 1, False))
    pool.add(
        [6, 7],
        12.0,
        haos_tag=HAOSTag(3, 0.5, "vertex", "kmeans", "pyvrp", 2, True),
    )
    pool.add([8, 9], 13.0)

    best = [[2, 3], [4, 5], [6, 7], [8, 9]]
    logger.log_final(3, 12.5, 40.0, False, True, best, pool, instance_12)
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    assert "BEST SOLUTION — HAOS TAG SUMMARY" in text
    assert "ROUTE POOL — HAOS TAG SUMMARY" in text
    assert "[untagged]" in text
    assert "post-SP/SC improvement" in text


def test_final_summary_sorted_descending(instance_12: CVRPInstance, tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = PipelineLogger(instance_12.name, tmp_path, stream=stream)
    pool = RoutePool()
    tag_a = HAOSTag(1, 0.1, "vertex", "kmeans", "pyvrp", 0, False)
    tag_b = HAOSTag(2, 0.2, "vertex", "spectral", "filo", 0, False)
    pool.add([2, 3], 1.0, haos_tag=tag_a)
    pool.add([4, 5], 1.0, haos_tag=tag_a)
    pool.add([6, 7], 1.0, haos_tag=tag_a)
    pool.add([8, 9], 2.0, haos_tag=tag_b)
    logger.log_final(1, 1.0, 10.0, False, False, [[2, 3]], pool, instance_12)
    text = (logger.run_dir / "run.log").read_text(encoding="utf-8")
    pool_section = text.split("ROUTE POOL — HAOS TAG SUMMARY", 1)[1]
    counts: list[int] = []
    for line in pool_section.splitlines():
        if " routes  " not in line or "[untagged]" in line or "post-SP/SC" in line:
            continue
        payload = line.split("] ", 1)[-1].strip()
        counts.append(int(payload.split(" routes", 1)[0]))
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 3
