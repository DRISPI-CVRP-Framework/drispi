"""Unit tests for cores parsing and resolution."""

from __future__ import annotations

import pytest

from drispi.pipeline.cores import parse_cpu_list, resolve_cores


def test_parse_cpu_list_range() -> None:
    assert parse_cpu_list("0-7") == list(range(8))
    assert parse_cpu_list("0,2,4") == [0, 2, 4]
    assert parse_cpu_list("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11]


def test_resolve_legacy_n_workers() -> None:
    cfg = resolve_cores(
        cores_total=None,
        cores_dri=None,
        cores_sp=None,
        cores_cpu_list=None,
        n_workers=8,
    )
    assert not cfg.from_cores_block
    assert cfg.dri == 8
    assert cfg.sp == 0
    assert cfg.cpu_list is None


def test_resolve_cores_block_assert() -> None:
    with pytest.raises(AssertionError, match="exceeds"):
        resolve_cores(
            cores_total=8,
            cores_dri=7,
            cores_sp=2,
            cores_cpu_list=None,
            n_workers=8,
        )


def test_resolve_cores_block_idle_warning(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        cfg = resolve_cores(
            cores_total=8,
            cores_dri=6,
            cores_sp=1,
            cores_cpu_list=list(range(8)),
            n_workers=8,
        )
    assert cfg.dri == 6
    assert cfg.sp == 1
    assert cfg.idle_cpus == [7]
    assert any("idle" in r.message.lower() or "remainder" in r.message.lower() for r in caplog.records)


def test_resolve_cores_cli_cpus() -> None:
    cfg = resolve_cores(
        cores_total=4,
        cores_dri=3,
        cores_sp=1,
        cores_cpu_list=None,
        n_workers=8,
        cli_cpus="4-7",
    )
    assert cfg.cpu_list == [4, 5, 6, 7]
    assert cfg.dri_cpus == [4, 5, 6]
    assert cfg.sp_cpus == [7]


def test_resolve_cores_6_1_1_split_cpu_order_dri_bg_sp() -> None:
    cfg = resolve_cores(
        cores_total=8,
        cores_dri=6,
        cores_bg=1,
        cores_sp=1,
        cores_cpu_list=list(range(8)),
        n_workers=8,
    )
    assert (cfg.dri, cfg.bg, cfg.sp) == (6, 1, 1)
    assert cfg.dri_cpus == [0, 1, 2, 3, 4, 5]
    assert cfg.bg_cpus == [6]
    assert cfg.sp_cpus == [7]
    assert cfg.idle_cpus == []


def test_resolve_cores_bg_defaults_to_zero() -> None:
    cfg = resolve_cores(
        cores_total=8,
        cores_dri=7,
        cores_sp=1,
        cores_cpu_list=list(range(8)),
        n_workers=8,
    )
    assert cfg.bg == 0
    assert cfg.bg_cpus == []
    assert cfg.sp_cpus == [7]


def test_resolve_cores_bg_counts_against_total() -> None:
    with pytest.raises(AssertionError, match="exceeds"):
        resolve_cores(
            cores_total=8,
            cores_dri=7,
            cores_bg=1,
            cores_sp=1,
            cores_cpu_list=None,
            n_workers=8,
        )


def test_resolve_cores_bg_alone_requires_full_block() -> None:
    with pytest.raises(ValueError, match="requires all"):
        resolve_cores(
            cores_total=None,
            cores_dri=None,
            cores_bg=1,
            cores_sp=None,
            cores_cpu_list=None,
            n_workers=8,
        )
