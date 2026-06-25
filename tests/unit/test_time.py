"""Tests for log timezone helpers."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from drispi.utils import time as time_utils


@pytest.fixture(autouse=True)
def _clear_timezone_cache() -> None:
    time_utils.log_timezone.cache_clear()
    yield
    time_utils.log_timezone.cache_clear()


def test_local_now_uses_drispi_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setenv("DRISPI_TZ", "Europe/Berlin")

    expected = datetime.now(ZoneInfo("Europe/Berlin")).replace(tzinfo=None)
    actual = time_utils.local_now()
    assert abs((actual - expected).total_seconds()) < 2.0
    assert time_utils.log_timezone() == ZoneInfo("Europe/Berlin")


def test_local_now_falls_back_to_tz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRISPI_TZ", raising=False)
    monkeypatch.setenv("TZ", "Europe/Berlin")

    assert time_utils.log_timezone() == ZoneInfo("Europe/Berlin")


def test_local_now_defaults_to_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRISPI_TZ", raising=False)
    monkeypatch.delenv("TZ", raising=False)

    assert time_utils.log_timezone() == ZoneInfo("UTC")


def test_invalid_timezone_falls_back_to_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRISPI_TZ", "Not/A/Timezone")
    monkeypatch.delenv("TZ", raising=False)

    assert time_utils.log_timezone() == ZoneInfo("UTC")
