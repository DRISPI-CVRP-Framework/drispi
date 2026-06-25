"""Local time for logs, run labels, and dashboard snapshots."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_DEFAULT_TZ = "UTC"


@lru_cache(maxsize=1)
def log_timezone() -> ZoneInfo:
    """Timezone for log timestamps and run directory labels (``DRISPI_TZ`` or ``TZ``)."""
    name = os.environ.get("DRISPI_TZ") or os.environ.get("TZ") or _DEFAULT_TZ
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(_DEFAULT_TZ)


def local_now() -> datetime:
    """Current wall-clock time in :func:`log_timezone` (naive, for ``strftime``)."""
    return datetime.now(log_timezone()).replace(tzinfo=None)
