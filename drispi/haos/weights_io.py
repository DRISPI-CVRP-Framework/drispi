from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from drispi.haos.haos import HAOS


def save_weights(
    haos: HAOS,
    path: Path,
    instance_name: str,
    iterations_completed: int,
) -> None:
    payload = {
        "instance": instance_name,
        "iterations_completed": iterations_completed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "levels": haos.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_weights(haos: HAOS, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    levels = data.get("levels")
    if not isinstance(levels, dict):
        raise ValueError("Invalid weights file: missing/invalid 'levels'")
    haos.load_state_dict(levels)
