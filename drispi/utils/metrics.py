"""Benchmark and solution quality metrics."""

from __future__ import annotations

import json
from pathlib import Path


def read_bks(bks_path: Path) -> dict[str, float]:
    """Read BKS JSON file and return instance_name -> best known cost."""
    if not bks_path.exists():
        raise FileNotFoundError(f"BKS file not found: {bks_path}")
    try:
        raw = json.loads(bks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed BKS JSON at {bks_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Malformed BKS JSON at {bks_path}: expected a JSON object")
    return {str(name): float(cost) for name, cost in raw.items()}


def gap_to_bks(cost: float, instance_name: str, bks: dict[str, float]) -> float:
    """Compute percentage gap to BKS for an instance."""
    if instance_name not in bks:
        raise KeyError(f"Instance '{instance_name}' not found in BKS dictionary")
    bks_cost = bks[instance_name]
    return ((cost - bks_cost) / bks_cost) * 100.0
