"""YAML load/save and CLI merge helpers for DRISPIConfig."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import yaml

from drispi.pipeline.config import DRISPIConfig

LOGGER = logging.getLogger(__name__)

_SECTION_FIELDS: dict[str, list[str]] = {
    "stopping": ["time_limit", "max_no_improve"],
    "parallelism": ["n_workers"],
    "sp_sc": [
        "warmup_iterations",
        "sp_interval",
        "min_coverage",
        "sp_time_limit",
        "mip_gap",
    ],
    "subcluster": ["subcluster_time_per_customer"],
    "bg_ails": [
        "bg_ails_time_limit",
        "bg_ails_initial_omega",
        "bg_ails_boundary_threshold",
        "bg_ails_small_cluster_cap",
        "bg_ails_small_cluster_alpha",
    ],
    "standard_improvement": ["standard_improvement_time_limit"],
    "pool": ["max_pool_size", "pool_diversity_weight"],
    "haos": [
        "haos_decay",
        "haos_warmup",
        "haos_starting_weight",
        "haos_min_weight_k",
        "haos_min_weight_lambda",
        "haos_min_weight_paradigm",
        "haos_min_weight_vertex_method",
        "haos_min_weight_route_method",
        "haos_min_weight_solver",
        "haos_k_candidates",
        "haos_lambda_demand_values",
        "haos_paradigm_values",
        "haos_vertex_method_values",
        "haos_route_method_values",
        "haos_solver_values",
    ],
    "haos_rewards": [
        "haos_reward_new_best",
        "haos_reward_improvement",
        "haos_reward_no_improvement",
        "haos_reward_no_solution",
        "haos_deferred_new_best",
        "haos_deferred_improvement",
        "haos_deferred_no_improvement",
    ],
    "seed": ["seed"],
    "output": ["output_dir", "run_analysis"],
}

_SECTION_COMMENTS: dict[str, str] = {
    "stopping": "Stopping criteria",
    "parallelism": "Parallelism",
    "sp_sc": "SP/SC scheduling",
    "subcluster": "Subcluster solver",
    "bg_ails": "BG-AILS",
    "standard_improvement": "Standard improvement",
    "pool": "Route pool",
    "haos": "HAOS",
    "haos_rewards": "HAOS rewards",
    "seed": "Seed (all sub-seeds derived deterministically from this)",
    "output": "Output",
}

_VALID_FIELDS = {f.name for f in fields(DRISPIConfig)}

DEFAULT_CONFIG_FILENAME = "default.yaml"


def _read_yaml_mapping(yaml_path: Path) -> dict[str, Any]:
    """Read a YAML file and return a flat field -> value mapping."""
    path = Path(yaml_path).expanduser().resolve()
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    overrides: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("#") or key is None:
            continue
        if key not in _VALID_FIELDS:
            warnings.warn(f"Unknown config key ignored: {key!r}", stacklevel=3)
            continue
        overrides[key] = _coerce_value(key, value)
    return overrides


def _resolve_base_config_path(yaml_path: Path) -> Path | None:
    """Return sibling default.yaml for profile configs, or None if not applicable."""
    path = Path(yaml_path).expanduser().resolve()
    if path.name == DEFAULT_CONFIG_FILENAME:
        return None
    base_path = path.parent / DEFAULT_CONFIG_FILENAME
    return base_path if base_path.is_file() else None


def _apply_yaml_mapping(config: DRISPIConfig, overrides: dict[str, Any]) -> DRISPIConfig:
    if not overrides:
        return config
    return replace(config, **overrides)


def _coerce_value(field_name: str, value: Any) -> Any:
    if field_name == "output_dir":
        return Path(value)
    return value


def load_config(yaml_path: Path) -> DRISPIConfig:
    """
    Load DRISPIConfig from a YAML file.

    Profile configs (e.g. ``configs/benchmark.yaml``) merge on top of sibling
    ``default.yaml`` in the same directory when present. Keys missing from both
    files use ``DRISPIConfig`` dataclass defaults. Unknown keys are ignored with
    a warning. List fields replace entirely; path fields become ``Path`` objects.
    """
    path = Path(yaml_path).expanduser().resolve()
    config = DRISPIConfig()

    base_path = _resolve_base_config_path(path)
    if base_path is not None:
        config = _apply_yaml_mapping(config, _read_yaml_mapping(base_path))

    config = _apply_yaml_mapping(config, _read_yaml_mapping(path))
    return config


def config_to_dict(config: DRISPIConfig) -> dict[str, Any]:
    """Convert DRISPIConfig to a nested dict grouped by section."""
    flat = asdict(config)
    out: dict[str, Any] = {}
    for section, keys in _SECTION_FIELDS.items():
        section_dict: dict[str, Any] = {}
        for key in keys:
            value = flat[key]
            if isinstance(value, Path):
                value = str(value)
            section_dict[key] = value
        out[section] = section_dict
    return out


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        inner = ", ".join(_yaml_scalar(v) for v in value)
        return f"[{inner}]"
    return str(value)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def save_config(config: DRISPIConfig, yaml_path: Path) -> None:
    """Save DRISPIConfig to a YAML file with section comments."""
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = asdict(config)
    lines: list[str] = ["# DRISPI Configuration", ""]
    for section, keys in _SECTION_FIELDS.items():
        comment = _SECTION_COMMENTS.get(section, section)
        lines.append(f"# {comment}")
        for key in keys:
            lines.append(f"{key}: {_yaml_scalar(_serialize_value(flat[key]))}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def merge_cli_overrides(config: DRISPIConfig, overrides: dict) -> DRISPIConfig:
    """
    Apply CLI argument overrides to a DRISPIConfig.

    overrides: dict of {field_name: value} for non-None CLI args.
    Returns new DRISPIConfig with overrides applied.
    Raises ValueError for unknown field names.
    """
    unknown = set(overrides) - _VALID_FIELDS
    if unknown:
        raise ValueError(f"Unknown config field(s): {', '.join(sorted(unknown))}")
    coerced = {k: _coerce_value(k, v) for k, v in overrides.items()}
    return replace(config, **coerced)
