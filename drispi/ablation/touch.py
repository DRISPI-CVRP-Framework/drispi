"""Read AILS-II touch dumps; map Node.name → DRISPI global customer ID."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from drispi.core.instance import CVRPInstance


def parse_touch_tsv(path: Path) -> dict[str, Any]:
    """Parse ``<outpath>.touch.tsv``. Header lines start with ``#``."""
    header: dict[str, str] = {}
    names: list[int] = []
    evals: list[int] = []
    accepts: list[int] = []
    perturbs: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            key, _, rest = body.partition(" ")
            header[key] = rest
            continue
        if line.startswith("name"):
            continue
        parts = line.split("\t")
        names.append(int(parts[0]))
        evals.append(int(parts[1]))
        accepts.append(int(parts[2]))
        perturbs.append(int(parts[3]))
    return {
        "header": header,
        "name": np.asarray(names, dtype=np.int64),
        "eval": np.asarray(evals, dtype=np.int64),
        "accept": np.asarray(accepts, dtype=np.int64),
        "perturb": np.asarray(perturbs, dtype=np.int64),
    }


def to_global_ids(internal_names: np.ndarray) -> np.ndarray:
    """Dump internal ``Node.name`` m → DRISPI customer ID m+1 (depot 0 → 1)."""
    return internal_names.astype(np.int64) + 1


def customer_touch_maps(
    dump: dict[str, Any],
    instance: CVRPInstance,
) -> dict[str, dict[int, int]]:
    """Per-customer eval/accept/perturb counts keyed by global ID (depot dropped)."""
    globals_ids = to_global_ids(dump["name"])
    out: dict[str, dict[int, int]] = {"eval": {}, "accept": {}, "perturb": {}}
    customers = set(instance.customers)
    for i, gid in enumerate(globals_ids.tolist()):
        if gid not in customers:
            continue
        out["eval"][gid] = int(dump["eval"][i])
        out["accept"][gid] = int(dump["accept"][i])
        out["perturb"][gid] = int(dump["perturb"][i])
    return out
