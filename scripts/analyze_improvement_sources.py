#!/usr/bin/env python3
"""Attribute incumbent updates, cost reduction, and stage wall time.

Cost reduction is measured from each run's first logged solution to its final
incumbent. The first solution is an incumbent update but not part of that
reduction. Stage wall time is the sum of phase_done elapsed fields; asynchronous
stages overlap, so the shares are of recorded stage time, not of the 7,200 s budget.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "data/results/lagrange_benchmark"
CHARS = ROOT / "data/results/instance_chars.csv"
CAMPAIGN = ROOT / "data/results/finalBenchmarkResults_lagrange.csv"
OUT = ROOT / "data/results/improvement_sources_lagrange.json"

STAGE_OF = {
    "decompose": "decompose_route",
    "route": "decompose_route",
    "decompose_route": "decompose_route",
    "bg_ails": "bg_ails",
    "sp_sc": "sp_sc",
    "sp": "sp_sc",
    "sc": "sp_sc",
    "standard_ails": "standard_ails",
    "standard_improvement": "standard_ails",
    "improve": "standard_ails",
}


def _stage(name: str) -> str:
    return STAGE_OF.get(name, name)


def _runs() -> list[Path]:
    paths = sorted(LOG_ROOT.glob("*/seed*/**/run.jsonl"))
    if len(paths) != 300:
        raise SystemExit(f"expected 300 run logs, found {len(paths)}")
    return paths


def _instance(path: Path) -> str:
    return "XL-" + path.parents[2].name[2:]


def main() -> None:
    chars = pd.read_csv(CHARS)
    campaign = pd.read_csv(CAMPAIGN)
    meta = chars.merge(campaign[["instance", "n"]], on="instance")
    meta["long"] = meta.r_tab >= 50
    meta["large"] = meta.n > 3400
    meta = meta.set_index("instance")

    per_run = []
    unknown_phases = set()
    for path in _runs():
        instance = _instance(path)
        updates = defaultdict(int)
        reduction = defaultdict(float)
        wall = defaultdict(float)
        best = None
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if '"type":"improve"' not in line and '"type":"phase_done"' not in line:
                    continue
                event = json.loads(line)
                kind = event.get("type")
                if kind == "improve":
                    phase = event["phase_name"]
                    if phase not in STAGE_OF:
                        unknown_phases.add(("improve", phase))
                    stage = _stage(phase)
                    cost = float(event["cost"])
                    updates[stage] += 1
                    if best is None:
                        best = cost
                    elif cost < best:
                        reduction[stage] += best - cost
                        best = cost
                elif kind == "phase_done":
                    phase = event.get("phase_name")
                    if phase not in STAGE_OF:
                        unknown_phases.add(("phase_done", phase))
                    elapsed = event.get("elapsed")
                    if elapsed is None:
                        continue
                    wall[_stage(phase)] += float(elapsed)
        per_run.append(
            {
                "instance": instance,
                "updates": dict(updates),
                "reduction": dict(reduction),
                "wall": dict(wall),
            }
        )

    stages = ["decompose_route", "bg_ails", "sp_sc", "standard_ails"]

    def accumulate(rows: list[dict]) -> dict:
        updates = defaultdict(int)
        reduction = defaultdict(float)
        wall = defaultdict(float)
        for row in rows:
            for stage, value in row["updates"].items():
                updates[stage] += value
            for stage, value in row["reduction"].items():
                reduction[stage] += value
            for stage, value in row["wall"].items():
                wall[stage] += value
        u_tot = sum(updates.values()) or 1
        r_tot = sum(reduction.values()) or 1
        w_tot = sum(wall.values()) or 1
        out = {}
        for stage in stages:
            out[stage] = {
                "updates": int(updates[stage]),
                "update_share": updates[stage] / u_tot,
                "reduction": reduction[stage],
                "reduction_share": reduction[stage] / r_tot,
                "wall_s": wall[stage],
                "wall_share": wall[stage] / w_tot,
            }
        out["totals"] = {
            "updates": int(sum(updates.values())),
            "reduction": float(sum(reduction.values())),
            "wall_s": float(sum(wall.values())),
        }
        return out

    groups = {
        "all": per_run,
        "r_lt_50": [row for row in per_run if not meta.loc[row["instance"], "long"]],
        "r_ge_50": [row for row in per_run if meta.loc[row["instance"], "long"]],
        "n_lt_3400": [row for row in per_run if not meta.loc[row["instance"], "large"]],
        "n_gt_3400": [row for row in per_run if meta.loc[row["instance"], "large"]],
    }
    payload = {
        "unknown_phases": sorted(f"{kind}:{name}" for kind, name in unknown_phases),
        "groups": {name: accumulate(rows) for name, rows in groups.items()},
        "n_runs": {name: len(rows) for name, rows in groups.items()},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print("unknown", payload["unknown_phases"])
    for name, group in payload["groups"].items():
        print(f"\n{name} n={payload['n_runs'][name]} updates={group['totals']['updates']}")
        for stage in stages:
            row = group[stage]
            print(
                f"  {stage:16} upd {row['updates']:5d} ({row['update_share']:.3f})  "
                f"red {row['reduction_share']:.3f}  wall {row['wall_share']:.3f}"
            )


if __name__ == "__main__":
    main()
