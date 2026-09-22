#!/usr/bin/env python3
"""Build the per-instance final-benchmark CSV from a campaign.jsonl.

Column rules were checked against the Dantzig CSV: every cost, iteration,
wall-time, checkpoint, route length, and gap cell matches
``str(round(value, nd))`` applied to campaign.jsonl. Seed standard deviation
is the sample standard deviation of the three unrounded gaps to the current
BKS; seed range is ``(max cost - min cost) / bks_current * 100``. Both are
rounded to 4 decimal places the same way as the gap columns.

Reference columns that do not depend on the run (initial BKS, current BKS,
AILS-II best/mean, k-domain flags) are copied from the Dantzig CSV. The
k-domain logged at init is identical across the two campaigns. Lagrange has
one finished run per instance and seed and no contamination rerun, so
``rerun_after_contamination`` is false on every row.

Usage:
    python scripts/build_final_benchmark_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33)
MARKS = (30, 60, 90, 120)

REFERENCE_COLUMNS = (
    "bks_initial",
    "bks_current",
    "ails2_best",
    "ails2_mean",
)


def _sround(value: float, digits: int) -> str:
    return str(round(value, digits))


def _gap(cost: float, reference: float) -> float:
    return (cost - reference) / reference * 100.0


def _load_campaign(path: Path) -> dict[str, dict[int, dict]]:
    by_instance: dict[str, dict[int, dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("status") != "ok":
            raise RuntimeError(f"Non-ok campaign row: {event.get('instance')} seed {event.get('seed')}")
        stem = event["instance_stem"]
        seed = int(event["seed"])
        slot = by_instance.setdefault(stem, {})
        if seed in slot:
            raise RuntimeError(f"Duplicate campaign row for {stem} seed {seed}")
        slot[seed] = event
    return by_instance


def _load_reference(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["instance"]: row for row in csv.DictReader(fh)}


def _header(seeds: tuple[int, ...]) -> list[str]:
    seed_cost = [f"cost_seed{s}" for s in seeds]
    seed_iters = [f"iters_seed{s}" for s in seeds]
    seed_wall = [f"wall_s_seed{s}" for s in seeds]
    gap_init = [f"gap_init_seed{s}" for s in seeds]
    gap_bks = [f"gap_bks_seed{s}" for s in seeds]
    marks = [f"mark{mark}_seed{s}" for mark in MARKS for s in seeds]
    return [
        "instance",
        "n",
        "K",
        "route_len",
        *seed_cost,
        "cost_best3",
        "cost_mean3",
        "cost_worst3",
        *seed_iters,
        *seed_wall,
        *REFERENCE_COLUMNS,
        *gap_init,
        "gap_init_best3",
        "gap_init_mean3",
        *gap_bks,
        "gap_bks_best3",
        "gap_bks_mean3",
        "gap_init_ails2_mean",
        "gap_bks_ails2_mean",
        "seed_sd_pp",
        "seed_range_pp",
        "beats_initial_bks",
        "ties_initial_bks",
        "beats_ails2_mean",
        "kdomain_changed",
        "kdomain_group",
        "rerun_after_contamination",
        *marks,
    ]


def build_rows(
    campaign: dict[str, dict[int, dict]],
    reference: dict[str, dict[str, str]],
    seeds: tuple[int, ...],
    *,
    contaminated: set[str] | None = None,
) -> list[dict[str, str]]:
    missing = set(reference) ^ set(campaign)
    if missing:
        raise RuntimeError(f"Instance mismatch between campaign and reference CSV: {sorted(missing)}")

    rows: list[dict[str, str]] = []
    for instance in sorted(reference, key=lambda name: int(reference[name]["n"])):
        ref = reference[instance]
        runs = campaign[instance]
        if set(runs) != set(seeds):
            raise RuntimeError(f"{instance}: expected seeds {seeds}, found {sorted(runs)}")

        n = int(ref["n"])
        k = int(ref["K"])
        bks_initial = float(ref["bks_initial"])
        bks_current = float(ref["bks_current"])
        ails2_mean = float(ref["ails2_mean"])
        costs = [float(runs[seed]["final_best_cost"]) for seed in seeds]
        if any(cost != int(cost) for cost in costs):
            raise RuntimeError(f"{instance}: non-integer final cost {costs}")

        for seed, cost in zip(seeds, costs, strict=True):
            reported = runs[seed]["gap_to_bks"]
            recomputed = _gap(cost, bks_current)
            if reported != recomputed:
                raise RuntimeError(
                    f"{instance} seed {seed}: campaign gap {reported} != recomputed {recomputed}"
                )

        best = min(costs)
        worst = max(costs)
        mean_cost = sum(costs) / len(costs)
        raw_bks_gaps = [_gap(cost, bks_current) for cost in costs]

        row: dict[str, str] = {
            "instance": instance,
            "n": str(n),
            "K": str(k),
            "route_len": _sround((n - 1) / k, 3),
            "cost_best3": str(int(best)),
            "cost_mean3": _sround(mean_cost, 1),
            "cost_worst3": str(int(worst)),
            "bks_initial": ref["bks_initial"],
            "bks_current": ref["bks_current"],
            "ails2_best": ref["ails2_best"],
            "ails2_mean": ref["ails2_mean"],
            "gap_init_best3": _sround(_gap(best, bks_initial), 4),
            "gap_init_mean3": _sround(_gap(mean_cost, bks_initial), 4),
            "gap_bks_best3": _sround(_gap(best, bks_current), 4),
            "gap_bks_mean3": _sround(_gap(mean_cost, bks_current), 4),
            "gap_init_ails2_mean": _sround(_gap(ails2_mean, bks_initial), 4),
            "gap_bks_ails2_mean": _sround(_gap(ails2_mean, bks_current), 4),
            "seed_sd_pp": _sround(statistics.stdev(raw_bks_gaps), 4),
            "seed_range_pp": _sround(_gap(worst, bks_current) - _gap(best, bks_current), 4),
            "beats_initial_bks": str(best < bks_initial),
            "ties_initial_bks": str(best == bks_initial),
            "beats_ails2_mean": str(best < ails2_mean),
            "kdomain_changed": ref["kdomain_changed"],
            "kdomain_group": ref["kdomain_group"],
            "rerun_after_contamination": str(instance in (contaminated or set())),
        }
        for seed, cost in zip(seeds, costs, strict=True):
            run = runs[seed]
            row[f"cost_seed{seed}"] = str(int(cost))
            row[f"iters_seed{seed}"] = str(int(run["iterations"]))
            row[f"wall_s_seed{seed}"] = _sround(float(run["wall_clock_s"]), 1)
            row[f"gap_init_seed{seed}"] = _sround(_gap(cost, bks_initial), 4)
            row[f"gap_bks_seed{seed}"] = _sround(_gap(cost, bks_current), 4)
            marks = {int(mark["mark_min"]): int(mark["best_cost"]) for mark in run["wall_marks"]}
            if set(marks) != set(MARKS):
                raise RuntimeError(f"{instance} seed {seed}: marks {sorted(marks)} != {MARKS}")
            for mark in MARKS:
                row[f"mark{mark}_seed{seed}"] = str(marks[mark])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], seeds: tuple[int, ...]) -> None:
    header = _header(seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "data/results/lagrange_benchmark/campaign.jsonl",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "data/results/finalBenchmarkResults_dantzig.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/results/finalBenchmarkResults_lagrange.csv",
    )
    parser.add_argument("--seeds", type=int, nargs=3, default=SEEDS)
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    campaign = _load_campaign(args.campaign)
    reference = _load_reference(args.reference)
    rows = build_rows(campaign, reference, seeds)
    write_csv(args.output, rows, seeds)


if __name__ == "__main__":
    main()
