#!/usr/bin/env python3
"""Build a consolidated, reusable per-instance solver comparison table.

Combines three sources for all 100 XL instances into one dataset so future
tables/plots don't need to re-parse PDFs:
  1. artifacts/comparison/xl_heuristic_comparison.json -- the eight Queiroga
     et al. (2026) monolithic solvers (best-of-60 / mean-of-60 per instance)
     plus DRSCI, plus the current official BKS (CVRPLib BKS Challenge).
  2. data/results/finalBenchmarkResults.csv -- DRISPI's own best-of-3-seed
     and mean-of-3-seed cost per instance.
  3. Cross-validates that both sources agree on the current BKS per instance.

Writes:
  - data/results/xl_solver_comparison.json (nested: per instance -> bks +
    per-solver {best, mean} costs, for AILS-II, FILO, FILO2, KGLSXXL,
    HGS-CVRP, SISRs, LKH-3, OR-Tools, DRSCI, DRISPI)
  - data/results/xl_solver_comparison.csv (flat, one row per instance, one
    best/mean column pair per solver)

Usage:
    python scripts/build_solver_comparison_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEURISTIC_JSON = ROOT / "artifacts/comparison/xl_heuristic_comparison.json"
CSV_PATH = ROOT / "data/results/finalBenchmarkResults_lagrange.csv"
OUTPUT_JSON = ROOT / "data/results/xl_solver_comparison_lagrange.json"
OUTPUT_CSV = ROOT / "data/results/xl_solver_comparison_lagrange.csv"

# Order mirrors Queiroga et al. (2026) Table 2, then the two "own work" rows.
SOLVER_ORDER = [
    "AILS-II",
    "FILO",
    "FILO2",
    "KGLSXXL",
    "HGS-CVRP",
    "SISRs",
    "LKH-3",
    "OR-Tools",
    "DRSCI",
    "DRISPI",
]


def _load_csv_rows(csv_path: Path) -> dict[str, dict]:
    with csv_path.open(newline="") as fh:
        return {row["instance"]: row for row in csv.DictReader(fh)}


def build(csv_path: Path) -> dict[str, dict]:
    heuristics = json.loads(HEURISTIC_JSON.read_text(encoding="utf-8"))
    csv_rows = _load_csv_rows(csv_path)

    missing = set(heuristics) ^ set(csv_rows)
    if missing:
        raise RuntimeError(f"Instance mismatch between JSON and CSV sources: {sorted(missing)}")

    combined: dict[str, dict] = {}
    for instance in sorted(csv_rows, key=lambda name: int(csv_rows[name]["n"])):
        row = csv_rows[instance]
        entry = heuristics[instance]

        bks_json = entry["BKS"]
        bks_csv = int(round(float(row["bks_current"])))
        if bks_json != bks_csv:
            raise RuntimeError(f"{instance}: BKS mismatch json={bks_json} csv={bks_csv}")

        solvers: dict[str, dict[str, float]] = dict(entry["heuristics"])
        drispi_best = float(row["cost_best3"])
        solvers["DRISPI"] = {
            "best": int(drispi_best) if drispi_best.is_integer() else drispi_best,
            "mean": round(float(row["cost_mean3"]), 1),
        }
        # Normalize Queiroga's "avg" key to "mean" for a uniform schema.
        for solver, metrics in solvers.items():
            if solver == "DRISPI":
                continue
            if "avg" in metrics:
                metrics["mean"] = metrics.pop("avg")

        combined[instance] = {
            "n": int(row["n"]),
            "K": int(row["K"]),
            "bks": bks_json,
            "solvers": solvers,
        }

    return combined


def write_json(combined: dict[str, dict], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(combined)} instances to {output_json}")


def write_csv(combined: dict[str, dict], output_csv: Path) -> None:
    fieldnames = ["instance", "n", "K", "bks"]
    for solver in SOLVER_ORDER:
        fieldnames += [f"{solver}_best", f"{solver}_mean"]

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for instance, entry in combined.items():
            row = {"instance": instance, "n": entry["n"], "K": entry["K"], "bks": entry["bks"]}
            for solver in SOLVER_ORDER:
                metrics = entry["solvers"].get(solver, {})
                row[f"{solver}_best"] = metrics.get("best", "")
                row[f"{solver}_mean"] = metrics.get("mean", "")
            writer.writerow(row)
    print(f"Wrote {len(combined)} rows to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    combined = build(args.csv)
    write_json(combined, args.output_json)
    write_csv(combined, args.output_csv)


if __name__ == "__main__":
    main()
