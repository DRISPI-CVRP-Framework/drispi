#!/usr/bin/env python3
"""Analyze Gurobi Threads sweep JSONL from ``sweep_gurobi_threads.py``.

Usage:
    python scripts/analyze_gurobi_thread_sweep.py artifacts/sweeps \\
        --output-dir artifacts/sweeps/analysis

Reads every ``*_gurobi_threads.jsonl`` under the input directory, aggregates
repeats, prints mean wall-time tables (total / LP / MIP), writes PNG curves,
and emits a short ceiling recommendation (where extra threads stop buying
meaningful speedup).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")

from drispi.pipeline.analysis import make_figure, save_figure, style_axes, style_legend

# Relative speedup vs 1-thread below which we treat further threads as wasted.
_SPEEDUP_FLOOR = 1.05
# Minimum mean total wall (s) before we bother recommending a ceiling.
_MIN_TIME_FOR_CEILING_S = 0.5


def _load_jsonl_files(sweeps_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(sweeps_dir.glob("*_gurobi_threads.jsonl"))
    if not paths:
        paths = sorted(sweeps_dir.glob("*.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source"] = path.name
            rows.append(row)
    return rows


def _agg_key(row: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row["instance"]),
        int(row["mark_min"]),
        str(row["mode"]),
        int(row["threads"]),
    )


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("total_wall_s") is None:
            continue
        buckets[_agg_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for (instance, mark, mode, threads), group in sorted(buckets.items()):
        totals = [float(r["total_wall_s"]) for r in group]
        lps = [float(r["lp_wall_s"]) for r in group]
        mips = [float(r["mip_wall_s"]) for r in group]
        gaps = [float(r["mip_gap"]) for r in group if r.get("mip_gap") is not None]
        timeouts = sum(1 for r in group if r.get("timed_out"))
        pool_sizes = {int(r["pool_size"]) for r in group if r.get("pool_size") is not None}
        out.append(
            {
                "instance": instance,
                "mark_min": mark,
                "mode": mode,
                "threads": threads,
                "n_repeats": len(group),
                "pool_size": next(iter(pool_sizes)) if len(pool_sizes) == 1 else None,
                "total_mean_s": mean(totals),
                "total_std_s": stdev(totals) if len(totals) > 1 else 0.0,
                "lp_mean_s": mean(lps),
                "mip_mean_s": mean(mips),
                "mip_gap_mean": mean(gaps) if gaps else None,
                "n_timeout": timeouts,
            }
        )
    return out


def _print_tables(agg: list[dict[str, Any]]) -> None:
    by_curve: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in agg:
        by_curve[(row["instance"], row["mark_min"], row["mode"])].append(row)

    print("\nMean TOTAL wall-time (s)  [LP / MIP]")
    for key in sorted(by_curve):
        instance, mark, mode = key
        series = sorted(by_curve[key], key=lambda r: r["threads"])
        pool = series[0].get("pool_size")
        pool_s = f"pool={pool}" if pool is not None else "pool=?"
        print(f"\n{instance}  {mark}min  {mode}  {pool_s}")
        print(f"  {'thr':>4}  {'total':>8}  {'±std':>7}  {'lp':>8}  {'mip':>8}  {'gap':>8}  to")
        for r in series:
            gap = r["mip_gap_mean"]
            gap_s = f"{gap:.2e}" if gap is not None else "—"
            print(
                f"  {r['threads']:>4}  {r['total_mean_s']:8.3f}  {r['total_std_s']:7.3f}  "
                f"{r['lp_mean_s']:8.3f}  {r['mip_mean_s']:8.3f}  {gap_s:>8}  {r['n_timeout']}"
            )


def _suggest_ceiling(series: list[dict[str, Any]]) -> int | None:
    """Smallest thread count that is within SPEEDUP_FLOOR of the best mean total."""
    if not series:
        return None
    ordered = sorted(series, key=lambda r: r["threads"])
    best = min(r["total_mean_s"] for r in ordered)
    if best < _MIN_TIME_FOR_CEILING_S:
        return ordered[0]["threads"]  # trivial — 1 thread is enough
    baseline = next(r["total_mean_s"] for r in ordered if r["threads"] == ordered[0]["threads"])
    if baseline <= 0 or not math.isfinite(baseline):
        return None
    # Prefer the smallest t whose mean is within 5% of the best observed mean.
    for r in ordered:
        if r["total_mean_s"] <= best * _SPEEDUP_FLOOR:
            return int(r["threads"])
    return int(ordered[-1]["threads"])


def _print_ceilings(agg: list[dict[str, Any]]) -> None:
    by_curve: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in agg:
        by_curve[(row["instance"], row["mark_min"], row["mode"])].append(row)

    print("\nSuggested thread ceiling (first t within 5% of best mean total)")
    print(f"  {'instance':<18} {'mark':>5} {'mode':>4}  {'ceiling':>7}  note")
    for key in sorted(by_curve):
        instance, mark, mode = key
        series = by_curve[key]
        ceiling = _suggest_ceiling(series)
        best = min(r["total_mean_s"] for r in series)
        note = "trivial (<0.5s)" if best < _MIN_TIME_FOR_CEILING_S else ""
        print(f"  {instance:<18} {mark:>4}m {mode:>4}  {ceiling!s:>7}  {note}")


def _plot_curves(agg: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in agg:
        by_instance[row["instance"]].append(row)

    written: list[Path] = []
    for instance, rows in sorted(by_instance.items()):
        marks = sorted({r["mark_min"] for r in rows})
        modes = sorted({r["mode"] for r in rows})
        n_panels = len(marks) * len(modes)
        ncols = min(2, n_panels)
        nrows = math.ceil(n_panels / ncols)
        fig = make_figure(figsize=(6.0 * ncols, 4.0 * nrows))
        axes = fig.subplots(nrows, ncols, squeeze=False)

        panel = 0
        for mark in marks:
            for mode in modes:
                ax = axes[panel // ncols][panel % ncols]
                series = sorted(
                    (
                        r
                        for r in rows
                        if r["mark_min"] == mark and r["mode"] == mode
                    ),
                    key=lambda r: r["threads"],
                )
                if not series:
                    ax.set_visible(False)
                    panel += 1
                    continue
                thr = [r["threads"] for r in series]
                tot = [r["total_mean_s"] for r in series]
                err = [r["total_std_s"] for r in series]
                lp = [r["lp_mean_s"] for r in series]
                mip = [r["mip_mean_s"] for r in series]
                ax.errorbar(
                    thr,
                    tot,
                    yerr=err,
                    fmt="-o",
                    color="#e8e8e8",
                    ecolor="#888888",
                    capsize=3,
                    label="total",
                )
                ax.plot(thr, lp, "--", color="#4a90d9", label="LP")
                ax.plot(thr, mip, "--", color="#ff9f43", label="MIP")
                pool = series[0].get("pool_size")
                pool_s = f"  pool={pool}" if pool is not None else ""
                ax.set_title(f"{mark}min {mode}{pool_s}")
                ax.set_xlabel("Gurobi Threads")
                ax.set_ylabel("wall time (s)")
                ax.set_xticks(thr)
                style_axes(ax, horizontal_grid_only=True)
                style_legend(ax.legend(loc="best", fontsize=8))
                panel += 1

        for j in range(panel, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.suptitle(
            f"Gurobi Threads sweep — {instance}",
            color="#e8e8e8",
            fontfamily="monospace",
            fontsize=12,
        )
        safe = instance.replace("/", "_")
        path = output_dir / f"threads_vs_time_{safe}.png"
        save_figure(fig, path)
        written.append(path)
    return written


def _write_csv(agg: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "instance",
        "mark_min",
        "mode",
        "threads",
        "n_repeats",
        "pool_size",
        "total_mean_s",
        "total_std_s",
        "lp_mean_s",
        "mip_mean_s",
        "mip_gap_mean",
        "n_timeout",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in agg:
            writer.writerow({k: row.get(k) for k in fields})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Gurobi Threads sweep JSONL outputs.",
    )
    parser.add_argument(
        "sweeps_dir",
        type=Path,
        help="Directory containing *_gurobi_threads.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write plots/CSV (default: <sweeps_dir>/analysis)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sweeps_dir = args.sweeps_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else sweeps_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl_files(sweeps_dir)
    if not rows:
        print(f"No JSONL rows found under {sweeps_dir}", file=sys.stderr)
        sys.exit(1)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"Loaded {len(rows)} rows ({ok} ok) from {sweeps_dir}")

    agg = _aggregate(rows)
    if not agg:
        print("No successful rows to aggregate.", file=sys.stderr)
        sys.exit(2)

    _print_tables(agg)
    _print_ceilings(agg)

    csv_path = output_dir / "thread_sweep_summary.csv"
    _write_csv(agg, csv_path)
    plots = _plot_curves(agg, output_dir)

    print(f"\nWrote {csv_path}")
    for p in plots:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
