#!/usr/bin/env python3
"""Verify async BG-AILS invariants in a run.jsonl (stdlib only, runs on host).

Checks:
  1. bg_launch fires for every completed iteration (minus skipped iterations
     and launches legitimately skipped at the wall-clock cap).
  2. Every applied BG result has exactly one iteration of lag
     (bg_ails_apply_iteration == bg_ails_launch_iteration + 1).
  3. handoff_block_wait_s stays small (report distribution; warn > threshold).
  4. No bg_apply with reason error:* / worker_died; bg_run_totals.crash_count == 0.
  5. Every launch is accounted for: applied, or the single in-flight job
     discarded at shutdown.

Usage:
  python3 scripts/check_bg_async_sanity.py <run_dir_or_run.jsonl> [--max-block-wait 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OK = "PASS"
BAD = "FAIL"
WARN = "WARN"


def _load_events(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_path", type=Path, help="run dir or run.jsonl")
    ap.add_argument(
        "--max-block-wait",
        type=float,
        default=30.0,
        help="warn when any handoff_block_wait_s exceeds this (seconds)",
    )
    args = ap.parse_args()

    path = args.run_path
    if path.is_dir():
        path = path / "run.jsonl"
    if not path.is_file():
        print(f"{BAD}: no run.jsonl at {path}")
        return 1

    events = _load_events(path)
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e.get("type", "?"), []).append(e)

    summaries = by_type.get("summary", [])
    launches = by_type.get("bg_launch", [])
    applies = by_type.get("bg_apply", [])
    skipped_cap = by_type.get("bg_launch_skipped_cap", [])
    skipped_iter = by_type.get("iteration_skipped", [])
    totals = (by_type.get("bg_run_totals") or [{}])[-1]

    failures = 0
    warnings = 0

    def report(ok: bool, msg: str) -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"{OK if ok else BAD}: {msg}")

    print(f"run.jsonl: {path}")
    print(
        f"iterations={len(summaries)} bg_launch={len(launches)} "
        f"bg_apply={len(applies)} launch_skipped_cap={len(skipped_cap)} "
        f"iteration_skipped={len(skipped_iter)}"
    )

    # 0. Async mode actually active
    report(bool(launches) or bool(skipped_cap),
           "async BG active (bg_launch/bg_launch_skipped_cap events present)")

    # 1. A launch (or cap-skip) for every completed iteration
    summary_iters = {e["iteration"] for e in summaries}
    launch_iters = {e["iteration"] for e in launches}
    cap_skip_iters = {e["iteration"] for e in skipped_cap}
    dr_fail_iters = {e["iteration"] for e in skipped_iter}
    unaccounted = summary_iters - launch_iters - cap_skip_iters - dr_fail_iters
    report(
        not unaccounted,
        f"bg_launch every iteration (unaccounted: {sorted(unaccounted) or 'none'})",
    )

    # 2. Exactly one iteration of lag on every successful apply
    ok_applies = [a for a in applies if a.get("reason") in ("adopted", "worse_than_incumbent")]
    bad_lag = [
        (a["bg_ails_launch_iteration"], a["bg_ails_apply_iteration"])
        for a in ok_applies
        if a["bg_ails_apply_iteration"] != a["bg_ails_launch_iteration"] + 1
    ]
    report(not bad_lag, f"one-iteration lag on all applies (violations: {bad_lag or 'none'})")

    drain_points = {}
    for a in ok_applies:
        drain_points[a.get("drain_point")] = drain_points.get(a.get("drain_point"), 0) + 1
    print(f"       drain_point distribution: {drain_points}")
    adopted = sum(1 for a in ok_applies if a.get("adopted"))
    print(f"       applies adopted: {adopted}/{len(ok_applies)}")

    # 3. Handoff block waits
    waits = [e.get("handoff_block_wait_s") for e in launches + skipped_cap]
    waits = [w for w in waits if isinstance(w, (int, float))]
    if waits:
        w_max, w_mean = max(waits), sum(waits) / len(waits)
        print(f"       handoff_block_wait_s: max={w_max:.2f}s mean={w_mean:.2f}s n={len(waits)}")
        if w_max > args.max_block_wait:
            warnings += 1
            print(
                f"{WARN}: max handoff block wait {w_max:.1f}s exceeds "
                f"{args.max_block_wait:.0f}s — BG budget may exceed DR-stage wall time"
            )
    else:
        print(f"{WARN}: no handoff_block_wait_s values found")
        warnings += 1

    # 4. No worker failures
    failed_applies = [
        a for a in applies
        if str(a.get("reason", "")).startswith("error:") or a.get("reason") == "worker_died"
    ]
    report(not failed_applies, f"no BG worker failures (found: {len(failed_applies)})")
    crash_count = totals.get("crash_count")
    report(crash_count in (0, None), f"bg_run_totals.crash_count == 0 (got {crash_count})")

    # 5. Launch/apply accounting
    if totals:
        launched = totals.get("launched", len(launches))
        applied = totals.get("applied", len(ok_applies))
        discarded = 1 if totals.get("discarded_in_flight_at_shutdown") else 0
        report(
            launched == applied + discarded + (totals.get("crash_count") or 0),
            f"accounting: launched={launched} == applied={applied} "
            f"+ discarded_at_shutdown={discarded} + crashes={totals.get('crash_count', 0)}",
        )
    else:
        print(f"{WARN}: no bg_run_totals event (run not finalized?)")
        warnings += 1

    print()
    if failures:
        print(f"RESULT: {BAD} ({failures} failed check(s), {warnings} warning(s))")
        return 1
    print(f"RESULT: {OK} ({warnings} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
