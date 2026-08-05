#!/usr/bin/env python3
"""Standalone AILS-II JVM ergonomics diagnostic (not through the pipeline).

Compares:
  (a) taskset -c <cpu> + -XX:+UseSerialGC -XX:ActiveProcessorCount=1 -Xmx4g
  (b) unpinned, no added JVM flags (today's production behaviour)

Each run: fixed .vrp + .sol, -stoppingCriterion Time -limit 180, -initialSolution.
AILS-II has no RNG seed flag; reproducibility is input-only.

Usage:
  python scripts/diag_ails_jvm_ergonomics.py
  python scripts/diag_ails_jvm_ergonomics.py --limit 30   # smoke
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_JAR = _ROOT / "ext" / "ails2" / "build" / "AILSII.jar"
_DATA = _ROOT / "ext" / "ails2" / "data"

_DEFAULT_INSTANCES = ("X-n101-k25", "X-n214-k11", "X-n459-k26")
_JVM_FLAGS_A = ("-XX:+UseSerialGC", "-XX:ActiveProcessorCount=1", "-Xmx4g")
_COST_RE = re.compile(r"^Cost\s+(\S+)", re.MULTILINE)


def _parse_cost(sol_path: Path) -> float | None:
    text = sol_path.read_text(encoding="utf-8", errors="replace")
    m = _COST_RE.search(text)
    if not m:
        return None
    return float(m.group(1))


def _run_one(
    *,
    name: str,
    vrp: Path,
    init_sol: Path,
    out_sol: Path,
    limit: float,
    pinned: bool,
    cpu: int,
) -> dict:
    out_sol.parent.mkdir(parents=True, exist_ok=True)
    if out_sol.exists():
        out_sol.unlink()

    java_argv: list[str] = ["java"]
    if pinned:
        java_argv.extend(_JVM_FLAGS_A)
    java_argv.extend(
        [
            "-jar",
            str(_JAR),
            "-file",
            str(vrp),
            "-rounded",
            "true",
            "-stoppingCriterion",
            "Time",
            "-limit",
            str(float(limit)),
            "-initialSolution",
            str(init_sol),
            "-outpath",
            str(out_sol),
        ]
    )
    cmd = (["taskset", "-c", str(cpu)] + java_argv) if pinned else java_argv
    label = "pinned+flags" if pinned else "unpinned+default"
    print(f"\n=== {name}  {label}  ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.perf_counter()
    completed = subprocess.run(
        cmd, capture_output=True, text=True, timeout=float(limit) + 120.0
    )
    wall = time.perf_counter() - t0
    cost = _parse_cost(out_sol) if out_sol.is_file() else None
    if completed.returncode != 0:
        print(completed.stderr[-2000:] or completed.stdout[-2000:], file=sys.stderr)
    row = {
        "instance": name,
        "config": label,
        "pinned": pinned,
        "returncode": completed.returncode,
        "wall_s": round(wall, 2),
        "cost": cost,
        "init_cost": _parse_cost(init_sol),
    }
    print(
        f"-> returncode={row['returncode']} wall={row['wall_s']}s "
        f"init={row['init_cost']} final={row['cost']}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=float, default=180.0)
    parser.add_argument("--cpu", type=int, default=0, help="CPU for pinned arm")
    parser.add_argument("--instances", nargs="+", default=list(_DEFAULT_INSTANCES))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "artifacts" / "diag_ails_jvm",
    )
    args = parser.parse_args()

    if not _JAR.is_file():
        print(f"JAR not found: {_JAR}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for name in args.instances:
        vrp = _DATA / f"{name}.vrp"
        init_sol = _DATA / f"{name}.sol"
        if not vrp.is_file() or not init_sol.is_file():
            print(f"skip {name}: missing vrp/sol under {_DATA}", file=sys.stderr)
            continue
        for pinned in (True, False):
            tag = "pinned" if pinned else "unpinned"
            out_sol = args.out_dir / f"{name}_{tag}.sol"
            rows.append(
                _run_one(
                    name=name,
                    vrp=vrp,
                    init_sol=init_sol,
                    out_sol=out_sol,
                    limit=args.limit,
                    pinned=pinned,
                    cpu=args.cpu,
                )
            )

    print("\n========== SUMMARY ==========")
    print(
        f"{'instance':<16} {'config':<20} {'init':>10} {'final':>10} "
        f"{'delta':>10} {'wall_s':>8}"
    )
    for r in rows:
        init_c = r["init_cost"]
        final = r["cost"]
        delta = (
            (final - init_c) if (final is not None and init_c is not None) else None
        )
        print(
            f"{r['instance']:<16} {r['config']:<20} "
            f"{init_c if init_c is not None else 'NA':>10} "
            f"{final if final is not None else 'NA':>10} "
            f"{delta if delta is not None else 'NA':>10} "
            f"{r['wall_s']:>8}"
        )

    print("\n========== PINNED vs UNPINNED (final cost) ==========")
    by_inst: dict[str, dict[str, float | None]] = {}
    for r in rows:
        by_inst.setdefault(r["instance"], {})[r["config"]] = r["cost"]
    for inst, costs in by_inst.items():
        a = costs.get("pinned+flags")
        b = costs.get("unpinned+default")
        if a is None or b is None:
            print(f"{inst}: incomplete")
            continue
        abs_d = a - b
        rel = 100.0 * abs_d / b if b else float("nan")
        print(
            f"{inst}: pinned={a} unpinned={b}  "
            f"abs_delta={abs_d:+.3f}  rel={rel:+.4f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
