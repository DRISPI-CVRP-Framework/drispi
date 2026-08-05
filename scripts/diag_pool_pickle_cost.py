#!/usr/bin/env python3
"""Measure pickle.dumps wall time and bytes for a ~10k-entry RoutePool.

No Phase B .pkl was found in the workspace; builds a synthetic pool with the
same RouteEntry / HAOSTag shape production uses, then times
pickle.dumps(..., HIGHEST_PROTOCOL).

Usage:
  python scripts/diag_pool_pickle_cost.py
  python scripts/diag_pool_pickle_cost.py --pkl path/to/phase_b.pkl
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections import deque
from pathlib import Path

from drispi.haos.tag import HAOSTag
from drispi.route_pool.pool import RoutePool


def _build_pool(n_entries: int, route_len: int, n_customers: int) -> RoutePool:
    if route_len < 2:
        raise ValueError("route_len must be >= 2")
    if n_customers < route_len + 10:
        n_customers = route_len + n_entries + 10

    pool = RoutePool()
    base = 2
    added = 0
    offset = 0
    while added < n_entries:
        start = base + (offset % (n_customers - route_len))
        route = list(range(start, start + route_len))
        route[-1] = base + ((start + route_len + offset) % n_customers)
        key = frozenset(route)
        if key in pool:
            offset += 1
            continue
        tag = HAOSTag(
            k=4 + (added % 5),
            lambda_demand=0.2 * (added % 6),
            paradigm="vertex" if added % 2 == 0 else "route",
            method="kmeans",
            solver="ails2",
            iteration=added % 50,
            is_improvement_route=False,
        )
        cost = 1000.0 + float(added) * 0.1
        pool.add(route, cost, haos_tag=tag)
        entry = pool._entries[key]  # noqa: SLF001 — diagnostic only
        entry.quality_scores = deque([0.1, 0.2, 0.3], maxlen=3)
        entry.diversity_scores = deque([0.4, 0.5], maxlen=3)
        if added < 50:
            entry.is_elite = True
            pool._elite_keys.add(key)  # noqa: SLF001
        added += 1
        offset += 1
    return pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-entries", type=int, default=10_000)
    parser.add_argument("--route-len", type=int, default=20)
    parser.add_argument("--n-customers", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help="Optional Phase B .pkl to load instead of building synthetic",
    )
    args = parser.parse_args()

    if args.pkl is not None:
        print(f"Loading {args.pkl} ...")
        t0 = time.perf_counter()
        with args.pkl.open("rb") as fh:
            pool = pickle.load(fh)
        print(f"load wall={time.perf_counter() - t0:.3f}s  type={type(pool).__name__}")
        if not isinstance(pool, RoutePool):
            raise SystemExit(f"Expected RoutePool, got {type(pool)}")
    else:
        print(
            f"No Phase B pkl provided; building synthetic pool "
            f"n_entries={args.n_entries} route_len={args.route_len}"
        )
        t0 = time.perf_counter()
        pool = _build_pool(args.n_entries, args.route_len, args.n_customers)
        print(f"build wall={time.perf_counter() - t0:.3f}s  size={pool.size()}")

    print(f"pool.size()={pool.size()}  elite={len(pool._elite_keys)}")  # noqa: SLF001

    times: list[float] = []
    nbytes = 0
    for i in range(args.repeats):
        t0 = time.perf_counter()
        payload = pickle.dumps(pool, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        nbytes = len(payload)
        print(f"dumps[{i}] wall={elapsed:.4f}s  bytes={nbytes}  ({nbytes / 1e6:.3f} MiB)")

    print(
        f"\nSUMMARY: n={pool.size()}  "
        f"dumps_mean_s={sum(times) / len(times):.4f}  "
        f"dumps_min_s={min(times):.4f}  dumps_max_s={max(times):.4f}  "
        f"payload_bytes={nbytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
