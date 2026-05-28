"""CLI entry point for the DRISPI pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import vrplib

from drispi.core.instance import CVRPInstance
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.pipeline import DRISPIPipeline
from drispi.utils.metrics import resolve_bks_cost

DEFAULT_BKS_FILE = Path("data/bks/xl-bks.json")


def load_instance_from_vrp_path(instance_path: Path) -> CVRPInstance:
    """Load a VRPLIB file from an explicit path (not only configured search dirs)."""
    path = instance_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    data = vrplib.read_instance(path)
    node_coords = data["node_coord"]
    demands_raw = data["demand"]
    capacity = int(data["capacity"])
    name = str(data.get("name", path.stem))

    coordinates: dict[int, tuple[float, float]] = {
        idx + 1: (float(coord[0]), float(coord[1])) for idx, coord in enumerate(node_coords)
    }
    demands: dict[int, int] = {idx + 1: int(demand) for idx, demand in enumerate(demands_raw)}

    n_nodes = len(node_coords)
    if n_nodes < 1:
        raise ValueError("Instance must contain at least depot node")

    customers = list(range(2, n_nodes + 1))
    n_customers = n_nodes - 1

    if customers != list(range(2, n_customers + 2)):
        raise ValueError("Customer IDs must be 2..n in internal format")
    if 1 not in coordinates or 1 not in demands:
        raise ValueError("Depot node 1 is missing")
    if demands[1] != 0:
        raise ValueError("Depot demand must be 0")
    if sum(demands[cid] for cid in customers) <= 0:
        raise ValueError("Sum of customer demands must be positive")

    return CVRPInstance(
        name=name,
        n_customers=n_customers,
        capacity=capacity,
        depot=coordinates[1],
        customers=customers,
        coordinates=coordinates,
        demands=demands,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DRISPI pipeline on a CVRP instance.")
    parser.add_argument("instance", type=Path, help="Path to a .vrp instance file")
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--max-no-improve", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--sp-interval", type=int, default=3)
    parser.add_argument("--min-coverage", type=int, default=5)
    parser.add_argument("--sp-time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-pool-size", type=int, default=10000)
    parser.add_argument("--bg-ails-omega", type=float, default=0.8)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--bks", type=float, default=None, help="Known BKS cost (overrides file)")
    parser.add_argument(
        "--bks-file",
        type=Path,
        default=DEFAULT_BKS_FILE,
        help="JSON table of instance name -> BKS cost",
    )
    parser.add_argument(
        "--no-bks",
        action="store_true",
        help="Do not load BKS from file (gaps and NEW BKS events disabled)",
    )
    args = parser.parse_args()

    from drispi.haos.config import HAOSConfig

    haos_cfg = HAOSConfig(decay=args.decay)
    config = DRISPIConfig(
        time_limit=args.time_limit,
        max_no_improve=args.max_no_improve,
        n_workers=args.n_workers,
        warmup_iterations=args.warmup,
        sp_interval=args.sp_interval,
        min_coverage=args.min_coverage,
        sp_time_limit=args.sp_time_limit,
        mip_gap=args.mip_gap,
        max_pool_size=args.max_pool_size,
        bg_ails_initial_omega=args.bg_ails_omega,
        haos_config=haos_cfg,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    inst = load_instance_from_vrp_path(args.instance)
    bks_cost = None if args.no_bks else resolve_bks_cost(
        inst.name,
        bks_override=args.bks,
        bks_file=args.bks_file,
    )
    DRISPIPipeline(inst, config, bks_cost=bks_cost).run()


if __name__ == "__main__":
    main()
