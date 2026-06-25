"""CLI entry point for the DRISPI pipeline."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import vrplib

from drispi.core.instance import CVRPInstance
from drispi.pipeline.config import DRISPIConfig
from drispi.pipeline.config_io import load_config, merge_cli_overrides
from drispi.pipeline.pipeline import DRISPIPipeline, make_run_label
from drispi.utils.metrics import resolve_bks_cost

DEFAULT_BKS_FILE = Path("data/bks/xl-bks.json")
DASHBOARD_PORT = 8501

_CLI_TO_FIELD: dict[str, str] = {
    "time_limit": "time_limit",
    "max_no_improve": "max_no_improve",
    "n_workers": "n_workers",
    "warmup": "warmup_iterations",
    "sp_interval": "sp_interval",
    "min_coverage": "min_coverage",
    "sp_time_limit": "sp_time_limit",
    "mip_gap": "mip_gap",
    "max_pool_size": "max_pool_size",
    "bg_ails_omega": "bg_ails_initial_omega",
    "decay": "haos_decay",
    "output_dir": "output_dir",
    "seed": "seed",
    "analysis": "run_analysis",
}


def _build_config_from_args(args: argparse.Namespace) -> DRISPIConfig:
    config = DRISPIConfig()
    if args.config:
        config = load_config(args.config)
    overrides: dict[str, object] = {}
    for arg_name, field_name in _CLI_TO_FIELD.items():
        value = getattr(args, arg_name, None)
        if field_name == "run_analysis":
            if value is True:
                overrides[field_name] = True
        elif value is not None:
            overrides[field_name] = value
    return merge_cli_overrides(config, overrides)


def _try_open_browser(url: str) -> None:
    """Best-effort browser launch (works on Linux, macOS, and WSL → Windows)."""
    if webbrowser.open(url):
        return
    for cmd in (
        ["wslview", url],
        ["cmd.exe", "/c", "start", "", url],
        ["xdg-open", url],
    ):
        try:
            subprocess.run(cmd, check=False, timeout=5)
            return
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _stop_dashboard_server(port: int = DASHBOARD_PORT) -> None:
    """Free the dashboard port so a new run can bind the correct --run-dir."""
    subprocess.run(
        ["pkill", "-f", "streamlit run app.py"],
        check=False,
        capture_output=True,
    )
    deadline = time.monotonic() + 8.0
    while _port_open("127.0.0.1", port) and time.monotonic() < deadline:
        time.sleep(0.2)


def _launch_dashboard(run_dir: Path, dashboard_app: Path) -> None:
    """Start Streamlit in the background; server must bind before pipeline continues."""
    _stop_dashboard_server(DASHBOARD_PORT)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "streamlit.log"
    app_dir = dashboard_app.resolve().parent
    streamlit_env = os.environ.copy()
    streamlit_env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    streamlit_env["STREAMLIT_SERVER_HEADLESS"] = "true"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                dashboard_app.name,
                "--server.headless",
                "true",
                "--server.port",
                str(DASHBOARD_PORT),
                "--",
                "--run-dir",
                str(run_dir.resolve()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(app_dir),
            env=streamlit_env,
            start_new_session=True,
        )

    url = f"http://localhost:{DASHBOARD_PORT}"
    deadline = time.monotonic() + 30.0
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if _port_open("127.0.0.1", DASHBOARD_PORT):
            try:
                log_text = log_path.read_text(encoding="utf-8")
            except OSError:
                log_text = ""
            if "Uvicorn server started" in log_text or "Local URL:" in log_text:
                ready = True
                break
        time.sleep(0.25)

    print(
        f"DRISPI dashboard: {url}\n"
        f"  Run dir: {run_dir.resolve()}\n"
        f"  Streamlit log: {log_path.resolve()}",
        file=sys.stderr,
    )
    if not ready:
        print(
            "  Warning: dashboard did not start within 30s; "
            f"check {log_path} (e.g. port already in use).",
            file=sys.stderr,
        )
    elif proc.poll() is not None:
        print(
            f"  Warning: Streamlit exited early (code {proc.returncode}); see {log_path}.",
            file=sys.stderr,
        )
    else:
        _try_open_browser(url)


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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (e.g. configs/benchmark.yaml). "
        "CLI args override YAML values.",
    )
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--max-no-improve", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--sp-interval", type=int, default=None)
    parser.add_argument("--min-coverage", type=int, default=None)
    parser.add_argument("--sp-time-limit", type=float, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument("--max-pool-size", type=int, default=None)
    parser.add_argument("--bg-ails-omega", type=float, default=None)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Streamlit dashboard alongside the pipeline run.",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run post-run analysis and generate charts after the pipeline completes.",
    )
    args = parser.parse_args()

    config = _build_config_from_args(args)

    inst = load_instance_from_vrp_path(args.instance)
    bks_cost = None if args.no_bks else resolve_bks_cost(
        inst.name,
        bks_override=args.bks,
        bks_file=args.bks_file,
    )
    run_label = make_run_label(inst.name)
    run_dir = config.output_dir / run_label
    if args.gui:
        dashboard_app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
        _launch_dashboard(run_dir, dashboard_app)
    DRISPIPipeline(inst, config, bks_cost=bks_cost, run_label=run_label).run()


if __name__ == "__main__":
    main()
