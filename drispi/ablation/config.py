"""Pinned constants for the BG-AILS ablation (Stage 2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PINNED_PARADIGM = "vertex"
PINNED_METHOD = "kmedoids"
PINNED_LAMBDA_Q = 0.4
PINNED_SOLVER = "filo2"
PINNED_PAIR_SELECTION = "greedy"
PINNED_N_CHAINS_MODE = "k"
PINNED_TAU = 0.5
PINNED_SMALL_CLUSTER_CAP = 20
PINNED_SMALL_CLUSTER_ALPHA = 0.5
PINNED_OMEGA_BC = 10.0
GRANTED_DRI_WORKERS = 6
SUBCLUSTER_RATE = 0.06
SUBCLUSTER_FLOOR_S = 5.0
BG_BUDGET_FLOOR_S = 60.0
BG_BUDGET_MARGIN = 1.0

# Six generator seeds, two waves of three (plan §1.5).
GENERATOR_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105, 106)
WAVE_SEEDS: dict[int, tuple[int, ...]] = {
    1: (101, 102, 103),
    2: (104, 105, 106),
}

STOCK_JAR = ROOT / "ext/ails2/build/AILSII.jar"
TOUCH_JAR = ROOT / "ext/ails2/build/AILSII-touch.jar"

CHECKPOINT_DIR = ROOT / "artifacts/bg_ails_ablation/checkpoints"
PERF_DIR = ROOT / "artifacts/bg_ails_ablation/performance"
MEAS_DIR = ROOT / "artifacts/bg_ails_ablation/measurement"
FIGURE_DATA_DIR = ROOT / "data/results/bg_ails_ablation"

CHECKPOINT_JOBS = 3
ABLATION_JOBS = 24
# Hang detector floor for checkpoint FILO2 (production is predicted+120s, too
# tight when several generators share the machine).
CHECKPOINT_WALL_FLOOR_S = 480.0


def pinned_config() -> dict[str, object]:
    return {
        "paradigm": PINNED_PARADIGM,
        "method": PINNED_METHOD,
        "lambda_q": PINNED_LAMBDA_Q,
        "solver": PINNED_SOLVER,
        "pair_selection": PINNED_PAIR_SELECTION,
        "n_chains_mode": PINNED_N_CHAINS_MODE,
        "tau": PINNED_TAU,
        "small_cluster_cap": PINNED_SMALL_CLUSTER_CAP,
        "small_cluster_alpha": PINNED_SMALL_CLUSTER_ALPHA,
        "omega_bc": PINNED_OMEGA_BC,
        "granted_dri_workers": GRANTED_DRI_WORKERS,
        "subcluster_rate": SUBCLUSTER_RATE,
        "subcluster_floor_s": SUBCLUSTER_FLOOR_S,
        "bg_budget_floor_s": BG_BUDGET_FLOOR_S,
        "bg_budget_margin": BG_BUDGET_MARGIN,
    }


def config_hash() -> str:
    blob = json.dumps(pinned_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
