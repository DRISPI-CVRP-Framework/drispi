from __future__ import annotations

import json
import random

import pytest

from drispi.core.instance import CVRPInstance
from drispi.haos.config import HAOSConfig
from drispi.haos.haos import HAOS
from drispi.haos.weights_io import load_weights, save_weights


def make_instance_20() -> CVRPInstance:
    customers = list(range(2, 22))
    coordinates = {1: (0.0, 0.0)}
    demands = {1: 0}
    for idx, customer in enumerate(customers, start=1):
        coordinates[customer] = (float(idx), float(idx % 5))
        demands[customer] = 10
    return CVRPInstance(
        name="synthetic20",
        n_customers=20,
        capacity=50,
        depot=(0.0, 0.0),
        customers=customers,
        coordinates=coordinates,
        demands=demands,
    )


def test_save_weights_writes_expected_json(tmp_path) -> None:
    haos = HAOS(config=HAOSConfig(k_min_routes_per_cluster=0, k_min_arm_spacing=1), instance=make_instance_20())
    output = tmp_path / "weights.json"
    save_weights(haos, output, instance_name="x20", iterations_completed=42)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["instance"] == "x20"
    assert payload["iterations_completed"] == 42
    assert "timestamp" in payload
    assert set(payload["levels"].keys()) == {
        "level_1_k",
        "level_2_lambda_demand",
        "level_3_paradigm",
        "level_4a_vertex_method",
        "level_4b_route_method",
        "level_5_solver",
    }


def test_load_weights_round_trip_restores_state(tmp_path) -> None:
    config = HAOSConfig(haos_warmup=0, k_min_routes_per_cluster=0, k_min_arm_spacing=1)
    source = HAOS(config=config, instance=make_instance_20())
    target = HAOS(config=config, instance=make_instance_20())
    selection = source.select(iteration=0, rng=random.Random(11))
    source.update_immediate(selection, iteration=0, reward=6.0)
    source.update_final(selection, iteration=0)

    output = tmp_path / "weights.json"
    save_weights(source, output, instance_name="x20", iterations_completed=1)
    load_weights(target, output)
    assert target.state_dict() == source.state_dict()


def test_load_weights_missing_file_raises(tmp_path) -> None:
    haos = HAOS(config=HAOSConfig(k_min_routes_per_cluster=0, k_min_arm_spacing=1), instance=make_instance_20())
    with pytest.raises(FileNotFoundError):
        load_weights(haos, tmp_path / "missing.json")
