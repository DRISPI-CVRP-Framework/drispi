from __future__ import annotations

import random

import pytest

from drispi.core.instance import CVRPInstance
from drispi.haos.config import HAOSConfig, HAOSRewardConfig
from drispi.haos.haos import HAOS


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


def test_select_returns_complete_valid_selection() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    selection = haos.select(iteration=0, rng=random.Random(1))
    assert selection.k in haos.wheel_1_k.choices
    assert selection.lambda_demand in haos.wheel_2_lambda.choices
    assert selection.paradigm in haos.wheel_3_paradigm.choices
    assert selection.method in (
        haos.wheel_4a_vertex_method.choices
        if selection.paradigm == "vertex"
        else haos.wheel_4b_route_method.choices
    )
    assert selection.solver in haos.wheel_5_solver.choices


def test_warmup_leaves_weights_unchanged() -> None:
    config = HAOSConfig(haos_warmup=5)
    haos = HAOS(config=config, instance=make_instance_20())
    selection = haos.select(iteration=0, rng=random.Random(2))
    before = haos.state_dict()
    haos.update_immediate(selection, iteration=0, reward=9.0)
    haos.update_final(selection, iteration=0)
    after = haos.state_dict()
    assert before == after


def test_compute_reward_tiers() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    rewards = HAOSRewardConfig()
    assert haos.compute_reward(None, 100.0, 110.0, rewards) == rewards.reward_no_solution
    assert haos.compute_reward(90.0, 100.0, 110.0, rewards) == rewards.reward_new_best
    assert haos.compute_reward(105.0, 100.0, 110.0, rewards) == rewards.reward_improvement
    assert haos.compute_reward(115.0, 100.0, 110.0, rewards) == rewards.reward_no_improvement
    assert (
        haos.compute_reward(None, 100.0, 110.0, rewards, is_deferred=True)
        == rewards.deferred_no_improvement
    )
    assert (
        haos.compute_reward(90.0, 100.0, 110.0, rewards, is_deferred=True)
        == rewards.deferred_new_best
    )
    assert (
        haos.compute_reward(105.0, 100.0, 110.0, rewards, is_deferred=True)
        == rewards.deferred_improvement
    )
    assert (
        haos.compute_reward(115.0, 100.0, 110.0, rewards, is_deferred=True)
        == rewards.deferred_no_improvement
    )


def test_to_tag_from_selection() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    selection = haos.select(iteration=3, rng=random.Random(3))
    tag = selection.to_tag(iteration=3)
    assert tag.k == selection.k
    assert tag.lambda_demand == selection.lambda_demand
    assert tag.paradigm == selection.paradigm
    assert tag.method == selection.method
    assert tag.solver == selection.solver
    assert tag.iteration == 3


def test_update_final_after_warmup_updates_selected_weights() -> None:
    config = HAOSConfig(haos_warmup=0, decay=0.8)
    haos = HAOS(config=config, instance=make_instance_20())
    selection = haos.select(iteration=0, rng=random.Random(4))
    before = haos.state_dict()
    haos.update_immediate(selection, iteration=0, reward=2.0)
    haos.update_final(selection, iteration=0)
    after = haos.state_dict()
    assert after != before
    assert after["level_1_k"]["raw_weights"][selection.k_index] == pytest.approx(2.8)


def test_state_dict_has_all_levels() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    state = haos.state_dict()
    assert set(state.keys()) == {
        "level_1_k",
        "level_2_lambda_demand",
        "level_3_paradigm",
        "level_4a_vertex_method",
        "level_4b_route_method",
        "level_5_solver",
    }


def test_load_state_dict_restores_weights() -> None:
    config = HAOSConfig(haos_warmup=0)
    source = HAOS(config=config, instance=make_instance_20())
    target = HAOS(config=config, instance=make_instance_20())
    selection = source.select(iteration=0, rng=random.Random(5))
    source.update_immediate(selection, iteration=0, reward=7.0)
    source.update_final(selection, iteration=0)
    target.load_state_dict(source.state_dict())
    assert target.state_dict() == source.state_dict()


def test_load_state_dict_choice_mismatch_raises() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    state = haos.state_dict()
    state["level_5_solver"]["values"] = ["other"]
    with pytest.raises(ValueError):
        haos.load_state_dict(state)


def test_compute_k_values_includes_one_and_kmax() -> None:
    instance = make_instance_20()
    values = HAOSConfig.compute_k_values(instance, candidates=[2, 8, 12])
    assert 1 in values
    assert 4 in values
    assert all(value <= 4 for value in values)


def test_historical_deferred_updates_all_reverse_mapped_levels() -> None:
    config = HAOSConfig(haos_warmup=0)
    haos = HAOS(config=config, instance=make_instance_20())
    selection = haos.select(iteration=10, rng=random.Random(6))
    historical = haos.select(iteration=5, rng=random.Random(7)).to_tag(iteration=5)
    before = haos.state_dict()
    haos.update_immediate(selection, iteration=10, reward=1.0)
    haos.update_deferred([historical], iteration=10, deferred_reward=2.5)
    haos.update_final(selection, iteration=10)
    after = haos.state_dict()
    assert after["level_1_k"]["raw_weights"] != before["level_1_k"]["raw_weights"]
    assert after["level_2_lambda_demand"]["raw_weights"] != before["level_2_lambda_demand"]["raw_weights"]
    assert after["level_3_paradigm"]["raw_weights"] != before["level_3_paradigm"]["raw_weights"]
    assert after["level_5_solver"]["raw_weights"] != before["level_5_solver"]["raw_weights"]
