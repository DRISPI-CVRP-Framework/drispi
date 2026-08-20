from __future__ import annotations

import random

import pytest

from drispi.core.instance import CVRPInstance
from drispi.haos.config import HAOSConfig, HAOSRewardConfig
from drispi.haos.haos import HAOS, HAOSSelection
from drispi.haos.tag import HAOSTag


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


def test_coerce_vertex_when_no_routes_switches_paradigm_and_method() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    route_selection = HAOSSelection(
        k=2,
        lambda_demand=0.4,
        paradigm="route",
        method="agglomerative_complete",
        solver="pyvrp",
        k_index=0,
        lambda_index=2,
        paradigm_index=1,
        method_index=2,
        solver_index=0,
    )
    coerced = haos.coerce_vertex_when_no_routes(
        route_selection,
        best_solution_available=False,
        rng=random.Random(42),
    )
    assert coerced.paradigm == "vertex"
    assert coerced.method in haos.wheel_4a_vertex_method.choices
    assert coerced.paradigm_index == haos.wheel_3_paradigm.choices.index("vertex")
    assert coerced.k == route_selection.k
    assert coerced.lambda_demand == route_selection.lambda_demand
    assert coerced.solver == route_selection.solver


def test_coerce_vertex_when_no_routes_noop_if_best_solution_available() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    route_selection = HAOSSelection(
        k=2,
        lambda_demand=0.4,
        paradigm="route",
        method="kmeans",
        solver="pyvrp",
        k_index=1,
        lambda_index=2,
        paradigm_index=1,
        method_index=0,
        solver_index=0,
    )
    unchanged = haos.coerce_vertex_when_no_routes(
        route_selection,
        best_solution_available=True,
        rng=random.Random(42),
    )
    assert unchanged is route_selection


def test_cap_k_for_route_clustering_clips_k_and_keeps_k_index() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    route_selection = HAOSSelection(
        k=4,
        lambda_demand=0.4,
        paradigm="route",
        method="kmeans",
        solver="pyvrp",
        k_index=2,
        lambda_index=2,
        paradigm_index=1,
        method_index=0,
        solver_index=0,
    )
    capped = haos.cap_k_for_route_clustering(route_selection, n_routes=2)
    assert capped.k == 2
    # Credit stays on the arm that was actually rolled.
    assert capped.k_index == route_selection.k_index


def test_cap_k_for_route_clustering_clips_below_domain_without_raising() -> None:
    # A single-route incumbent clips to k=1 even though 1 is not a domain arm.
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    route_selection = HAOSSelection(
        k=4,
        lambda_demand=0.4,
        paradigm="route",
        method="kmeans",
        solver="pyvrp",
        k_index=2,
        lambda_index=2,
        paradigm_index=1,
        method_index=0,
        solver_index=0,
    )
    capped = haos.cap_k_for_route_clustering(route_selection, n_routes=1)
    assert capped.k == 1
    assert capped.k_index == route_selection.k_index


def test_cap_k_for_route_clustering_noop_when_k_within_route_count() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    route_selection = HAOSSelection(
        k=2,
        lambda_demand=0.4,
        paradigm="route",
        method="kmeans",
        solver="pyvrp",
        k_index=1,
        lambda_index=2,
        paradigm_index=1,
        method_index=0,
        solver_index=0,
    )
    unchanged = haos.cap_k_for_route_clustering(route_selection, n_routes=10)
    assert unchanged is route_selection


def test_cap_k_for_route_clustering_noop_for_vertex_paradigm() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    vertex_selection = HAOSSelection(
        k=4,
        lambda_demand=0.4,
        paradigm="vertex",
        method="kmeans",
        solver="pyvrp",
        k_index=3,
        lambda_index=2,
        paradigm_index=0,
        method_index=0,
        solver_index=0,
    )
    unchanged = haos.cap_k_for_route_clustering(vertex_selection, n_routes=2)
    assert unchanged is vertex_selection


def test_coerce_vertex_when_no_routes_noop_if_already_vertex() -> None:
    haos = HAOS(config=HAOSConfig(), instance=make_instance_20())
    vertex_selection = HAOSSelection(
        k=2,
        lambda_demand=0.4,
        paradigm="vertex",
        method="kmeans",
        solver="pyvrp",
        k_index=1,
        lambda_index=2,
        paradigm_index=0,
        method_index=0,
        solver_index=0,
    )
    unchanged = haos.coerce_vertex_when_no_routes(
        vertex_selection,
        best_solution_available=False,
        rng=random.Random(42),
    )
    assert unchanged is vertex_selection


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
    assert tag.is_improvement_route is False
    imp = selection.to_tag(iteration=3, is_improvement_route=True)
    assert imp.is_improvement_route is True
    assert imp.k == selection.k and imp.solver == selection.solver


def test_coerced_vertex_selection_credits_vertex_wheels_on_update_final() -> None:
    config = HAOSConfig(haos_warmup=0, decay=0.95)
    haos = HAOS(config=config, instance=make_instance_20())
    route_selection = HAOSSelection(
        k=2,
        lambda_demand=0.4,
        paradigm="route",
        method="agglomerative_complete",
        solver="pyvrp",
        k_index=0,
        lambda_index=2,
        paradigm_index=1,
        method_index=2,
        solver_index=0,
    )
    before = haos.state_dict()
    coerced = haos.coerce_vertex_when_no_routes(
        route_selection,
        best_solution_available=False,
        rng=random.Random(7),
    )
    haos.update_immediate(coerced, iteration=0, reward=2.0)
    haos.update_final(coerced, iteration=0)
    after = haos.state_dict()

    # (10.0 + 2.0) * 0.95 = 11.4; unrewarded weights decay to 9.5 but are
    # floored back at 10.0, so they stay unchanged.
    assert after["level_3_paradigm"]["raw_weights"][coerced.paradigm_index] == pytest.approx(11.4)
    assert after["level_4a_vertex_method"]["raw_weights"][coerced.method_index] == pytest.approx(
        11.4
    )
    assert (
        after["level_4b_route_method"]["raw_weights"]
        == before["level_4b_route_method"]["raw_weights"]
    )
    assert (
        after["level_3_paradigm"]["raw_weights"][route_selection.paradigm_index]
        == before["level_3_paradigm"]["raw_weights"][route_selection.paradigm_index]
    )


def test_update_final_after_warmup_updates_selected_weights() -> None:
    config = HAOSConfig(haos_warmup=0, decay=0.95)
    haos = HAOS(config=config, instance=make_instance_20())
    selection = haos.select(iteration=0, rng=random.Random(4))
    before = haos.state_dict()
    haos.update_immediate(selection, iteration=0, reward=2.0)
    haos.update_final(selection, iteration=0)
    after = haos.state_dict()
    assert after != before
    assert after["level_1_k"]["raw_weights"][selection.k_index] == pytest.approx(11.4)


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


def test_compute_k_values_fallback_uses_min_feasible_fleet() -> None:
    """Unparseable name: n=20, K_min=ceil(200/50)=4 caps the domain at [2, 3, 4]."""
    instance = make_instance_20()
    assert HAOSConfig().compute_k_values(instance) == [2, 3, 4]


def test_compute_k_values_prefers_filename_n_and_kmin() -> None:
    """A CVRPLib-style name overrides customer/demand-derived bounds."""
    import dataclasses

    instance = dataclasses.replace(make_instance_20(), name="XL-n9571-k55")
    values = HAOSConfig().compute_k_values(instance)
    assert values == [2, 3, 4, 6, 8, 10, 12, 18, 27, 40]


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


def test_update_deferred_ignores_improvement_route_tags() -> None:
    """Improvement-tagged HAOSTags must not receive deferred credit."""
    config = HAOSConfig(haos_warmup=0, decay=0.8)
    haos_only_immediate = HAOS(config=config, instance=make_instance_20())
    haos_with_filtered_deferred = HAOS(config=config, instance=make_instance_20())
    rng = random.Random(99)
    sel_a = haos_only_immediate.select(iteration=2, rng=rng)
    rng = random.Random(99)
    sel_b = haos_with_filtered_deferred.select(iteration=2, rng=rng)
    assert sel_a.k_index == sel_b.k_index

    haos_only_immediate.update_immediate(sel_a, iteration=2, reward=3.0)
    haos_only_immediate.update_final(sel_a, iteration=2)
    baseline = haos_only_immediate.state_dict()

    imp = sel_b.to_tag(iteration=2, is_improvement_route=True)
    haos_with_filtered_deferred.update_immediate(sel_b, iteration=2, reward=3.0)
    haos_with_filtered_deferred.update_deferred([imp], iteration=2, deferred_reward=999.0)
    haos_with_filtered_deferred.update_final(sel_b, iteration=2)

    assert haos_with_filtered_deferred.state_dict() == baseline
