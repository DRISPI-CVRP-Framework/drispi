from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from drispi.core.instance import CVRPInstance
from drispi.haos.config import HAOSConfig, HAOSRewardConfig
from drispi.haos.tag import HAOSTag
from drispi.haos.wheel import RouletteWheel

LOGGER = logging.getLogger(__name__)


@dataclass
class HAOSSelection:
    """Full operator selection for one DRI iteration."""

    k: int
    lambda_demand: float
    paradigm: str
    method: str
    solver: str
    k_index: int
    lambda_index: int
    paradigm_index: int
    method_index: int
    solver_index: int

    def to_tag(self, iteration: int) -> HAOSTag:
        return HAOSTag(
            k=self.k,
            lambda_demand=self.lambda_demand,
            paradigm=self.paradigm,
            method=self.method,
            solver=self.solver,
            iteration=iteration,
        )


class HAOS:
    """Hierarchical Adaptive Operator Selection system."""

    def __init__(
        self,
        config: HAOSConfig,
        instance: CVRPInstance,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.instance = instance
        self.rng = rng or random.Random()

        self.k_values = HAOSConfig.compute_k_values(instance, config.k_candidates)
        self.wheel_1_k = RouletteWheel(self.k_values, config.min_weight_k)
        self.wheel_2_lambda = RouletteWheel(
            config.lambda_demand_values,
            config.min_weight_lambda,
        )
        self.wheel_3_paradigm = RouletteWheel(
            config.paradigm_values,
            config.min_weight_paradigm,
        )
        self.wheel_4a_vertex_method = RouletteWheel(
            config.vertex_method_values,
            config.min_weight_vertex_method,
        )
        self.wheel_4b_route_method = RouletteWheel(
            config.route_method_values,
            config.min_weight_route_method,
        )
        self.wheel_5_solver = RouletteWheel(config.solver_values, config.min_weight_solver)

        self._immediate_rewards: dict[int, float] = {}
        self._deferred_rewards: dict[int, dict[HAOSTag, float]] = {}

    def select(self, iteration: int, rng: random.Random) -> HAOSSelection:
        k_index, k = self.wheel_1_k.select(rng)
        lambda_index, lambda_demand = self.wheel_2_lambda.select(rng)
        paradigm_index, paradigm = self.wheel_3_paradigm.select(rng)
        method_wheel = (
            self.wheel_4a_vertex_method if paradigm == "vertex" else self.wheel_4b_route_method
        )
        method_index, method = method_wheel.select(rng)
        solver_index, solver = self.wheel_5_solver.select(rng)
        return HAOSSelection(
            k=k,
            lambda_demand=lambda_demand,
            paradigm=paradigm,
            method=method,
            solver=solver,
            k_index=k_index,
            lambda_index=lambda_index,
            paradigm_index=paradigm_index,
            method_index=method_index,
            solver_index=solver_index,
        )

    def update_immediate(self, selection: HAOSSelection, iteration: int, reward: float) -> None:
        del selection
        if iteration < self.config.haos_warmup:
            return
        self._immediate_rewards[iteration] = self._immediate_rewards.get(iteration, 0.0) + reward

    def update_deferred(
        self,
        contributing_tags: list[HAOSTag],
        iteration: int,
        deferred_reward: float,
    ) -> None:
        if iteration < self.config.haos_warmup:
            return
        tags_for_iteration = self._deferred_rewards.setdefault(iteration, {})
        for tag in set(contributing_tags):
            tags_for_iteration[tag] = tags_for_iteration.get(tag, 0.0) + deferred_reward

    def _try_reverse_index(self, wheel: RouletteWheel, value: object, level_name: str) -> int | None:
        try:
            return wheel.choices.index(value)
        except ValueError:
            LOGGER.warning(
                "Skipping deferred update for %s: value %r not present in choices %r",
                level_name,
                value,
                wheel.choices,
            )
            return None

    def _update_by_tag_all_levels(self, tag: HAOSTag, reward: float) -> None:
        k_index = self._try_reverse_index(self.wheel_1_k, tag.k, "level_1_k")
        if k_index is not None:
            self.wheel_1_k.update(k_index, reward, decay=0.0)

        lambda_index = self._try_reverse_index(
            self.wheel_2_lambda, tag.lambda_demand, "level_2_lambda_demand"
        )
        if lambda_index is not None:
            self.wheel_2_lambda.update(lambda_index, reward, decay=0.0)

        paradigm_index = self._try_reverse_index(
            self.wheel_3_paradigm, tag.paradigm, "level_3_paradigm"
        )
        if paradigm_index is not None:
            self.wheel_3_paradigm.update(paradigm_index, reward, decay=0.0)

        vertex_index = self._try_reverse_index(
            self.wheel_4a_vertex_method, tag.method, "level_4a_vertex_method"
        )
        if vertex_index is not None:
            self.wheel_4a_vertex_method.update(vertex_index, reward, decay=0.0)

        route_index = self._try_reverse_index(
            self.wheel_4b_route_method, tag.method, "level_4b_route_method"
        )
        if route_index is not None:
            self.wheel_4b_route_method.update(route_index, reward, decay=0.0)

        solver_index = self._try_reverse_index(self.wheel_5_solver, tag.solver, "level_5_solver")
        if solver_index is not None:
            self.wheel_5_solver.update(solver_index, reward, decay=0.0)

    def update_final(self, selection: HAOSSelection, iteration: int) -> None:
        if iteration < self.config.haos_warmup:
            self._immediate_rewards.pop(iteration, None)
            self._deferred_rewards.pop(iteration, None)
            return

        immediate_reward = self._immediate_rewards.pop(iteration, 0.0)
        deferred_map = self._deferred_rewards.pop(iteration, {})
        current_tag = selection.to_tag(iteration)
        current_deferred = deferred_map.pop(current_tag, 0.0)
        total_reward = immediate_reward + current_deferred

        decay = self.config.decay
        self.wheel_1_k.update(selection.k_index, total_reward, decay)
        self.wheel_2_lambda.update(selection.lambda_index, total_reward, decay)
        self.wheel_3_paradigm.update(selection.paradigm_index, total_reward, decay)
        if selection.paradigm == "vertex":
            self.wheel_4a_vertex_method.update(selection.method_index, total_reward, decay)
        else:
            self.wheel_4b_route_method.update(selection.method_index, total_reward, decay)
        self.wheel_5_solver.update(selection.solver_index, total_reward, decay)

        for tag, reward in deferred_map.items():
            self._update_by_tag_all_levels(tag, reward)

    def compute_reward(
        self,
        solution_cost: float | None,
        best_cost: float,
        last_cost: float,
        reward_config: HAOSRewardConfig,
        is_deferred: bool = False,
    ) -> float:
        if solution_cost is None:
            return (
                reward_config.deferred_no_improvement
                if is_deferred
                else reward_config.reward_no_solution
            )
        if solution_cost < best_cost:
            return reward_config.deferred_new_best if is_deferred else reward_config.reward_new_best
        if solution_cost < last_cost:
            return (
                reward_config.deferred_improvement
                if is_deferred
                else reward_config.reward_improvement
            )
        return (
            reward_config.deferred_no_improvement
            if is_deferred
            else reward_config.reward_no_improvement
        )

    def _wheel_state(self, wheel: RouletteWheel) -> dict:
        return {
            "values": list(wheel.choices),
            "raw_weights": wheel.raw_weights(),
            "probabilities": wheel.probabilities(),
            "min_weight": wheel.min_weight,
        }

    def state_dict(self) -> dict:
        return {
            "level_1_k": self._wheel_state(self.wheel_1_k),
            "level_2_lambda_demand": self._wheel_state(self.wheel_2_lambda),
            "level_3_paradigm": self._wheel_state(self.wheel_3_paradigm),
            "level_4a_vertex_method": self._wheel_state(self.wheel_4a_vertex_method),
            "level_4b_route_method": self._wheel_state(self.wheel_4b_route_method),
            "level_5_solver": self._wheel_state(self.wheel_5_solver),
        }

    def load_state_dict(self, state: dict) -> None:
        def load_level(level_name: str, wheel: RouletteWheel) -> None:
            if level_name not in state:
                raise ValueError(f"Missing level in state_dict: {level_name}")
            level_state = state[level_name]
            if level_state.get("values") != list(wheel.choices):
                raise ValueError(f"Choice mismatch for {level_name}")
            raw_weights = level_state.get("raw_weights")
            if not isinstance(raw_weights, list):
                raise ValueError(f"raw_weights missing/invalid for {level_name}")
            wheel.set_weights(raw_weights)

        load_level("level_1_k", self.wheel_1_k)
        load_level("level_2_lambda_demand", self.wheel_2_lambda)
        load_level("level_3_paradigm", self.wheel_3_paradigm)
        load_level("level_4a_vertex_method", self.wheel_4a_vertex_method)
        load_level("level_4b_route_method", self.wheel_4b_route_method)
        load_level("level_5_solver", self.wheel_5_solver)
