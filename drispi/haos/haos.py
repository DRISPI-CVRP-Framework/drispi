from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace

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

    def to_tag(self, iteration: int, *, is_improvement_route: bool = False) -> HAOSTag:
        return HAOSTag(
            k=self.k,
            lambda_demand=self.lambda_demand,
            paradigm=self.paradigm,
            method=self.method,
            solver=self.solver,
            iteration=iteration,
            is_improvement_route=is_improvement_route,
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

        self.k_values = config.compute_k_values(instance)
        sw = config.starting_weight
        self.wheel_1_k = RouletteWheel(self.k_values, config.min_weight_k, starting_weight=sw)
        self.wheel_2_lambda = RouletteWheel(
            config.lambda_demand_values,
            config.min_weight_lambda,
            starting_weight=sw,
        )
        self.wheel_3_paradigm = RouletteWheel(
            config.paradigm_values,
            config.min_weight_paradigm,
            starting_weight=sw,
        )
        self.wheel_4a_vertex_method = RouletteWheel(
            config.vertex_method_values,
            config.min_weight_vertex_method,
            starting_weight=sw,
        )
        self.wheel_4b_route_method = RouletteWheel(
            config.route_method_values,
            config.min_weight_route_method,
            starting_weight=sw,
        )
        self.wheel_5_solver = RouletteWheel(
            config.solver_values, config.min_weight_solver, starting_weight=sw
        )

        self._immediate_rewards: dict[int, float] = {}
        self._deferred_rewards: dict[int, dict[HAOSTag, float]] = {}
        self._pending_selections: dict[int, HAOSSelection] = {}

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

    def coerce_vertex_when_no_routes(
        self,
        selection: HAOSSelection,
        *,
        best_solution_available: bool,
        rng: random.Random,
    ) -> HAOSSelection:
        """
        When route paradigm was rolled but S* is unavailable, switch to vertex
        paradigm and re-draw a vertex clustering method.
        """
        if selection.paradigm != "route" or best_solution_available:
            return selection

        vertex_paradigm_index = self.wheel_3_paradigm.choices.index("vertex")
        method_index, method = self.wheel_4a_vertex_method.select(rng)
        return replace(
            selection,
            paradigm="vertex",
            method=method,
            paradigm_index=vertex_paradigm_index,
            method_index=method_index,
        )

    def cap_k_for_route_clustering(
        self,
        selection: HAOSSelection,
        n_routes: int,
    ) -> HAOSSelection:
        """Clip k to the live route count for route-paradigm clustering.

        Never raises: with a scale-adaptive domain the rolled k can exceed the
        incumbent's route count, so the partition just uses fewer clusters.
        ``k_index`` is kept so HAOS credit still lands on the arm that was
        actually rolled.
        """
        if selection.paradigm != "route" or selection.k <= n_routes:
            return selection
        return replace(selection, k=max(1, n_routes))

    def joint_probability(self, selection: HAOSSelection) -> float:
        """Product of per-level roulette probabilities for the current selection."""
        method_wheel = (
            self.wheel_4a_vertex_method
            if selection.paradigm == "vertex"
            else self.wheel_4b_route_method
        )
        p_k = self.wheel_1_k.probabilities()[selection.k_index]
        p_l = self.wheel_2_lambda.probabilities()[selection.lambda_index]
        p_p = self.wheel_3_paradigm.probabilities()[selection.paradigm_index]
        p_m = method_wheel.probabilities()[selection.method_index]
        p_s = self.wheel_5_solver.probabilities()[selection.solver_index]
        return p_k * p_l * p_p * p_m * p_s

    def format_operator_roll(self, iteration: int, selection: HAOSSelection) -> str:
        """Human-readable HAOS draw: chosen operators and per-level roulette probabilities."""
        method_wheel = (
            self.wheel_4a_vertex_method
            if selection.paradigm == "vertex"
            else self.wheel_4b_route_method
        )
        p_k = self.wheel_1_k.probabilities()[selection.k_index]
        p_l = self.wheel_2_lambda.probabilities()[selection.lambda_index]
        p_p = self.wheel_3_paradigm.probabilities()[selection.paradigm_index]
        p_m = method_wheel.probabilities()[selection.method_index]
        p_s = self.wheel_5_solver.probabilities()[selection.solver_index]
        p_joint = p_k * p_l * p_p * p_m * p_s
        return (
            f"[it {iteration}] HAOS Roll: "
            f"k={selection.k} (p={p_k:.4f}); "
            f"λ={selection.lambda_demand:.3f} (p={p_l:.4f}); "
            f"paradigm={selection.paradigm} (p={p_p:.4f}); "
            f"method={selection.method} (p={p_m:.4f}); "
            f"solver={selection.solver} (p={p_s:.4f}); "
            f"P≈{p_joint:.6f}"
        )

    def register_selection(self, iteration: int, selection: HAOSSelection) -> None:
        """Retain the rolled selection so a late reward credits iteration i's arms."""
        self._pending_selections[iteration] = selection

    def clear_pending(self, iteration: int) -> None:
        """Drop pending selection and reward accumulators with no wheel credit.

        Used for infra failures (worker crash, discard-at-cap, launch skipped):
        clear-on-flush is unconditional so the registry never leaks.
        """
        self._pending_selections.pop(iteration, None)
        self._immediate_rewards.pop(iteration, None)
        self._deferred_rewards.pop(iteration, None)

    def apply_pending_immediate(self, iteration: int, reward: float) -> int | None:
        """
        Credit ``reward`` to the stored selection for ``iteration`` (immediate only).

        Pops the pending selection (consume-once). Returns the iteration credited,
        or ``None`` when nothing was credited (no pending entry, or warmup).
        During warmup the entry is still cleared but no wheel is updated.
        """
        selection = self._pending_selections.pop(iteration, None)
        self._immediate_rewards.pop(iteration, None)
        if selection is None:
            return None
        if iteration < self.config.haos_warmup:
            return None
        self._credit_selection(selection, float(reward))
        return iteration

    def decay_on_schedule(self, iteration: int, selection: HAOSSelection | None = None) -> None:
        """
        Apply deferred rewards for ``iteration``, then run global decay (post-warmup).

        The immediate component is NOT handled here — it lands via
        :meth:`apply_pending_immediate` (async) or was already credited through
        it (sync). Uses ``selection`` if given, otherwise the pending registry
        (without popping — async still needs the entry for the late immediate).
        During warmup: clear deferred with no credit and no decay, matching the
        old ``update_final`` early return.
        """
        deferred_map = self._deferred_rewards.pop(iteration, {})
        sel = selection if selection is not None else self._pending_selections.get(iteration)

        if iteration < self.config.haos_warmup:
            self._pending_selections.pop(iteration, None)
            self._immediate_rewards.pop(iteration, None)
            return

        if sel is not None:
            current_tag = sel.to_tag(iteration)
            current_deferred = float(deferred_map.pop(current_tag, 0.0))
            if current_deferred:
                self._credit_selection(sel, current_deferred)
        for tag, reward in deferred_map.items():
            self._update_by_tag_all_levels(tag, float(reward))

        self._decay_all_wheels()

    def _credit_selection(self, selection: HAOSSelection, reward: float) -> None:
        self.wheel_1_k.update(selection.k_index, reward)
        self.wheel_2_lambda.update(selection.lambda_index, reward)
        self.wheel_3_paradigm.update(selection.paradigm_index, reward)
        if selection.paradigm == "vertex":
            self.wheel_4a_vertex_method.update(selection.method_index, reward)
        else:
            self.wheel_4b_route_method.update(selection.method_index, reward)
        self.wheel_5_solver.update(selection.solver_index, reward)

    def _decay_all_wheels(self) -> None:
        decay = self.config.decay
        for wheel in (
            self.wheel_1_k,
            self.wheel_2_lambda,
            self.wheel_3_paradigm,
            self.wheel_4a_vertex_method,
            self.wheel_4b_route_method,
            self.wheel_5_solver,
        ):
            wheel.decay_all(decay)

    def update_immediate(self, selection: HAOSSelection, iteration: int, reward: float) -> None:
        del selection
        if iteration < self.config.haos_warmup:
            return
        self._immediate_rewards[iteration] = self._immediate_rewards.get(iteration, 0.0) + reward

    def flush_immediate(self, selection: HAOSSelection, iteration: int) -> float:
        """Credit accumulated immediate rewards for ``iteration`` to ``selection``.

        Async-path counterpart of the immediate part of :meth:`update_final`:
        the pending registry is left untouched so the late BG-AILS credit via
        :meth:`apply_pending_immediate` still lands on the same arms.
        Returns the credited amount (0.0 during warmup or when nothing accrued).
        """
        reward = float(self._immediate_rewards.pop(iteration, 0.0))
        if iteration < self.config.haos_warmup:
            return 0.0
        if reward:
            self._credit_selection(selection, reward)
        return reward

    def update_deferred(
        self,
        contributing_tags: list[HAOSTag],
        iteration: int,
        deferred_reward: float,
    ) -> None:
        if iteration < self.config.haos_warmup:
            return
        tags_for_iteration = self._deferred_rewards.setdefault(iteration, {})
        filtered = (t for t in contributing_tags if not t.is_improvement_route)
        for tag in set(filtered):
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
            self.wheel_1_k.update(k_index, reward)

        lambda_index = self._try_reverse_index(
            self.wheel_2_lambda, tag.lambda_demand, "level_2_lambda_demand"
        )
        if lambda_index is not None:
            self.wheel_2_lambda.update(lambda_index, reward)

        paradigm_index = self._try_reverse_index(
            self.wheel_3_paradigm, tag.paradigm, "level_3_paradigm"
        )
        if paradigm_index is not None:
            self.wheel_3_paradigm.update(paradigm_index, reward)

        # The method belongs to exactly one wheel, determined by the tag's
        # paradigm; updating both would double-credit methods present in both.
        if tag.paradigm == "vertex":
            method_wheel = self.wheel_4a_vertex_method
            method_level = "level_4a_vertex_method"
        else:
            method_wheel = self.wheel_4b_route_method
            method_level = "level_4b_route_method"
        method_index = self._try_reverse_index(method_wheel, tag.method, method_level)
        if method_index is not None:
            method_wheel.update(method_index, reward)

        solver_index = self._try_reverse_index(self.wheel_5_solver, tag.solver, "level_5_solver")
        if solver_index is not None:
            self.wheel_5_solver.update(solver_index, reward)

    def update_final(self, selection: HAOSSelection, iteration: int) -> None:
        """Sync-path flush: immediate credit, then deferred + decay.

        Equivalent to the old combined update because wheel updates are additive
        with a floor and all configured rewards are >= 0 (floor cannot bind
        between the two component updates).
        """
        self._pending_selections.pop(iteration, None)
        if iteration < self.config.haos_warmup:
            self._immediate_rewards.pop(iteration, None)
            self._deferred_rewards.pop(iteration, None)
            return

        immediate_reward = self._immediate_rewards.pop(iteration, 0.0)
        if immediate_reward:
            self._credit_selection(selection, immediate_reward)
        self.decay_on_schedule(iteration, selection)

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
