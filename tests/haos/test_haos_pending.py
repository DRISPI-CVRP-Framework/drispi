"""Pending-selection registry and decay policy (b) for async BG-AILS."""

from __future__ import annotations

import random

import pytest

from drispi.core.instance import CVRPInstance
from drispi.haos.config import HAOSConfig
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


def _fresh(warmup: int = 0, decay: float = 0.95) -> HAOS:
    return HAOS(config=HAOSConfig(haos_warmup=warmup, decay=decay), instance=make_instance_20())


def test_apply_pending_immediate_credits_registered_arms_without_redecay() -> None:
    haos = _fresh()
    selection = haos.select(iteration=0, rng=random.Random(1))
    haos.register_selection(0, selection)
    # End of iteration 0: global decay on schedule (unrewarded arms floor at 10).
    haos.decay_on_schedule(0, selection)
    # BG(0) result arrives during iteration 1: immediate credit, no second decay.
    credited = haos.apply_pending_immediate(0, 2.0)
    assert credited == 0
    state = haos.state_dict()
    # Decay floored the weight back to 10.0, then +2.0 with no further decay.
    assert state["level_1_k"]["raw_weights"][selection.k_index] == pytest.approx(12.0)
    assert state["level_5_solver"]["raw_weights"][selection.solver_index] == pytest.approx(12.0)


def test_apply_pending_immediate_is_consume_once() -> None:
    haos = _fresh()
    selection = haos.select(iteration=0, rng=random.Random(2))
    haos.register_selection(0, selection)
    haos.decay_on_schedule(0, selection)
    assert haos.apply_pending_immediate(0, 2.0) == 0
    after_first = haos.state_dict()
    # Second drain of the same iteration must be a no-op.
    assert haos.apply_pending_immediate(0, 2.0) is None
    assert haos.state_dict() == after_first


def test_clear_pending_drops_credit_without_wheel_update() -> None:
    haos = _fresh()
    selection = haos.select(iteration=3, rng=random.Random(3))
    haos.register_selection(3, selection)
    before = haos.state_dict()
    haos.clear_pending(3)
    assert haos.apply_pending_immediate(3, 99.0) is None
    assert haos.state_dict() == before


def test_warmup_decay_on_schedule_flushes_pending_with_no_credit() -> None:
    haos = _fresh(warmup=5)
    selection = haos.select(iteration=0, rng=random.Random(4))
    haos.register_selection(0, selection)
    before = haos.state_dict()
    haos.decay_on_schedule(0, selection)
    assert haos.state_dict() == before  # no decay, no credit during warmup
    # The pending entry was flushed: a late BG(0) reward credits nothing.
    assert haos.apply_pending_immediate(0, 50.0) is None
    assert haos.state_dict() == before


def test_decay_on_schedule_applies_deferred_and_keeps_pending() -> None:
    haos = _fresh()
    selection = haos.select(iteration=0, rng=random.Random(5))
    historical = haos.select(iteration=7, rng=random.Random(6)).to_tag(iteration=7)
    haos.register_selection(0, selection)
    haos.update_deferred([historical], iteration=0, deferred_reward=3.0)
    haos.decay_on_schedule(0)  # selection resolved from the registry
    state = haos.state_dict()
    # Deferred landed ((10 + 3) * 0.95 = 12.35 on the historical k arm).
    k_idx = haos.wheel_1_k.choices.index(historical.k) if historical.k in haos.wheel_1_k.choices else None
    if k_idx is not None:
        assert state["level_1_k"]["raw_weights"][k_idx] == pytest.approx(12.35)
    # Pending selection survives for the late immediate credit.
    assert haos.apply_pending_immediate(0, 2.0) == 0


def test_update_final_pops_pending_registry() -> None:
    """Sync flush must not leak registry entries when selection was registered."""
    haos = _fresh()
    selection = haos.select(iteration=0, rng=random.Random(7))
    haos.register_selection(0, selection)
    haos.update_immediate(selection, iteration=0, reward=2.0)
    haos.update_final(selection, iteration=0)
    state = haos.state_dict()
    # Same result as the historical combined path: (10 + 2) * 0.95 = 11.4.
    assert state["level_1_k"]["raw_weights"][selection.k_index] == pytest.approx(11.4)
    # Registry is empty: a late apply credits nothing.
    assert haos.apply_pending_immediate(0, 50.0) is None
