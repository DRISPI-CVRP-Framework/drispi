from __future__ import annotations

import random

import pytest

from drispi.haos.wheel import WEIGHT_FLOOR, RouletteWheel


def test_wheel_initializes_with_floor_weights() -> None:
    wheel = RouletteWheel(choices=["a", "b", "c"], min_weight=0.05)
    assert wheel.raw_weights() == [WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR]
    assert WEIGHT_FLOOR == 10.0


def test_select_returns_valid_index_and_value() -> None:
    wheel = RouletteWheel(choices=[10, 20, 30], min_weight=0.05)
    index, value = wheel.select(random.Random(7))
    assert index in (0, 1, 2)
    assert value == wheel.choices[index]


def test_probabilities_sum_to_one() -> None:
    wheel = RouletteWheel(choices=["x", "y", "z"], min_weight=0.05)
    probs = wheel.probabilities()
    assert sum(probs) == pytest.approx(1.0)


def test_floor_enforced_in_probabilities() -> None:
    wheel = RouletteWheel(choices=["a", "b", "c", "d"], min_weight=0.10)
    wheel.set_weights([1000.0, 1.0, 1.0, 1.0])
    probs = wheel.probabilities()
    raw = [1000.0 / 1003.0, 1.0 / 1003.0, 1.0 / 1003.0, 1.0 / 1003.0]
    clamped = [max(p, 0.10) for p in raw]
    expected = [p / sum(clamped) for p in clamped]
    assert probs == pytest.approx(expected)
    assert sum(probs) == pytest.approx(1.0)


def test_high_reward_choice_gets_higher_probability() -> None:
    wheel = RouletteWheel(choices=["a", "b", "c"], min_weight=0.01)
    for _ in range(100):
        wheel.update(index=0, reward=5.0)
    probs = wheel.probabilities()
    assert probs[0] > probs[1]
    assert probs[0] > probs[2]


def test_update_adds_reward_to_selected_index_only() -> None:
    wheel = RouletteWheel(choices=["a", "b", "c"], min_weight=0.05)
    before = wheel.raw_weights()
    wheel.update(index=1, reward=2.0)
    after = wheel.raw_weights()
    assert after[1] == pytest.approx(before[1] + 2.0)
    assert after[0] == before[0]
    assert after[2] == before[2]


def test_decay_all_decays_every_weight() -> None:
    wheel = RouletteWheel(choices=["a", "b", "c"], min_weight=0.05)
    wheel.set_weights([20.0, 40.0, 100.0])
    wheel.decay_all(0.95)
    assert wheel.raw_weights() == pytest.approx([19.0, 38.0, 95.0])


def test_decay_all_never_drops_below_floor() -> None:
    wheel = RouletteWheel(choices=["a", "b", "c"], min_weight=0.05)
    wheel.set_weights([10.0, 10.5, 200.0])
    wheel.decay_all(0.5)
    assert wheel.raw_weights() == pytest.approx([WEIGHT_FLOOR, WEIGHT_FLOOR, 100.0])


def test_update_cannot_push_weight_below_floor() -> None:
    wheel = RouletteWheel(choices=["a", "b"], min_weight=0.05)
    wheel.update(index=0, reward=-100.0)
    assert wheel.raw_weights()[0] == WEIGHT_FLOOR


def test_set_weights_validation_errors() -> None:
    wheel = RouletteWheel(choices=["a", "b"], min_weight=0.05)
    with pytest.raises(ValueError):
        wheel.set_weights([1.0])
    with pytest.raises(ValueError):
        wheel.set_weights([1.0, -1.0])


def test_single_choice_wheel_always_returns_only_choice() -> None:
    wheel = RouletteWheel(choices=["only"], min_weight=0.0)
    for _ in range(10):
        index, value = wheel.select(random.Random(123))
        assert index == 0
        assert value == "only"
