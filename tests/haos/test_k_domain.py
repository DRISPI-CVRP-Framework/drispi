"""Tests for the scale-adaptive HAOS level-1 k domain."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from drispi.haos.k_domain import DEFAULT_BASE_ARMS, k_domain, parse_n_kmin

XL_DIR = Path("data/instances/xl")
THIN_BASE = [2, 4, 8, 12]

# Golden rows at ext_per_1000=4 with clamps disabled (0, 1) — the pre-change domain.
GOLDEN: dict[tuple[int, int], list[int]] = {
    (1048, 237): [2, 3, 4, 6, 8, 10, 12],
    (1654, 11): [2, 3, 4, 6, 8, 10, 11],
    (5061, 184): [2, 3, 4, 6, 8, 10, 12, 14, 17, 20],
    (7037, 38): [2, 3, 4, 6, 8, 10, 12, 16, 21, 28],
    (8028, 294): [2, 3, 4, 6, 8, 10, 12, 17, 23, 32],
    (10001, 1570): [2, 3, 4, 6, 8, 10, 12, 18, 27, 40],
}

MUST_CHANGE_OLD: dict[tuple[int, int], list[int]] = {
    (9571, 55): [2, 3, 4, 6, 8, 10, 12, 18, 27, 40],
    (8207, 108): [2, 3, 4, 6, 8, 10, 12, 17, 23, 32],
    (7037, 38): [2, 3, 4, 6, 8, 10, 12, 16, 21, 28],
    (6034, 61): [2, 3, 4, 6, 8, 10, 12, 15, 19, 24],
    (5174, 55): [2, 3, 4, 6, 8, 10, 12, 14, 17, 20],
    (4436, 48): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (3804, 29): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (3721, 77): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (9363, 209): [2, 3, 4, 6, 8, 10, 12, 17, 25, 36],
    (6884, 148): [2, 3, 4, 6, 8, 10, 12, 16, 21, 28],
    (5902, 122): [2, 3, 4, 6, 8, 10, 12, 15, 19, 24],
    (3561, 229): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (3640, 211): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (3888, 1010): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (3975, 687): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (4063, 347): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (4153, 291): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (4245, 203): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
    (4340, 148): [2, 3, 4, 6, 8, 10, 12, 13, 15, 16],
}

MUST_CHANGE: dict[tuple[int, int], list[int]] = {
    (9571, 55): [2, 3, 4, 6, 8, 10, 12],
    (8207, 108): [2, 3, 4, 6, 8, 10, 12],
    (7037, 38): [2, 3, 4, 6, 8, 10, 12],
    (6034, 61): [2, 3, 4, 6, 8, 10, 12],
    (5174, 55): [2, 3, 4, 6, 8, 10, 12],
    (4436, 48): [2, 3, 4, 6, 8, 10, 12],
    (3804, 29): [2, 3, 4, 6, 8, 10, 12],
    (3721, 77): [2, 3, 4, 6, 8, 10, 12],
    (9363, 209): [2, 3, 4, 6, 8, 10, 12, 16, 20, 26],
    (6884, 148): [2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
    (5902, 122): [2, 3, 4, 6, 8, 10, 12, 14],
    (3561, 229): [2, 3, 4, 6, 8, 10, 12, 15],
    (3640, 211): [2, 3, 4, 6, 8, 10, 12, 15],
    (3888, 1010): [2, 3, 4, 6, 8, 10, 12, 15],
    (3975, 687): [2, 3, 4, 6, 8, 10, 12, 15],
    (4063, 347): [2, 3, 4, 6, 8, 10, 12, 15],
    (4153, 291): [2, 3, 4, 6, 8, 10, 12, 15],
    (4245, 203): [2, 3, 4, 6, 8, 10, 12, 15],
    (4340, 148): [2, 3, 4, 6, 8, 10, 12, 15],
}

MUST_NOT_CHANGE: dict[tuple[int, int], list[int]] = {
    (9784, 2774): [2, 3, 4, 6, 8, 10, 12, 18, 27, 40],
    (10001, 1570): [2, 3, 4, 6, 8, 10, 12, 18, 27, 40],
    (8389, 2028): [2, 3, 4, 6, 8, 10, 12, 17, 23, 32],
    (7193, 1683): [2, 3, 4, 6, 8, 10, 12, 16, 21, 28],
    (5288, 1246): [2, 3, 4, 6, 8, 10, 12, 14, 17, 20],
    (1048, 237): [2, 3, 4, 6, 8, 10, 12],
    (1094, 157): [2, 3, 4, 6, 8, 10, 12],
    (3287, 30): [2, 3, 4, 6, 8, 10, 12],
    (2634, 17): [2, 3, 4, 6, 8, 10, 12],
    (1654, 11): [2, 3, 4, 6, 8, 10, 11],
}


def test_parse_n_kmin() -> None:
    assert parse_n_kmin("XL-n5288-k1246") == (5288, 1246)
    assert parse_n_kmin("X-n1001-k43") == (1001, 43)
    assert parse_n_kmin("XL-n10001-k1570_sub_512") == (10001, 1570)
    assert parse_n_kmin("toy_grid_12") is None
    assert parse_n_kmin("") is None


@pytest.mark.parametrize(("n", "k_min"), sorted(GOLDEN))
def test_k_domain_golden_rows(n: int, k_min: int) -> None:
    assert (
        k_domain(n, k_min, min_routes_per_cluster=0, min_arm_spacing=1)
        == GOLDEN[(n, k_min)]
    )


def test_k1_never_in_default_domain_for_real_instances() -> None:
    for (n, k_min), dom in GOLDEN.items():
        assert 1 not in dom, (n, k_min)


def test_small_k_min_appended_as_top_arm() -> None:
    # K_min = 11 < 12: the fleet size itself becomes the top arm.
    assert k_domain(1654, 11, min_routes_per_cluster=0, min_arm_spacing=1) == [
        2,
        3,
        4,
        6,
        8,
        10,
        11,
    ]


def test_k_min_one_collapses_to_single_arm() -> None:
    assert k_domain(50, 1, min_routes_per_cluster=0, min_arm_spacing=1) == [1]


def test_all_100_xl_instances_produce_valid_domains() -> None:
    paths = sorted(XL_DIR.glob("*.vrp"))
    assert len(paths) == 100
    for path in paths:
        parsed = parse_n_kmin(path.stem)
        assert parsed is not None, path.stem
        n, k_min = parsed
        dom = k_domain(n, k_min, min_routes_per_cluster=0, min_arm_spacing=1)
        assert dom, path.stem
        assert dom == sorted(set(dom)), path.stem
        assert len(dom) <= 10, path.stem
        assert all(k <= k_min for k in dom), path.stem
        if 1 in dom:
            assert k_min == 1, path.stem
        expected_base = [k for k in DEFAULT_BASE_ARMS if k <= k_min]
        assert [k for k in dom if k in expected_base] == expected_base, path.stem


def test_domain_is_deterministic() -> None:
    assert k_domain(
        7037, 38, min_routes_per_cluster=0, min_arm_spacing=1
    ) == k_domain(7037, 38, min_routes_per_cluster=0, min_arm_spacing=1)


def test_ext_per_1000_zero_is_a_noop_full_base() -> None:
    # Extension off: full base survives, no geometric ladder.
    assert k_domain(
        10001, 1570, ext_per_1000=0, min_routes_per_cluster=0, min_arm_spacing=1
    ) == [2, 3, 4, 6, 8, 10, 12]
    assert k_domain(
        1048, 237, ext_per_1000=0, min_routes_per_cluster=0, min_arm_spacing=1
    ) == [2, 3, 4, 6, 8, 10, 12]
    assert k_domain(
        7037, 38, ext_per_1000=0, min_routes_per_cluster=0, min_arm_spacing=1
    ) == [2, 3, 4, 6, 8, 10, 12]


def test_cell_0prime_base_includes_k1() -> None:
    assert k_domain(
        10001,
        1570,
        base_arms=[1, 2, 3, 4, 6, 8, 10, 12],
        ext_per_1000=0,
        min_routes_per_cluster=0,
        min_arm_spacing=1,
    ) == [1, 2, 3, 4, 6, 8, 10, 12]


def test_c6_ext6_pins() -> None:
    assert k_domain(
        5061, 184, ext_per_1000=6, min_routes_per_cluster=0, min_arm_spacing=1
    ) == [2, 3, 4, 6, 8, 10, 12, 16, 22, 30]
    assert k_domain(
        7037, 38, ext_per_1000=6, min_routes_per_cluster=0, min_arm_spacing=1
    ) == [2, 3, 4, 6, 8, 10, 12, 18, 26, 38]
    assert k_domain(
        8028, 294, ext_per_1000=6, min_routes_per_cluster=0, min_arm_spacing=1
    ) == [2, 3, 4, 6, 8, 10, 12, 19, 30, 48]
    assert k_domain(
        10001, 1570, ext_per_1000=6, min_routes_per_cluster=0, min_arm_spacing=1
    ) == [2, 3, 4, 6, 8, 10, 12, 21, 35, 60]


def test_c6_thin_confirmation_pins() -> None:
    assert k_domain(
        7037,
        38,
        base_arms=THIN_BASE,
        ext_per_1000=6,
        min_routes_per_cluster=0,
        min_arm_spacing=1,
    ) == [2, 4, 8, 12, 15, 18, 21, 26, 31, 38]
    assert k_domain(
        8028,
        294,
        base_arms=THIN_BASE,
        ext_per_1000=6,
        min_routes_per_cluster=0,
        min_arm_spacing=1,
    ) == [2, 4, 8, 12, 15, 19, 24, 30, 38, 48]
    assert k_domain(
        10001,
        1570,
        base_arms=THIN_BASE,
        ext_per_1000=6,
        min_routes_per_cluster=0,
        min_arm_spacing=1,
    ) == [2, 4, 8, 12, 16, 21, 27, 35, 46, 60]


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="n must"):
        k_domain(0, 5, min_routes_per_cluster=0, min_arm_spacing=1)
    with pytest.raises(ValueError, match="k_min"):
        k_domain(100, 0, min_routes_per_cluster=0, min_arm_spacing=1)
    with pytest.raises(ValueError, match="max_arms"):
        k_domain(100, 5, max_arms=0, min_routes_per_cluster=0, min_arm_spacing=1)
    with pytest.raises(ValueError, match="min_routes_per_cluster"):
        k_domain(100, 5, min_routes_per_cluster=-1, min_arm_spacing=1)
    with pytest.raises(ValueError, match="min_arm_spacing"):
        k_domain(100, 5, min_routes_per_cluster=0, min_arm_spacing=0)


@pytest.mark.parametrize(("n", "k_min"), sorted(MUST_CHANGE))
def test_shipping_must_change(n: int, k_min: int) -> None:
    assert (
        k_domain(n, k_min, min_routes_per_cluster=8, min_arm_spacing=2)
        == MUST_CHANGE[(n, k_min)]
    )


@pytest.mark.parametrize(("n", "k_min"), sorted(MUST_NOT_CHANGE))
def test_shipping_must_not_change(n: int, k_min: int) -> None:
    assert (
        k_domain(n, k_min, min_routes_per_cluster=8, min_arm_spacing=2)
        == MUST_NOT_CHANGE[(n, k_min)]
    )


def test_rejected_slots_are_not_refilled() -> None:
    assert k_domain(
        5902, 122, min_routes_per_cluster=8, min_arm_spacing=2
    ) == [2, 3, 4, 6, 8, 10, 12, 14]
    assert k_domain(
        8207, 108, min_routes_per_cluster=8, min_arm_spacing=2
    ) == [2, 3, 4, 6, 8, 10, 12]


def _xl_pairs() -> list[tuple[int, int]]:
    paths = sorted(XL_DIR.glob("*.vrp"))
    assert len(paths) == 100
    pairs: list[tuple[int, int]] = []
    for path in paths:
        parsed = parse_n_kmin(path.stem)
        assert parsed is not None, path.stem
        pairs.append(parsed)
    return pairs


def test_invariants_i1_to_i6_over_xl() -> None:
    pairs = _xl_pairs()
    n_differ = 0
    arm_counts: Counter[int] = Counter()
    for n, k_min in pairs:
        old = k_domain(n, k_min, min_routes_per_cluster=0, min_arm_spacing=1)
        new = k_domain(n, k_min, min_routes_per_cluster=8, min_arm_spacing=2)
        assert max(new) <= max(old), (n, k_min, old, new)
        assert len(new) <= len(old), (n, k_min, old, new)
        expected_base = [k for k in DEFAULT_BASE_ARMS if k <= k_min]
        assert all(k in new for k in expected_base), (n, k_min, new)
        top_base = max(expected_base) if expected_base else 0
        above = [k for k in new if k > top_base]
        for left, right in zip(above, above[1:]):
            assert right - left >= 2, (n, k_min, new)
        assert k_min / max(new) >= k_min / max(old), (n, k_min, old, new)
        if new != old:
            n_differ += 1
        arm_counts[len(new)] += 1
    assert n_differ == 19
    assert dict(arm_counts) == {7: 60, 8: 9, 10: 31}


def test_i6_zero_one_reproduces_prechange_pins() -> None:
    for (n, k_min), expected in MUST_CHANGE_OLD.items():
        assert (
            k_domain(n, k_min, min_routes_per_cluster=0, min_arm_spacing=1)
            == expected
        )
    for (n, k_min), expected in MUST_NOT_CHANGE.items():
        assert (
            k_domain(n, k_min, min_routes_per_cluster=0, min_arm_spacing=1)
            == expected
        )
