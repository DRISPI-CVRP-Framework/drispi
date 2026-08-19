"""Tests for the scale-adaptive HAOS level-1 k domain."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from drispi.haos.k_domain import DEFAULT_BASE_ARMS, k_domain, parse_n_kmin

XL_DIR = Path("data/instances/xl")

# Computed golden rows (ceil k_lo as a filter, not an arm).
GOLDEN: dict[tuple[int, int], list[int]] = {
    (1048, 237): [2, 3, 4, 6, 8, 10, 12],
    (1654, 11): [2, 3, 4, 6, 8, 10, 11],
    (2914, 95): [2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
    (5061, 184): [3, 4, 6, 8, 10, 12, 15, 19, 24, 30],
    (7037, 38): [6, 8, 10, 12, 15, 18, 21, 26, 31, 38],
    (9571, 55): [8, 10, 12, 15, 19, 23, 29, 36, 44, 55],
    (10001, 1570): [8, 10, 12, 15, 19, 24, 30, 38, 48, 60],
}


def test_parse_n_kmin() -> None:
    assert parse_n_kmin("XL-n5288-k1246") == (5288, 1246)
    assert parse_n_kmin("X-n1001-k43") == (1001, 43)
    assert parse_n_kmin("XL-n10001-k1570_sub_512") == (10001, 1570)
    assert parse_n_kmin("toy_grid_12") is None
    assert parse_n_kmin("") is None


@pytest.mark.parametrize(("n", "k_min"), sorted(GOLDEN))
def test_k_domain_golden_rows(n: int, k_min: int) -> None:
    assert k_domain(n, k_min) == GOLDEN[(n, k_min)]


def test_k1_never_in_domain_for_real_instances() -> None:
    for (n, k_min), dom in GOLDEN.items():
        assert 1 not in dom, (n, k_min)


def test_k_lo_is_a_filter_not_an_arm() -> None:
    # XL-n10001: k_lo = ceil((0.980 * 10001 / 2500) ** (1 / 0.725)) = 7, and 7
    # is not an arm — the smallest surviving base arm is 8.
    k_lo = math.ceil((0.980 * 10001 / 2500.0) ** (1.0 / 0.725))
    assert k_lo == 7
    assert k_domain(10001, 1570)[0] == 8


def test_small_k_min_appended_as_top_arm() -> None:
    # K_min = 11 < 12: the fleet size itself becomes the top arm.
    assert k_domain(1654, 11) == [2, 3, 4, 6, 8, 10, 11]


def test_k_min_one_collapses_to_single_arm() -> None:
    assert k_domain(50, 1) == [1]


def test_all_100_xl_instances_produce_valid_domains() -> None:
    paths = sorted(XL_DIR.glob("*.vrp"))
    assert len(paths) == 100
    for path in paths:
        parsed = parse_n_kmin(path.stem)
        assert parsed is not None, path.stem
        n, k_min = parsed
        dom = k_domain(n, k_min)
        assert dom, path.stem
        assert dom == sorted(set(dom)), path.stem
        assert len(dom) <= 10, path.stem
        assert 1 not in dom, path.stem
        assert all(2 <= k <= k_min for k in dom), path.stem
        # Base arms surviving the k_lo filter must all be retained.
        k_lo = min(
            max(2, math.ceil((0.980 * n / 2500.0) ** (1.0 / 0.725))),
            min(6 * max(1, round(n / 1000)), k_min),
        )
        expected_base = [k for k in DEFAULT_BASE_ARMS if k_lo <= k <= k_min]
        assert [k for k in dom if k in expected_base] == expected_base, path.stem


def test_domain_is_deterministic() -> None:
    assert k_domain(7037, 38) == k_domain(7037, 38)


def test_ext_per_1000_zero_disables_extension_but_keeps_k_lo_filter() -> None:
    # Pilot cell A (drop-low, cap 12): base arms filtered by k_lo, no extension.
    assert k_domain(10001, 1570, ext_per_1000=0) == [8, 10, 12]
    assert k_domain(1048, 237, ext_per_1000=0) == [2, 3, 4, 6, 8, 10, 12]


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="n must"):
        k_domain(0, 5)
    with pytest.raises(ValueError, match="k_min"):
        k_domain(100, 0)
    with pytest.raises(ValueError, match="max_arms"):
        k_domain(100, 5, max_arms=0)
