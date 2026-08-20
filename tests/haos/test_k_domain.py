"""Tests for the scale-adaptive HAOS level-1 k domain."""

from __future__ import annotations

from pathlib import Path

import pytest

from drispi.haos.k_domain import DEFAULT_BASE_ARMS, k_domain, parse_n_kmin

XL_DIR = Path("data/instances/xl")
THIN_BASE = [2, 4, 8, 12]

# Golden rows at ext_per_1000=4 (shipping default / confirmation cell C4).
GOLDEN: dict[tuple[int, int], list[int]] = {
    (1048, 237): [2, 3, 4, 6, 8, 10, 12],
    (1654, 11): [2, 3, 4, 6, 8, 10, 11],
    (5061, 184): [2, 3, 4, 6, 8, 10, 12, 14, 17, 20],
    (7037, 38): [2, 3, 4, 6, 8, 10, 12, 16, 21, 28],
    (8028, 294): [2, 3, 4, 6, 8, 10, 12, 17, 23, 32],
    (10001, 1570): [2, 3, 4, 6, 8, 10, 12, 18, 27, 40],
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


def test_k1_never_in_default_domain_for_real_instances() -> None:
    for (n, k_min), dom in GOLDEN.items():
        assert 1 not in dom, (n, k_min)


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
        assert all(k <= k_min for k in dom), path.stem
        if 1 in dom:
            assert k_min == 1, path.stem
        expected_base = [k for k in DEFAULT_BASE_ARMS if k <= k_min]
        assert [k for k in dom if k in expected_base] == expected_base, path.stem


def test_domain_is_deterministic() -> None:
    assert k_domain(7037, 38) == k_domain(7037, 38)


def test_ext_per_1000_zero_is_a_noop_full_base() -> None:
    # Extension off: full base survives, no geometric ladder.
    assert k_domain(10001, 1570, ext_per_1000=0) == [2, 3, 4, 6, 8, 10, 12]
    assert k_domain(1048, 237, ext_per_1000=0) == [2, 3, 4, 6, 8, 10, 12]
    assert k_domain(7037, 38, ext_per_1000=0) == [2, 3, 4, 6, 8, 10, 12]


def test_cell_0prime_base_includes_k1() -> None:
    assert k_domain(
        10001, 1570, base_arms=[1, 2, 3, 4, 6, 8, 10, 12], ext_per_1000=0
    ) == [1, 2, 3, 4, 6, 8, 10, 12]


def test_c6_ext6_pins() -> None:
    kwargs = {"ext_per_1000": 6}
    assert k_domain(5061, 184, **kwargs) == [2, 3, 4, 6, 8, 10, 12, 16, 22, 30]
    assert k_domain(7037, 38, **kwargs) == [2, 3, 4, 6, 8, 10, 12, 18, 26, 38]
    assert k_domain(8028, 294, **kwargs) == [2, 3, 4, 6, 8, 10, 12, 19, 30, 48]
    assert k_domain(10001, 1570, **kwargs) == [2, 3, 4, 6, 8, 10, 12, 21, 35, 60]


def test_c6_thin_confirmation_pins() -> None:
    kwargs = {"base_arms": THIN_BASE, "ext_per_1000": 6}
    assert k_domain(7037, 38, **kwargs) == [2, 4, 8, 12, 15, 18, 21, 26, 31, 38]
    assert k_domain(8028, 294, **kwargs) == [2, 4, 8, 12, 15, 19, 24, 30, 38, 48]
    assert k_domain(10001, 1570, **kwargs) == [2, 4, 8, 12, 16, 21, 27, 35, 46, 60]


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="n must"):
        k_domain(0, 5)
    with pytest.raises(ValueError, match="k_min"):
        k_domain(100, 0)
    with pytest.raises(ValueError, match="max_arms"):
        k_domain(100, 5, max_arms=0)
