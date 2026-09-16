"""Ablation unit tests: IDs, blind perturbation, touch I/O, sanity, Holm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drispi.ablation.sanity import assert_touch_covers_modifications, modified_customers
from drispi.ablation.stats import holm, share_above_tau
from drispi.ablation.touch import parse_touch_tsv, to_global_ids
from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route
from drispi.improvement.bg_ails import perturb_routes, run_blind_perturb


def test_from_vrplib_customer_ids_are_2_through_n_plus_1() -> None:
    inst = CVRPInstance.from_vrplib("X-n101-k25")
    assert inst.customers == list(range(2, inst.n_customers + 2))
    assert to_global_ids(np.array([0, 1, 2]))[1] == 2


def test_blind_perturb_preserves_customers(small_instance: CVRPInstance) -> None:
    routes = [
        Route(customers=[2, 3], cost=small_instance.route_cost([2, 3])),
        Route(customers=[4, 5], cost=small_instance.route_cost([4, 5])),
        Route(customers=[6], cost=small_instance.route_cost([6])),
    ]
    partition = [[2, 3], [4, 5, 6]]
    out, idx, trace = run_blind_perturb(
        small_instance, routes, partition, seed=0, n_chains_mode="k"
    )
    assert len(trace) == 2
    flat = sorted(c for r in out for c in r.customers)
    assert flat == sorted(small_instance.customers)
    assert idx


def test_uniform_picker_ignores_ranks(small_instance: CVRPInstance) -> None:
    n = small_instance.n_customers
    routes = [
        Route(customers=[2, 3], cost=0.0),
        Route(customers=[4, 5], cost=0.0),
        Route(customers=[6], cost=0.0),
    ]
    # Route 0 would dominate affinity picking if ranks mattered.
    ranks = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    d = np.ones((n, n))
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(40):
        _out, _idx, trace = perturb_routes(
            routes,
            small_instance,
            ranks,
            [[2, 3], [4, 5], [6]],
            d,
            rng,
            n_chains=1,
            pair_picker="uniform",
        )
        seen.add((trace[0]["i"], trace[0]["j"]))
    assert len(seen) > 1


def test_parse_touch_tsv(tmp_path: Path) -> None:
    p = tmp_path / "x.touch.tsv"
    p.write_text(
        "# iterator 12\n# elapsed_s 3.2\nname\teval\taccept\tperturb\n0\t0\t0\t0\n1\t10\t2\t1\n",
        encoding="utf-8",
    )
    dump = parse_touch_tsv(p)
    assert dump["header"]["iterator"] == "12"
    assert dump["eval"][1] == 10
    assert to_global_ids(dump["name"])[1] == 2


def test_sanity_catches_missing_touch() -> None:
    before = [[2, 3, 4]]
    after = [[2, 4, 3]]
    changed = modified_customers(before, after)
    assert 3 in changed
    with pytest.raises(AssertionError):
        assert_touch_covers_modifications(changed, {2: 0, 3: 0, 4: 0}, {2: 0, 3: 0, 4: 0})
    assert_touch_covers_modifications(changed, {2: 1, 3: 1, 4: 1}, {})


def test_reversed_route_is_not_a_modification() -> None:
    assert modified_customers([[2, 3, 4]], [[4, 3, 2]]) == set()


def test_pct_vs_checkpoint_ignores_own_start() -> None:
    from drispi.ablation.stats import instance_means, pct_vs_checkpoint

    row = {
        "instance": "XL-n1-k1",
        "arms": {
            "A": {"cost_in": 100.0, "cost_out": 90.0, "pct_improvement": 10.0},
            "B": {"cost_in": 150.0, "cost_out": 90.0, "pct_improvement": 40.0},
            "C": {"cost_in": 110.0, "cost_out": 89.0, "pct_improvement": 19.09},
        },
    }
    assert pct_vs_checkpoint(row, "A") == pytest.approx(10.0)
    assert pct_vs_checkpoint(row, "B") == pytest.approx(10.0)
    assert pct_vs_checkpoint(row, "C") == pytest.approx(11.0)
    means = instance_means([row])
    assert means["XL-n1-k1"]["B"] == pytest.approx(10.0)
    assert means["XL-n1-k1"]["C"] == pytest.approx(11.0)


def test_write_boundary_mask_filters_tau(tmp_path: Path) -> None:
    from drispi.ablation.arms import write_boundary_mask

    path = write_boundary_mask(
        tmp_path,
        customers=[2, 3, 4, 5],
        ranks_hat=[0.1, 0.6, 0.5, 0.9],
        tau=0.5,
    )
    body = path.read_text(encoding="utf-8")
    active = [ln for ln in body.splitlines() if ln.isdigit()]
    assert active == ["3", "5"]


def test_holm_monotonic_and_family_of_two() -> None:
    adj = holm([0.01, 0.04])
    assert adj[0] <= adj[1]
    assert adj[0] == pytest.approx(0.02)
    assert adj[1] == pytest.approx(0.04)


def test_share_above_tau() -> None:
    ranks = np.array([0.1, 0.2, 0.8, 0.9])
    touches = np.array([1.0, 1.0, 8.0, 0.0])
    assert share_above_tau(ranks, touches, 0.5) == pytest.approx(8.0 / 10.0)


def test_config_hash_stable() -> None:
    from drispi.ablation.config import config_hash

    assert len(config_hash()) == 64


def test_list_xl_instances_is_one_hundred() -> None:
    from drispi.ablation.checkpoint import instance_vrp_path, list_xl_instances
    from drispi.core.instance import CVRPInstance

    names = list_xl_instances()
    assert len(names) == 100
    assert names[0].startswith("XL-n")
    inst = CVRPInstance.from_vrplib("XL-n1281-k29")
    assert inst.name == "X-n1281-k29"
    path = instance_vrp_path(inst, "XL-n1281-k29")
    assert path.name == "XL-n1281-k29.vrp"
