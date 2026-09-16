"""Cross-reconnect, boundary affinities, and perturb_routes invariants."""

from __future__ import annotations

import numpy as np

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route
from drispi.improvement import bg_ails


def _square_instance() -> CVRPInstance:
    return CVRPInstance(
        name="toy",
        n_customers=6,
        capacity=50,
        depot=(0.0, 0.0),
        customers=[2, 3, 4, 5, 6, 7],
        coordinates={
            1: (0.0, 0.0),
            2: (0.0, 1.0),
            3: (1.0, 1.0),
            4: (1.0, 0.0),
            5: (2.0, 0.0),
            6: (2.0, 1.0),
            7: (3.0, 0.5),
        },
        demands={1: 0, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5},
    )


def test_compute_route_boundary_affinities_shape_and_row_stochastic() -> None:
    customers = [2, 3, 4, 5]
    partition = [[2, 3], [4, 5]]
    cid = {c: i for i, c in enumerate(customers)}
    d = np.array(
        [
            [0.0, 1.0, 10.0, 10.0],
            [1.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0, 2.0],
            [10.0, 10.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    seqs = [[2, 3], [4, 5]]
    rc = bg_ails._route_cluster_ids(seqs, partition)
    aff = bg_ails.compute_route_boundary_affinities(seqs, rc, partition, d, cid)
    assert aff.shape == (2, 2)
    assert aff[0, 0] == 0.0 and aff[1, 1] == 0.0
    assert np.allclose(aff.sum(axis=1), 1.0)


def test_cross_reconnect_preserves_combined_multiset() -> None:
    rng = np.random.default_rng(0)
    a = [1, 2, 3, 4]
    b = [10, 20, 30, 40]
    na, nb, _a, _b = bg_ails._cross_reconnect(a, b, rng)
    assert sorted(na + nb) == sorted(a + b)


def test_cross_reconnect_short_routes_unchanged() -> None:
    rng = np.random.default_rng(0)
    na, nb, a, b = bg_ails._cross_reconnect([1], [2, 3], rng)
    assert na == [1] and nb == [2, 3]
    assert a == 0 and b == 0


def test_pick_route_pair_by_affinity_distinct_when_two_routes() -> None:
    seqs = [[2, 3], [4, 5]]
    partition = [[2, 3], [4, 5]]
    rc = bg_ails._route_cluster_ids(seqs, partition)
    cid = {c: i for i, c in enumerate([2, 3, 4, 5])}
    d = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    aff = bg_ails.compute_route_boundary_affinities(seqs, rc, partition, d, cid)
    w = np.array([1.0, 1.0], dtype=np.float64)
    rng = np.random.default_rng(12345)
    for _ in range(30):
        i, j = bg_ails._pick_route_pair_by_affinity(seqs, rc, w, aff, rng)
        assert i != j


def test_pick_route_pair_greedy_takes_argmax_weight() -> None:
    seqs = [[2, 3], [4, 5], [6, 7]]
    partition = [[2, 3], [4, 5], [6, 7]]
    rc = bg_ails._route_cluster_ids(seqs, partition)
    customers = [2, 3, 4, 5, 6, 7]
    cid = {c: i for i, c in enumerate(customers)}
    d = np.ones((6, 6), dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    # Make route 0 closest to cluster 1 so greedy partner is route 1.
    d[0, 2] = d[2, 0] = 0.1
    d[0, 3] = d[3, 0] = 0.1
    d[1, 2] = d[2, 1] = 0.1
    d[1, 3] = d[3, 1] = 0.1
    aff = bg_ails.compute_route_boundary_affinities(seqs, rc, partition, d, cid)
    w = np.array([0.1, 0.2, 10.0], dtype=np.float64)
    rng = np.random.default_rng(0)
    for _ in range(20):
        i, j = bg_ails._pick_route_pair_by_affinity(
            seqs, rc, w, aff, rng, selection="greedy"
        )
        assert i == 2
        assert i != j


def test_pick_route_pair_greedy_skips_used_first_routes() -> None:
    seqs = [[2, 3], [4, 5], [6, 7]]
    partition = [[2, 3], [4, 5], [6, 7]]
    rc = bg_ails._route_cluster_ids(seqs, partition)
    customers = [2, 3, 4, 5, 6, 7]
    cid = {c: i for i, c in enumerate(customers)}
    d = np.ones((6, 6), dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    aff = bg_ails.compute_route_boundary_affinities(seqs, rc, partition, d, cid)
    w = np.array([0.1, 0.2, 10.0], dtype=np.float64)
    rng = np.random.default_rng(0)
    i, j = bg_ails._pick_route_pair_by_affinity(
        seqs, rc, w, aff, rng, selection="greedy", used_first={2}
    )
    assert i == 1
    assert i != j


def test_greedy_chain_uses_k_distinct_first_routes() -> None:
    inst = _square_instance()
    partition = [[2, 3], [4, 5], [6, 7]]
    n = len(inst.customers)
    d = np.ones((n, n), dtype=np.float64) * 2.0
    np.fill_diagonal(d, 0.0)
    routes = [
        Route(customers=[2, 3], cost=0.0),
        Route(customers=[4, 5], cost=0.0),
        Route(customers=[6, 7], cost=0.0),
    ]
    ranks = np.zeros(n, dtype=np.float64)
    ranks[0] = 1.0  # customer 2 → route 0
    ranks[2] = 0.5  # customer 4 → route 1
    ranks[4] = 0.1  # customer 6 → route 2
    _out, _idx, trace = bg_ails.perturb_routes(
        routes,
        inst,
        ranks,
        partition,
        d,
        np.random.default_rng(0),
        n_chains=3,
        pair_selection="greedy",
        unique_first_routes=True,
    )
    firsts = [step["i"] for step in trace]
    assert firsts == [0, 1, 2]


def test_greedy_replay_repeats_the_same_first_route() -> None:
    inst = _square_instance()
    partition = [[2, 3], [4, 5], [6, 7]]
    n = len(inst.customers)
    d = np.ones((n, n), dtype=np.float64) * 2.0
    np.fill_diagonal(d, 0.0)
    routes = [
        Route(customers=[2, 3], cost=0.0),
        Route(customers=[4, 5], cost=0.0),
        Route(customers=[6, 7], cost=0.0),
    ]
    ranks = np.zeros(n, dtype=np.float64)
    ranks[0] = 1.0
    ranks[2] = 0.5
    ranks[4] = 0.1
    _out, _idx, trace = bg_ails.perturb_routes(
        routes,
        inst,
        ranks,
        partition,
        d,
        np.random.default_rng(0),
        n_chains=3,
        pair_selection="greedy",
        unique_first_routes=False,
    )
    assert [step["i"] for step in trace] == [0, 0, 0]


def test_pick_route_pair_stochastic_can_pick_non_max() -> None:
    seqs = [[2, 3], [4, 5]]
    partition = [[2, 3], [4, 5]]
    rc = bg_ails._route_cluster_ids(seqs, partition)
    cid = {c: i for i, c in enumerate([2, 3, 4, 5])}
    d = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    aff = bg_ails.compute_route_boundary_affinities(seqs, rc, partition, d, cid)
    w = np.array([10.0, 1.0], dtype=np.float64)
    rng = np.random.default_rng(7)
    seen_i = {bg_ails._pick_route_pair_by_affinity(seqs, rc, w, aff, rng)[0] for _ in range(200)}
    assert seen_i == {0, 1}


def test_n_chains_mode_k_minus_1_vs_k(monkeypatch) -> None:
    inst = _square_instance()
    partition = [[2, 3], [4, 5], [6, 7]]
    assert len(partition) == 3
    n = len(inst.customers)
    d = np.ones((n, n), dtype=np.float64) * 2.0
    np.fill_diagonal(d, 0.0)
    routes = [
        Route(customers=[2, 3], cost=0.0),
        Route(customers=[4, 5], cost=0.0),
        Route(customers=[6, 7], cost=0.0),
    ]
    ranks = np.ones(n, dtype=np.float64)
    calls: list[int] = []

    def _count_pick(*args, **kwargs):
        calls.append(1)
        return 0, 1

    monkeypatch.setattr(bg_ails, "_pick_route_pair_by_affinity", _count_pick)

    calls.clear()
    bg_ails.perturb_routes(
        routes,
        inst,
        ranks,
        partition,
        d,
        np.random.default_rng(0),
        n_chains_mode="k_minus_1",
    )
    assert len(calls) == 2  # k - 1

    calls.clear()
    bg_ails.perturb_routes(
        routes,
        inst,
        ranks,
        partition,
        d,
        np.random.default_rng(0),
        n_chains_mode="k",
    )
    assert len(calls) == 3  # k


def test_perturb_routes_preserves_all_customers() -> None:
    inst = _square_instance()
    partition = [[2, 3, 4], [5, 6, 7]]
    cid = {c: i for i, c in enumerate(inst.customers)}
    n = len(inst.customers)
    d = np.ones((n, n), dtype=np.float64) * 2.0
    np.fill_diagonal(d, 0.0)
    routes = [
        Route(customers=[2, 3, 4], cost=0.0),
        Route(customers=[5, 6, 7], cost=0.0),
    ]
    ranks = np.ones(n, dtype=np.float64)
    rng = np.random.default_rng(1)
    out, _perturbed, _trace = bg_ails.perturb_routes(
        routes,
        inst,
        ranks,
        partition,
        d,
        rng,
        n_chains=2,
    )
    flat = sorted(c for r in out for c in r.customers)
    assert flat == sorted(inst.customers)
