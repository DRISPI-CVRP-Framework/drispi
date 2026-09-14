"""Boundary-guided AILS-II: Python-side ranks + perturbation, then Java AILS-II."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route, Solution
from drispi.improvement.base import BaseImprovement
from drispi.solvers.ails2 import Ails2Solver
from drispi.utils.io import write_sol

PairSelection = Literal["stochastic", "greedy"]
NChainsMode = Literal["k_minus_1", "k"]


def _validate_partition(partition: list[list[int]], customers: list[int]) -> None:
    flat = [c for g in partition for c in g]
    if len(flat) != len(set(flat)):
        raise ValueError("partition lists must not contain duplicate customers")
    if sorted(flat) != sorted(customers):
        raise ValueError("partition must cover exactly instance.customers")


def _customer_index_map(customers: list[int]) -> dict[int, int]:
    return {cid: i for i, cid in enumerate(customers)}


def compute_boundary_ranks(
    dissimilarity_matrix: np.ndarray,
    partition: list[list[int]],
    customers: list[int],
    *,
    small_cluster_cap: int,
    small_cluster_alpha: float,
) -> np.ndarray:
    """
    Per-customer boundary ranks in ``[0, 1]``, aligned with ``customers`` / ``D``.

    Low minimum cross-cluster dissimilarity ⇒ higher rank (more boundary-like).
    Applies inverse cluster-size weighting and a boost for clusters smaller than
    ``small_cluster_cap``.
    """
    _validate_partition(partition, customers)
    n = len(customers)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if dissimilarity_matrix.shape != (n, n):
        raise ValueError(
            f"dissimilarity_matrix expected ({n},{n}), got {dissimilarity_matrix.shape}"
        )
    cid_to_i = _customer_index_map(customers)
    cluster_id = np.zeros(n, dtype=np.int32)
    for idx, group in enumerate(partition):
        for c in group:
            cluster_id[cid_to_i[c]] = idx

    diff_cluster = cluster_id[:, None] != cluster_id[None, :]
    masked = np.where(diff_cluster, dissimilarity_matrix, np.inf)
    min_cross = np.min(masked, axis=1)
    finite = np.isfinite(min_cross)
    if not np.any(finite):
        ranks = np.ones(n, dtype=np.float64)
    else:
        mc = np.where(finite, min_cross, 0.0)
        lo = float(np.min(mc[finite]))
        hi = float(np.max(mc[finite]))
        span = hi - lo + 1e-12
        ranks = 1.0 - (mc - lo) / span
        ranks = np.where(finite, ranks, 1.0)

    sizes = np.array([len(g) for g in partition], dtype=np.float64)
    sizes_safe = np.maximum(sizes, 1.0)
    inv = 1.0 / sizes_safe[cluster_id]
    weighted = ranks * inv

    cs = sizes[cluster_id]
    boost = np.where(
        cs < float(small_cluster_cap),
        1.0
        + float(small_cluster_alpha)
        * (float(small_cluster_cap) - cs)
        / float(small_cluster_cap),
        1.0,
    )
    weighted = weighted * boost
    lo2, hi2 = float(np.min(weighted)), float(np.max(weighted))
    if hi2 <= lo2 + 1e-15:
        return np.ones(n, dtype=np.float64)
    out = (weighted - lo2) / (hi2 - lo2 + 1e-15)
    return np.clip(out, 0.0, 1.0)


def _apply_threshold(ranks: np.ndarray, boundary_threshold: float) -> np.ndarray:
    """Soft gate: spread mass above ``boundary_threshold`` then renormalize to [0,1]."""
    if boundary_threshold <= 0.0:
        return ranks
    if boundary_threshold >= 1.0:
        return np.ones_like(ranks) / max(len(ranks), 1)
    shifted = np.maximum(0.0, ranks - float(boundary_threshold))
    denom = 1.0 - float(boundary_threshold) + 1e-9
    out = shifted / denom
    m, M = float(np.min(out)), float(np.max(out))
    if M <= m + 1e-15:
        return np.ones_like(ranks) / max(len(ranks), 1)
    return (out - m) / (M - m + 1e-15)


def _route_cluster_ids(seqs: list[list[int]], partition: list[list[int]]) -> list[int]:
    """Majority-vote cluster index per route; empty routes use ``-1``."""
    cid_to_cluster: dict[int, int] = {}
    for k, group in enumerate(partition):
        for c in group:
            cid_to_cluster[c] = k
    out: list[int] = []
    for s in seqs:
        if not s:
            out.append(-1)
            continue
        counts: dict[int, int] = {}
        for c in s:
            k = cid_to_cluster[c]
            counts[k] = counts.get(k, 0) + 1
        out.append(max(counts, key=counts.get))
    return out


def compute_route_boundary_affinities(
    seqs: list[list[int]],
    route_cluster_ids: list[int],
    partition: list[list[int]],
    dissimilarity_matrix: np.ndarray,
    cid_to_i: dict[int, int],
) -> np.ndarray:
    """
    Returns matrix of shape ``(n_routes, n_clusters)``.

    Entry ``[r, k]`` measures how close route ``r``'s customers are to cluster ``k``,
    as affinity (higher = closer). Own-cluster column is ``0``; rows sum to ``1``.
    Empty routes get a uniform distribution over clusters.
    """
    n_r = len(seqs)
    n_c = len(partition)
    aff = np.zeros((n_r, n_c), dtype=np.float64)
    cluster_customer_indices = [
        np.array([cid_to_i[c] for c in group], dtype=np.int64) for group in partition
    ]

    for r in range(n_r):
        s_r = seqs[r]
        if not s_r:
            if n_c > 0:
                aff[r, :] = 1.0 / float(n_c)
            continue
        idx_r = np.array([cid_to_i[c] for c in s_r], dtype=np.int64)
        c_r = route_cluster_ids[r]
        for k in range(n_c):
            if k == c_r:
                aff[r, k] = 0.0
                continue
            idx_k = cluster_customer_indices[k]
            if idx_k.size == 0 or idx_r.size == 0:
                aff[r, k] = 0.0
                continue
            sub = dissimilarity_matrix[np.ix_(idx_r, idx_k)]
            mean_dissim = float(np.mean(sub))
            aff[r, k] = 1.0 / (mean_dissim + 1e-9)
        row_sum = float(np.sum(aff[r]))
        if row_sum > 1e-15:
            aff[r] /= row_sum
        elif n_c > 0:
            aff[r, :] = 1.0 / float(n_c)

    return aff


def _pick_from_weights(
    weights: np.ndarray,
    rng: np.random.Generator,
    *,
    selection: PairSelection,
) -> int:
    """Sample index ∝ ``weights``, or take ``argmax`` when ``selection == "greedy"``."""
    w = np.asarray(weights, dtype=np.float64)
    if selection == "greedy":
        return int(np.argmax(w))
    total = float(w.sum())
    if total < 1e-15:
        return int(rng.integers(0, len(w)))
    return int(rng.choice(len(w), p=w / total))


def _pick_route_pair_by_affinity(
    seqs: list[list[int]],
    route_cluster_ids: list[int],
    overall_weights: np.ndarray,
    affinity_matrix: np.ndarray,
    rng: np.random.Generator,
    *,
    selection: PairSelection = "stochastic",
) -> tuple[int, int]:
    """
    Select routes ``(i, j)`` near the same boundary: pick ``i`` by boundary weight,
    pick neighboring cluster ``k`` from ``i``'s affinity row, then pick ``j`` in
    ``k`` weighted by how close ``j`` is to ``i``'s cluster.

    ``selection="stochastic"`` samples at each step; ``"greedy"`` takes argmax.
    """
    n_r = len(seqs)
    if n_r < 2:
        return 0, 0

    w = overall_weights.astype(np.float64) + 1e-9
    i = _pick_from_weights(w, rng, selection=selection)
    c_i = route_cluster_ids[i]

    row = affinity_matrix[i].astype(np.float64)
    row_sum = float(np.sum(row))
    if row_sum < 1e-15:
        n_c = row.shape[0]
        if n_c <= 1:
            j = (i + 1) % n_r
            return i, j
        if selection == "greedy":
            # Prefer first cluster that is not i's own (deterministic fallback).
            k = 0 if c_i != 0 else (1 if n_c > 1 else 0)
        else:
            k = int(rng.integers(0, n_c))
            if k == c_i:
                k = (k + 1) % n_c
    else:
        k = _pick_from_weights(row, rng, selection=selection)

    candidates = [j for j in range(n_r) if j != i and route_cluster_ids[j] == k]
    if not candidates:
        w2 = w.copy()
        w2[i] = 0.0
        if w2.sum() < 1e-15:
            j = (i + 1) % n_r
            return i, j
        j = _pick_from_weights(w2, rng, selection=selection)
        return i, j

    if c_i < 0:
        scores = np.ones(len(candidates), dtype=np.float64)
    else:
        scores = np.array(
            [affinity_matrix[j, c_i] + 1e-9 for j in candidates],
            dtype=np.float64,
        )
    pick = _pick_from_weights(scores, rng, selection=selection)
    j = candidates[pick]
    return i, j


def _cross_reconnect(
    seq_i: list[int],
    seq_j: list[int],
    rng: np.random.Generator,
) -> tuple[list[int], list[int], int, int]:
    """
    Cross-reconnect move between two routes: cut ``i`` at ``a`` and ``j`` at ``b``,
    then ``new_i = seq_i[:a] + seq_j[b:]``, ``new_j = seq_j[:b] + seq_i[a:]``.

    Returns ``(new_i, new_j, a, b)``. Short routes are returned unchanged with
    cut indices ``(0, 0)``.
    """
    n_i, n_j = len(seq_i), len(seq_j)
    if n_i < 2 or n_j < 2:
        return seq_i[:], seq_j[:], 0, 0
    a = int(rng.integers(1, n_i))
    b = int(rng.integers(1, n_j))
    new_i = seq_i[:a] + seq_j[b:]
    new_j = seq_j[:b] + seq_i[a:]
    return new_i, new_j, a, b


def _resolve_n_chains(
    partition: list[list[int]],
    *,
    n_chains: int | None,
    n_chains_mode: NChainsMode,
) -> int:
    """Resolve chain count from explicit override or ``n_chains_mode``."""
    if n_chains is not None:
        return n_chains
    k = len(partition)
    if n_chains_mode == "k":
        return k
    return max(1, k - 1)


def perturb_routes(
    routes: list[Route],
    instance: CVRPInstance,
    ranks_weighted: np.ndarray,
    partition: list[list[int]],
    dissimilarity_matrix: np.ndarray,
    rng: np.random.Generator,
    *,
    n_chains: int | None = None,
    pair_selection: PairSelection = "stochastic",
    n_chains_mode: NChainsMode = "k_minus_1",
    pair_picker: Literal["affinity", "uniform"] = "affinity",
) -> tuple[list[Route], list[int], list[dict[str, int]]]:
    """
    Multi-route cross-reconnect perturbation.

    Uses max boundary rank per route, route–cluster affinity for pair selection,
    then :func:`_cross_reconnect` on the chosen pair, repeated for the resolved chain
    count (see below).

    If ``n_chains`` is ``None``, ``n_chains_mode="k_minus_1"`` uses
    ``max(1, len(partition) - 1)`` and ``"k"`` uses ``len(partition)``.
    ``pair_selection`` controls stochastic vs greedy route-pair picking when
    ``pair_picker="affinity"``. ``pair_picker="uniform"`` draws both routes
    uniformly at random and ignores ranks, affinities, and ``pair_selection``.

    Returns ``(routes, perturbed_route_indices, chain_trace)`` where each trace
    entry is ``{"i", "j", "a", "b"}``.
    """
    customers = list(instance.customers)
    cid_to_i = _customer_index_map(customers)
    seqs = [r.customers[:] for r in routes]
    n_r = len(seqs)
    empty_trace: list[dict[str, int]] = []
    if n_r < 2:
        s0 = seqs[0] if seqs else []
        return [Route(customers=s0[:], cost=instance.route_cost(s0))], [], empty_trace

    n_chain_iter = _resolve_n_chains(
        partition, n_chains=n_chains, n_chains_mode=n_chains_mode
    )

    weights = np.array(
        [
            float(np.max([float(ranks_weighted[cid_to_i[c]]) for c in s]) if s else 0.0)
            for s in seqs
        ],
        dtype=np.float64,
    )
    route_cluster_ids = _route_cluster_ids(seqs, partition)
    affinity = None
    if pair_picker == "affinity":
        affinity = compute_route_boundary_affinities(
            seqs, route_cluster_ids, partition, dissimilarity_matrix, cid_to_i
        )

    perturbed_indices: set[int] = set()
    trace: list[dict[str, int]] = []
    for _ in range(n_chain_iter):
        if pair_picker == "uniform":
            i = int(rng.integers(0, n_r))
            j = int(rng.integers(0, n_r - 1))
            if j >= i:
                j += 1
        else:
            assert affinity is not None
            i, j = _pick_route_pair_by_affinity(
                seqs,
                route_cluster_ids,
                weights,
                affinity,
                rng,
                selection=pair_selection,
            )
        perturbed_indices.add(i)
        perturbed_indices.add(j)
        seqs[i], seqs[j], cut_a, cut_b = _cross_reconnect(seqs[i], seqs[j], rng)
        trace.append(
            {
                "i": i,
                "j": j,
                "a": cut_a,
                "b": cut_b,
                "seq_i": seqs[i][:],
                "seq_j": seqs[j][:],
            }
        )

    out = [Route(customers=s, cost=instance.route_cost(s)) for s in seqs]
    return out, sorted(perturbed_indices), trace


def _routes_to_sol_payload(routes: list[Route], instance: CVRPInstance) -> tuple[list[list[int]], float]:
    seqs = [r.customers[:] for r in routes]
    cost = float(sum(instance.route_cost(s) for s in seqs))
    return seqs, cost


def export_boundary_ranks_csv(path: Path, customers: list[int], ranks: np.ndarray) -> None:
    """Write ``node_id,boundary_rank`` (debug / analysis)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "boundary_rank"])
        for c, r in zip(customers, ranks.tolist(), strict=True):
            w.writerow([c, f"{float(r):.8f}"])


def run_standard_improvement(
    instance: CVRPInstance,
    solution: list[Route],
    time_limit: float,
    *,
    solver: Ails2Solver | None = None,
    seed: int,
) -> list[Route]:
    """Post-SP/SC style: AILS-II with ``-initialSolution`` only (no ``-initialOmega``)."""
    solv = solver or Ails2Solver()
    seqs, cost = _routes_to_sol_payload(solution, instance)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        init_sol = root / "init.sol"
        write_sol(seqs, cost, init_sol)
        return solv.run_improvement(
            instance,
            float(time_limit),
            init_sol,
            initial_omega=None,
            seed=seed,
        )


def run_bg_ails_perturb(
    instance: CVRPInstance,
    solution: list[Route],
    dissimilarity_matrix: np.ndarray,
    partition: list[list[int]],
    *,
    boundary_threshold: float,
    small_cluster_cap: int,
    small_cluster_alpha: float,
    seed: int,
    pair_selection: PairSelection = "stochastic",
    n_chains_mode: NChainsMode = "k_minus_1",
) -> tuple[list[Route], list[int]]:
    """
    Boundary ranks + cross-reconnect perturbation only (no AILS-II).

    Returns ``(perturbed_routes, perturbed_route_indices)`` where indices refer to
    the input ``solution`` route list (routes chosen for perturbation).
    """
    if len(partition) <= 1:
        return solution, []
    rng = np.random.default_rng(seed)
    customers = list(instance.customers)
    ranks = compute_boundary_ranks(
        dissimilarity_matrix,
        partition,
        customers,
        small_cluster_cap=small_cluster_cap,
        small_cluster_alpha=small_cluster_alpha,
    )
    rw = _apply_threshold(ranks, boundary_threshold)
    pert, idx, _trace = perturb_routes(
        solution,
        instance,
        rw,
        partition,
        dissimilarity_matrix,
        rng,
        pair_selection=pair_selection,
        n_chains_mode=n_chains_mode,
    )
    return pert, idx


def run_blind_perturb(
    instance: CVRPInstance,
    solution: list[Route],
    partition: list[list[int]],
    *,
    seed: int,
    n_chains_mode: NChainsMode = "k",
) -> tuple[list[Route], list[int], list[dict[str, int]]]:
    """Arm-B perturbation: same chain length as BG-AILS, uniform route pairs."""
    if len(partition) <= 1:
        return solution, [], []
    rng = np.random.default_rng(seed)
    dummy_ranks = np.ones(len(instance.customers), dtype=np.float64)
    dummy_d = np.zeros((len(instance.customers), len(instance.customers)), dtype=np.float64)
    return perturb_routes(
        solution,
        instance,
        dummy_ranks,
        partition,
        dummy_d,
        rng,
        pair_selection="stochastic",
        n_chains_mode=n_chains_mode,
        pair_picker="uniform",
    )


def run_guided_perturb_traced(
    instance: CVRPInstance,
    solution: list[Route],
    dissimilarity_matrix: np.ndarray,
    partition: list[list[int]],
    *,
    boundary_threshold: float,
    small_cluster_cap: int,
    small_cluster_alpha: float,
    seed: int,
    pair_selection: PairSelection = "greedy",
    n_chains_mode: NChainsMode = "k",
) -> tuple[list[Route], list[int], list[dict[str, int]], np.ndarray]:
    """Arm-C perturbation plus pre-threshold ranks ``b̂_i`` for figures."""
    if len(partition) <= 1:
        n = len(instance.customers)
        return solution, [], [], np.ones(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    customers = list(instance.customers)
    ranks = compute_boundary_ranks(
        dissimilarity_matrix,
        partition,
        customers,
        small_cluster_cap=small_cluster_cap,
        small_cluster_alpha=small_cluster_alpha,
    )
    rw = _apply_threshold(ranks, boundary_threshold)
    pert, idx, trace = perturb_routes(
        solution,
        instance,
        rw,
        partition,
        dissimilarity_matrix,
        rng,
        pair_selection=pair_selection,
        n_chains_mode=n_chains_mode,
        pair_picker="affinity",
    )
    return pert, idx, trace, ranks


def run_bg_ails_improve(
    instance: CVRPInstance,
    perturbed: list[Route],
    partition: list[list[int]],
    initial_omega: float,
    *,
    time_limit: float,
    solver: Ails2Solver | None = None,
    seed: int,
) -> list[Route]:
    """Run AILS-II on a (possibly perturbed) solution with ``-initialOmega`` when multi-cluster."""
    if len(partition) <= 1:
        return run_standard_improvement(
            instance,
            perturbed,
            time_limit=float(time_limit),
            solver=solver,
            seed=seed,
        )
    solv = solver or Ails2Solver()
    seqs, cost = _routes_to_sol_payload(perturbed, instance)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        init_sol = root / "init.sol"
        write_sol(seqs, cost, init_sol)
        return solv.run_improvement(
            instance,
            float(time_limit),
            init_sol,
            initial_omega=float(initial_omega),
            seed=seed,
        )


def run_bg_ails(
    instance: CVRPInstance,
    solution: list[Route],
    dissimilarity_matrix: np.ndarray,
    partition: list[list[int]],
    initial_omega: float,
    *,
    time_limit: float,
    boundary_threshold: float,
    small_cluster_cap: int,
    small_cluster_alpha: float,
    solver: Ails2Solver | None = None,
    seed: int,
    pair_selection: PairSelection = "stochastic",
    n_chains_mode: NChainsMode = "k_minus_1",
) -> tuple[list[Route], list[int], list[Route]]:
    """
    Full BG-AILS pipeline: ranks → weighted perturbation → AILS-II with injected
    solution and ``-initialOmega``.

    Returns ``(perturbed_routes, perturbed_route_indices, improved_routes)`` where
    ``perturbed_route_indices`` are indices into the input solution's route list
    for routes selected in cross-reconnect perturbation chains.

    If ``partition`` has at most one cluster, delegates to
    :func:`run_standard_improvement` (same ``time_limit``).

    Cross-reconnect chains per run default to ``k - 1`` (``n_chains_mode``), or
    ``k`` when ``n_chains_mode="k"``. Pair selection is stochastic or greedy.
    """
    pert, perturbed_indices = run_bg_ails_perturb(
        instance,
        solution,
        dissimilarity_matrix,
        partition,
        boundary_threshold=boundary_threshold,
        small_cluster_cap=small_cluster_cap,
        small_cluster_alpha=small_cluster_alpha,
        seed=seed,
        pair_selection=pair_selection,
        n_chains_mode=n_chains_mode,
    )
    improved = run_bg_ails_improve(
        instance,
        pert,
        partition,
        initial_omega,
        time_limit=time_limit,
        solver=solver,
        seed=seed,
    )
    return pert, perturbed_indices, improved


class StandardAilsImprovement(BaseImprovement):
    """Post-SP/SC style improvement via :func:`run_standard_improvement`."""

    def __init__(self, *, solver: Ails2Solver | None = None, seed: int) -> None:
        self._solver = solver
        self._seed = seed

    def improve(self, solution: Solution, instance: CVRPInstance, time_limit: float) -> Solution:
        out_routes = run_standard_improvement(
            instance,
            solution.routes,
            float(time_limit),
            solver=self._solver,
            seed=self._seed,
        )
        total = float(sum(r.cost for r in out_routes))
        return Solution(
            routes=out_routes,
            total_cost=total,
            instance_name=solution.instance_name,
        )


class BgAilsImprovement(BaseImprovement):
    """
    HAOS-compatible wrapper: calls :func:`run_bg_ails` with clustering context
    supplied at construction time.
    """

    def __init__(
        self,
        dissimilarity_matrix: np.ndarray,
        partition: list[list[int]],
        initial_omega: float,
        *,
        boundary_threshold: float,
        small_cluster_cap: int,
        small_cluster_alpha: float,
        solver: Ails2Solver | None = None,
        seed: int,
        pair_selection: PairSelection = "stochastic",
        n_chains_mode: NChainsMode = "k_minus_1",
    ) -> None:
        self._d = np.asarray(dissimilarity_matrix, dtype=np.float64)
        self._partition = partition
        self._initial_omega = float(initial_omega)
        self._boundary_threshold = float(boundary_threshold)
        self._small_cluster_cap = small_cluster_cap
        self._small_cluster_alpha = small_cluster_alpha
        self._solver = solver
        self._seed = seed
        self._pair_selection = pair_selection
        self._n_chains_mode = n_chains_mode
        self.last_perturbed_route_indices: list[int] = []

    def improve(self, solution: Solution, instance: CVRPInstance, time_limit: float) -> Solution:
        _perturbed_routes, perturbed, out_routes = run_bg_ails(
            instance,
            solution.routes,
            self._d,
            self._partition,
            self._initial_omega,
            time_limit=float(time_limit),
            boundary_threshold=self._boundary_threshold,
            small_cluster_cap=self._small_cluster_cap,
            small_cluster_alpha=self._small_cluster_alpha,
            solver=self._solver,
            seed=self._seed,
            pair_selection=self._pair_selection,
            n_chains_mode=self._n_chains_mode,
        )
        self.last_perturbed_route_indices = perturbed
        total = float(sum(r.cost for r in out_routes))
        return Solution(
            routes=out_routes,
            total_cost=total,
            instance_name=solution.instance_name,
        )
