"""Gurobi set covering / set partitioning models (LP relaxation then MIP)."""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from drispi.core.instance import CVRPInstance
from drispi.core.types import Route
from drispi.route_pool.pool import RoutePool


def compute_route_cost(route: Route, instance: CVRPInstance) -> float:
    """
    Euclidean cost of a route: depot → customers → depot.

    Uses ``instance.distance_matrix`` via ``CVRPInstance.route_cost``.
    """
    return instance.route_cost(route)


def build_and_solve(
    pool: RoutePool,
    instance: CVRPInstance,
    use_sp: bool,
    time_limit: float = 300.0,
    mip_gap: float = 0.01,
) -> tuple[dict[frozenset[int], float], list[Route], bool, int]:
    """
    Build and solve the SC or SP model using two sequential Gurobi calls on one model.

    Returns:
        lp_weights: LP relaxation primal x_r in [0, 1] for each route column.
        raw_solution: routes selected in the MIP (x_r > 0.5).
        timed_out: True if the MIP stopped on the time limit before proving optimality.
        sol_count: ``model.SolCount`` after the MIP (feasible integer solutions found).
    """
    ordered: list[tuple[frozenset[int], Route]] = []
    for route in pool.as_route_pool():
        key = frozenset(route)
        ordered.append((key, list(route)))

    model = gp.Model("drispi_sp_sc")
    model.setParam("OutputFlag", 0)
    model.ModelSense = GRB.MINIMIZE

    vars_by_key: dict[frozenset[int], gp.Var] = {}
    for key, route in ordered:
        # Cost recomputed from distance matrix rather than RouteEntry.cost to ensure
        # objective is consistent with the current matrix definition.
        coeff = compute_route_cost(route, instance)
        vars_by_key[key] = model.addVar(
            lb=0.0,
            ub=1.0,
            obj=coeff,
            vtype=GRB.CONTINUOUS,
            name=f"x_{abs(hash(key)) % (10**9)}",
        )

    for customer in instance.customers:
        expr = gp.quicksum(v for key, v in vars_by_key.items() if customer in key)
        if use_sp:
            model.addConstr(expr == 1, name=f"cover_eq_{customer}")
        else:
            model.addConstr(expr >= 1, name=f"cover_ge_{customer}")

    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        msg = "LP relaxation is infeasible for the given pool and instance."
        raise RuntimeError(msg)

    lp_weights: dict[frozenset[int], float] = {}
    for key, var in vars_by_key.items():
        x = float(var.X)
        lp_weights[key] = min(1.0, max(0.0, x))

    for key, var in vars_by_key.items():
        var.setAttr(GRB.Attr.VType, GRB.BINARY)
        var.setAttr(GRB.Attr.Start, lp_weights[key])

    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.optimize()

    timed_out = model.Status == GRB.TIME_LIMIT
    sol_count = int(model.SolCount)

    if model.Status == GRB.INFEASIBLE:
        msg = "MIP is infeasible for the given pool and instance."
        raise RuntimeError(msg)

    raw_solution: list[Route] = []
    for key, route in ordered:
        if vars_by_key[key].X > 0.5:
            raw_solution.append(list(route))

    return lp_weights, raw_solution, timed_out, sol_count
