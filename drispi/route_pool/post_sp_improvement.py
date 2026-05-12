"""Pool updates after post-SP/SC standard AILS improvement."""

from __future__ import annotations

from drispi.core.instance import CVRPInstance
from drispi.core.types import Route
from drispi.haos.haos import HAOSSelection
from drispi.route_pool.pool import RoutePool


def add_post_standard_improvement_routes_to_pool(
    pool: RoutePool,
    instance: CVRPInstance,
    sp_sc_routes: list[Route],
    ails_output_routes: list[Route],
    final_cost: float,
    best_cost: float,
    iteration: int,
    selection: HAOSSelection,
) -> bool:
    """
    After ``run_standard_improvement``, add AILS routes to the pool only if
    ``final_cost`` strictly improves the global best ``best_cost``.

    Routes whose customer set matches an SP/SC result route reuse the tag from
    the pool (unchanged by AILS at the set level). New customer sets get an
    improvement tag (``is_improvement_route=True``) with the same HAOS field
    values as ``selection`` for traceability.
    """
    if final_cost >= best_cost:
        return False

    sp_sets = {frozenset(r) for r in sp_sc_routes}
    for route in ails_output_routes:
        cost = instance.route_cost(route)
        key = frozenset(route)
        if key in sp_sets:
            tag = pool.get_haos_tag(route)
            if tag is None:
                tag = selection.to_tag(iteration, is_improvement_route=False)
            pool.add(route, cost, haos_tag=tag)
        else:
            pool.add(
                route,
                cost,
                haos_tag=selection.to_tag(iteration, is_improvement_route=True),
            )
    return True
