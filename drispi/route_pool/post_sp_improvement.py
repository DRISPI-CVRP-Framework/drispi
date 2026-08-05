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
    iteration: int,
    selection: HAOSSelection,
) -> int:
    """
    After ``run_standard_improvement``, add all AILS output routes to the pool.

    Routes whose customer set matches an SP/SC result route reuse the tag from
    the pool (unchanged by AILS at the set level). New customer sets get an
    improvement tag (``is_improvement_route=True``) with the same HAOS field
    values as ``selection`` for traceability.

    Returns the number of routes passed to ``pool.add``.
    """
    sp_sets = {frozenset(r) for r in sp_sc_routes}
    n_added = 0
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
        n_added += 1
    return n_added
