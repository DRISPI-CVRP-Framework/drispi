from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HAOSTag:
    """
    Records the full HAOS operator combination that generated a route.
    Attached to RouteEntry after improvement phase 1 (BG-AILS).
    Used for deferred reward attribution and pool interpretation.

    All fields use string/float/int values matching the choice labels
    defined in HAOSConfig - no internal indices stored here.

    When ``is_improvement_route`` is True, the route was produced by the
    post-SP/SC standard AILS step; k/lambda/paradigm/method/solver/iteration
    still mirror the iteration's HAOS selection for traceability. Such tags
    must not participate in deferred HAOS reward attribution.
    """

    k: int
    lambda_demand: float
    paradigm: str
    method: str
    solver: str
    iteration: int
    is_improvement_route: bool = False
