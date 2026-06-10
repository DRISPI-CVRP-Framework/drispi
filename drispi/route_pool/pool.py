"""Central route memory structure for DRISPI."""

from __future__ import annotations

from collections.abc import Iterator

from drispi.core.types import Route
from drispi.haos.tag import HAOSTag
from drispi.route_pool._entry import RouteEntry
from drispi.route_pool.diversity import mean_jaccard_diversity


class RoutePool:
    """
    Central route memory structure for DRISPI.

    Routes are keyed by ``frozenset(route)`` so two routes with equal customer
    sets are treated as duplicates regardless of order.
    """

    def __init__(self) -> None:
        self._entries: dict[frozenset[int], RouteEntry] = {}
        self._elite_keys: set[frozenset[int]] = set()
        self._iter_rejected = 0
        self._iter_replaced = 0

    def add(
        self,
        route: Route,
        cost: float,
        haos_tag: HAOSTag | None = None,
    ) -> bool:
        """Add one route if it is new; update existing when cheaper."""
        key = frozenset(route)
        existing = self._entries.get(key)
        if existing is not None:
            if cost < existing.cost:
                replacement = RouteEntry(
                    route=list(route),
                    cost=cost,
                    customer_set=key,
                    is_elite=existing.is_elite,
                    haos_tag=haos_tag,
                    quality_scores=existing.quality_scores,
                    diversity_scores=existing.diversity_scores,
                )
                self._entries[key] = replacement
                self._iter_replaced += 1
            else:
                self._iter_rejected += 1
            return False

        self._entries[key] = RouteEntry(
            route=list(route),
            cost=cost,
            customer_set=key,
            haos_tag=haos_tag,
        )
        return True

    def remove(self, route: Route) -> bool:
        """Remove a route by customer-set key."""
        key = frozenset(route)
        if key not in self._entries:
            return False
        del self._entries[key]
        self._elite_keys.discard(key)
        return True

    def set_elite(self, routes: list[Route]) -> None:
        """Mark exactly these routes as elite."""
        new_elite_keys = {frozenset(route) for route in routes}
        for key in new_elite_keys:
            if key not in self._entries:
                raise KeyError(key)

        for key, entry in self._entries.items():
            entry.is_elite = key in new_elite_keys

        self._elite_keys = new_elite_keys

    def update_quality_scores(self, lp_weights: dict[frozenset[int], float]) -> None:
        """Append LP weight quality values for routes that are present."""
        for key, weight in lp_weights.items():
            entry = self._entries.get(key)
            if entry is not None:
                entry.quality_scores.append(weight)

    def update_diversity_scores(self) -> None:
        """Recompute and append diversity score for each route."""
        all_entries = list(self._entries.values())
        for entry in all_entries:
            entry.diversity_scores.append(mean_jaccard_diversity(entry, all_entries))

    def routes(self, elite_only: bool = False) -> list[RouteEntry]:
        """Return all route entries or only elite entries."""
        entries = list(self._entries.values())
        if not elite_only:
            return entries
        return [entry for entry in entries if entry.is_elite]

    def as_route_pool(self) -> list[Route]:
        """Export routes as plain list alias expected by SP/SC components."""
        return [list(entry.route) for entry in self._entries.values()]

    def size(self) -> int:
        """Number of routes currently stored."""
        return len(self._entries)

    def reset_iter_counters(self) -> None:
        """Reset per-iteration duplicate-rejection/replacement counters."""
        self._iter_rejected = 0
        self._iter_replaced = 0

    def get_iter_counters(self) -> tuple[int, int]:
        """Return ``(duplicates_rejected, duplicates_replaced)`` since last reset."""
        return self._iter_rejected, self._iter_replaced

    def get_haos_tag(self, route: Route) -> HAOSTag | None:
        """Return the HAOS tag for this customer set if the route is in the pool."""
        entry = self._entries.get(frozenset(route))
        return None if entry is None else entry.haos_tag

    def reset_to(self, routes: list[Route], costs: list[float]) -> None:
        """Reset pool to provided routes/costs, then mark all as elite."""
        if len(routes) != len(costs):
            raise ValueError("routes and costs must have equal length")
        self._entries.clear()
        self._elite_keys.clear()
        for route, cost in zip(routes, costs, strict=True):
            self.add(route, cost)
        self.set_elite(routes)

    def __iter__(self) -> Iterator[RouteEntry]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, route: Route) -> bool:
        return frozenset(route) in self._entries
