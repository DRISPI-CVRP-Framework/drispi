"""I/O utilities for VRP and solution formats."""

from __future__ import annotations

from pathlib import Path

from drispi.core.instance import CVRPInstance


def read_vrp(instance_name: str) -> CVRPInstance:
    """Thin wrapper around CVRPInstance.from_vrplib(instance_name)."""
    return CVRPInstance.from_vrplib(instance_name)


def write_vrp(
    instance: CVRPInstance,
    customers: list[int],
    path: Path,
) -> tuple[dict[int, int], dict[int, int]]:
    """Write a VRPLIB-format CVRP subproblem with local depot=1 indexing."""
    global_to_local: dict[int, int] = {1: 1}
    local_to_global: dict[int, int] = {1: 1}
    for idx, global_id in enumerate(customers, start=2):
        global_to_local[global_id] = idx
        local_to_global[idx] = global_id

    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"NAME : {instance.name}")
    lines.append('COMMENT : "Subproblem"')
    lines.append("TYPE : CVRP")
    lines.append(f"DIMENSION : {1 + len(customers)}")
    lines.append("EDGE_WEIGHT_TYPE : EUC_2D")
    lines.append(f"CAPACITY : {instance.capacity}")
    lines.append("NODE_COORD_SECTION")
    lines.append("")
    depot_x, depot_y = instance.depot
    lines.append(f"1 {round(depot_x)} {round(depot_y)}")
    for global_id in customers:
        local_id = global_to_local[global_id]
        x, y = instance.coordinates[global_id]
        lines.append(f"{local_id} {round(x)} {round(y)}")
    lines.append("")
    lines.append("DEMAND_SECTION")
    lines.append("")
    lines.append("1 0")
    for global_id in customers:
        local_id = global_to_local[global_id]
        lines.append(f"{local_id} {instance.demands[global_id]}")
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return global_to_local, local_to_global


def reindex_depot_zero_to_one(instance: CVRPInstance) -> CVRPInstance:
    """Shift depot/customer IDs from 0-based depot to internal 1-based depot."""
    customers = [customer + 1 for customer in instance.customers]
    coordinates = {node_id + 1: coord for node_id, coord in instance.coordinates.items()}
    demands = {node_id + 1: demand for node_id, demand in instance.demands.items()}
    return CVRPInstance(
        name=instance.name,
        n_customers=instance.n_customers,
        capacity=instance.capacity,
        depot=instance.depot,
        customers=customers,
        coordinates=coordinates,
        demands=demands,
    )


def reindex_depot_one_to_zero(instance: CVRPInstance) -> CVRPInstance:
    """Shift internal IDs from depot=1 to depot=0 conventions."""
    customers = [customer - 1 for customer in instance.customers]
    coordinates = {node_id - 1: coord for node_id, coord in instance.coordinates.items()}
    demands = {node_id - 1: demand for node_id, demand in instance.demands.items()}
    return CVRPInstance(
        name=instance.name,
        n_customers=instance.n_customers,
        capacity=instance.capacity,
        depot=instance.depot,
        customers=customers,
        coordinates=coordinates,
        demands=demands,
    )


def write_sol(routes: list[list[int]], cost: float, path: Path) -> None:
    """Write .sol routes/cost converting internal IDs (depot=1) to depot=0 indexing."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for idx, route in enumerate(routes, start=1):
        shifted = [str(customer - 1) for customer in route]
        lines.append(f"Route #{idx}: {' '.join(shifted)}")
    lines.append("")
    cost_value = float(cost)
    if cost_value.is_integer():
        cost_out = str(int(cost_value))
    else:
        cost_out = str(int(round(cost_value)))
    lines.append(f"Cost: {cost_out}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_sol(path: Path) -> tuple[list[list[int]], float]:
    """Read .sol routes/cost and convert customer IDs to internal indexing."""
    routes: list[list[int]] = []
    cost: float | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Route #"):
            if ":" not in line:
                raise ValueError("Invalid route line format")
            _, payload = line.split(":", maxsplit=1)
            payload = payload.strip()
            if payload:
                route = [int(token) + 1 for token in payload.split()]
            else:
                route = []
            routes.append(route)
            continue
        if line.startswith("Cost:"):
            _, payload = line.split(":", maxsplit=1)
            payload = payload.strip()
            if not payload:
                raise ValueError("Missing cost value")
            cost = float(payload)
            continue
        # Some solution files include trailing metadata (e.g., "Clustering: ...",
        # "Runtime: ...", "Gap: ..."). Ignore unknown non-route, non-cost lines.
        continue

    if cost is None:
        raise ValueError("Missing Cost line in .sol file")
    return routes, cost
