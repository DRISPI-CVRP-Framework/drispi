"""Smoke: instrumented AILS-II writes a touch dump that maps back to global IDs."""

from __future__ import annotations

from pathlib import Path

import pytest

from drispi.ablation import config as ac
from drispi.ablation.touch import customer_touch_maps, parse_touch_tsv
from drispi.core.instance import CVRPInstance
from drispi.core.solution import Route
from drispi.solvers.ails2 import Ails2Solver
from drispi.utils.io import write_sol


@pytest.mark.skipif(not ac.TOUCH_JAR.is_file(), reason="AILSII-touch.jar missing")
def test_instrumented_jar_dumps_touch_tsv(small_instance: CVRPInstance, tmp_path: Path) -> None:
    routes = [
        Route(customers=[2, 3, 4], cost=small_instance.route_cost([2, 3, 4])),
        Route(customers=[5, 6], cost=small_instance.route_cost([5, 6])),
    ]
    init = tmp_path / "init.sol"
    write_sol([r.customers for r in routes], sum(r.cost for r in routes), init)
    dump = tmp_path / "touch.tsv"
    solver = Ails2Solver(binary_path=ac.TOUCH_JAR, active_processor_count=1, xmx="512m")
    out = solver.run_improvement(
        small_instance,
        2.0,
        init,
        initial_omega=10.0,
        touch_dump_path=dump,
    )
    assert dump.is_file()
    parsed = parse_touch_tsv(dump)
    assert "iterator" in parsed["header"]
    maps = customer_touch_maps(parsed, small_instance)
    assert set(maps["eval"]) == set(small_instance.customers)
    assert sum(maps["eval"].values()) >= 0
    assert out
    import zipfile

    with zipfile.ZipFile(ac.STOCK_JAR) as zf:
        assert not any(n.endswith("Touch.class") for n in zf.namelist())
    with zipfile.ZipFile(ac.TOUCH_JAR) as zf:
        assert any(n.endswith("Touch.class") for n in zf.namelist())
