"""Tests for dashboard visualization builders."""

from __future__ import annotations

import plotly.graph_objects as go

from drispi.dashboard import viz


def _minimal_snapshot(phase_num: int) -> dict:
    return {
        "phase_num": phase_num,
        "phase_name": "test",
        "depot": [500.0, 500.0],
        "customer_ids": [2, 3, 4],
        "customers": [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]],
        "cluster_assignments": {"2": 0, "3": 0, "4": 1},
        "routes": [[2, 3], [4]],
        "route_cluster_ids": [0, 1],
        "perturbed_route_indices": [0],
        "changed_route_indices": [1],
        "selected_route_indices": [0, 1],
    }


def test_build_iteration_figure_phases() -> None:
    for phase in range(1, 6):
        fig = viz.build_iteration_figure(_minimal_snapshot(phase))
        assert isinstance(fig, go.Figure)


def test_build_best_solution_figure() -> None:
    best = {
        "depot": [0.0, 0.0],
        "customer_ids": [2, 3],
        "customers": [[1.0, 2.0], [3.0, 4.0]],
        "routes": [[2, 3]],
    }
    fig = viz.build_best_solution_figure(best)
    assert isinstance(fig, go.Figure)


def test_build_trajectory_figure_empty() -> None:
    fig = viz.build_trajectory_figure([], None)
    assert isinstance(fig, go.Figure)


def test_build_iteration_figure_none_fields() -> None:
    snap: dict = {
        "phase_num": 1,
        "depot": [0.0, 0.0],
        "customer_ids": [],
        "customers": [],
        "cluster_assignments": None,
        "routes": None,
    }
    fig = viz.build_iteration_figure(snap)
    assert isinstance(fig, go.Figure)


def test_build_haos_figure_none() -> None:
    html = viz.build_haos_figure(None)
    assert "Waiting" in html


def test_build_bg_figures() -> None:
    pre = {
        "depot": [500.0, 500.0],
        "customer_ids": [2, 3, 4],
        "customers": [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]],
        "routes": [[2, 3], [4]],
        "route_cluster_ids": [0, 1],
        "perturbed_route_indices": [0],
        "bg_stage": "pre",
    }
    post = {
        "depot": [500.0, 500.0],
        "customer_ids": [2, 3, 4],
        "customers": [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]],
        "routes": [[2, 3], [4]],
        "changed_route_indices": [1],
    }
    assert isinstance(viz.build_bg_perturb_figure(pre), go.Figure)
    assert isinstance(viz.build_bg_result_figure(post), go.Figure)
