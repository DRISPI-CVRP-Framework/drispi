# DRISPI

## Overview

DRISPI is a Python research framework for vehicle routing that combines hierarchical adaptive operator selection (HAOS), clustering-based decomposition, heterogeneous solvers, route pooling, and set partitioning / covering models. This repository is currently a **stub skeleton**: packages import cleanly, but algorithms and I/O are not implemented yet.

## Repo structure

The installable package lives under `drispi/` (core domain types, clustering, solvers, set partitioning, route pool, improvement, HAOS, pipeline, utils). Configuration YAML files are in `configs/`, CLI entry scripts in `scripts/`, tests in `tests/`, instance and benchmark data under `data/`, and external solver sources under `vendor/` (Git submodules). Docker assets are in `docker/` with a root `docker-compose.yml` for containerized runs.

## Setup

Clone the repository, then initialize submodules so `vendor/filo`, `vendor/filo2`, and `vendor/ails2` are populated (update `.gitmodules` URLs to your forks or upstreams). Create a Python 3.11+ environment and install the package with `pip install -e ".[dev]"`. For reproducible runs with Gurobi and compiled solvers, build the `docker/Dockerfile` image (two-stage: Ubuntu builds binaries, slim Python image installs DRISPI) and place `gurobi.lic` where Compose can mount it (see below).

## Running a benchmark

From the repo root after installation, `python scripts/run_benchmark.py` is the intended batch driver (stub today). With Docker Compose, `docker compose up --build` runs the same default command inside the image, with `./data` mounted at `/data` and the license file at `/opt/gurobi/gurobi.lic`.

## Config reference

`configs/default.yaml` holds global defaults: cluster cardinality cap (`k_max`), coverage floor (`min_coverage`), SP/SC time limits, segment count (`n_segments`), HAOS `decay`, worker counts, selector settings (`selectors`), enabled `solvers`, and lists of `vertex_methods` / `route_methods`. `configs/xl_instances.yaml` overrides a subset for large instances (`k_max`, `sp_time_limit`, `n_segments`). Experiment-specific YAML can be added under `configs/experiments/`.
