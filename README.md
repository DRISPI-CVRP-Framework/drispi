# DRISPI

## Overview

DRISPI is a Python research framework for vehicle routing that combines hierarchical adaptive operator selection (HAOS), clustering-based decomposition, heterogeneous solvers, route pooling, and set partitioning / covering models. This repository is currently a **stub skeleton**: packages import cleanly, but algorithms and I/O are not implemented yet.

## Repo structure

The installable package lives under `drispi/` (core domain types, clustering, solvers, set partitioning, route pool, improvement, HAOS, pipeline, utils). CLI entry scripts are in `scripts/`, tests in `tests/`, instance and benchmark data under `data/`, and external solver sources under `vendor/` (Git submodules). Docker assets are in `docker/` with a root `docker-compose.yml` for containerized runs.

## Setup

Clone the repository, then initialize submodules so `vendor/filo`, `vendor/filo2`, and `vendor/ails2` are populated (update `.gitmodules` URLs to your forks or upstreams). Create a Python 3.11+ environment and install the package with `pip install -e ".[dev]"`. For reproducible runs with Gurobi and compiled solvers, build the `docker/Dockerfile` image (two-stage: Ubuntu builds binaries, slim Python image installs DRISPI) and place `gurobi.lic` where Compose can mount it (see below).

## Running a benchmark

From the repo root after installation, run the pipeline via `drispi-pipeline <instance.vrp>` (or `python -m drispi.pipeline.runner`); see `--help` for time limits, SP/SC scheduling, `--gui`, and `--analysis` options. With Docker Compose, `docker compose up --build` runs the default command inside the image, with `./data` mounted at `/data` and the license file at `/opt/gurobi/gurobi.lic`.

## Configuration

All pipeline configuration lives in code: `drispi/pipeline/config.py` (`DRISPIConfig`) and `drispi/haos/config.py` (`HAOSConfig`), with CLI overrides exposed by `drispi/pipeline/runner.py` (see `drispi-pipeline --help`). YAML profiles live under `configs/` and merge on top of `configs/default.yaml`.

### SC/SP sync vs async

SC/SP (+ post-SP standard AILS) is controlled by `sp_sc.mode` (`off` | `sync` | `async`) and `sp_sc.trigger` (`iteration` | `wallclock`). Default is **sync** + **iteration**. Full guide, core pinning, and example commands: [docs/sp_sc_modes.md](docs/sp_sc_modes.md). Ready-to-run profiles: [`configs/sync_example.yaml`](configs/sync_example.yaml), [`configs/async_example.yaml`](configs/async_example.yaml).
