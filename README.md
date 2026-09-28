# DRISPI

Python solver for the capacitated vehicle routing problem. It decomposes an instance, solves the pieces with PyVRP, FILO, FILO2, and AILS-II, keeps a route pool, and improves the incumbent with set partitioning and boundary-guided AILS-II.

## Install

Python 3.11 or newer. From the repo root:

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`uv sync --dev` is the same install if you use uv.

Set covering and set partitioning call Gurobi (`gurobipy`). Put a license where Gurobi can see it, usually `GRB_LICENSE_FILE` or `~/gurobi.lic`. A run with `sp_sc.mode: "off"` still imports the package.

Build FILO, FILO2, and AILS-II before a run that uses them. Steps are in [docs/building.md](docs/building.md).

Instances used below are under `data/instances/`. Best-known costs are in `data/bks/xl-bks.json` and are loaded automatically.

## One instance

`configs/default.yaml` is the full flag list. Any other YAML in `configs/` is merged on top of it. CLI flags override the YAML.

Sync (DRI, BG-AILS, and SC/SP share the main process):

```bash
drispi-pipeline data/instances/xl/XL-n2307-k34.vrp \
  --config configs/sync_example.yaml \
  --time-limit 600 \
  --seed 100
```

Async production slice (6 decompose-route workers, 1 BG-AILS core, 1 SP core):

```bash
drispi-pipeline data/instances/xl/XL-n2307-k34.vrp \
  --config configs/async_example.yaml \
  --cpus 0-7 \
  --time-limit 7200 \
  --seed 100
```

`python -m drispi.pipeline.runner` is the same entry point. `--help` lists time limit, seed, output directory, `--gui`, and `--analysis`.

Output goes to `artifacts/runs/<instance>_<timestamp>/` unless the YAML sets `output_dir`. That directory is gitignored.

## Sync and async

Two switches, set independently in YAML. There is no CLI flag for them.

| Block | `sync` | `async` |
|-------|--------|---------|
| `bg_ails.mode` | Improvement runs on the main thread and blocks the next iteration. | A worker on `cores.bg` improves while the next decompose-route wave runs. Needs `cores.bg >= 1`. |
| `sp_sc.mode` | Gurobi and post-SP AILS run on the main thread after the trigger. | A worker on `cores.sp` owns the MIP. The main loop keeps going and applies a finished result at the start of a later iteration. |

`sp_sc.mode: "off"` skips set covering and set partitioning. Quote `off`. In YAML 1.1 an unquoted `off` becomes boolean false.

Sync SC/SP always fires on iterations: after `warmup_iterations`, every `sp_interval` iterations, once the pool meets `min_coverage`. Async can use that same iteration trigger, or `trigger: wallclock` with `interval_minutes`. `trigger: wallclock` is invalid with `mode: sync`.

Short overlays:

- [configs/sync_example.yaml](configs/sync_example.yaml) — sync, no CPU pin.
- [configs/async_example.yaml](configs/async_example.yaml) — 6+1+1, both workers async, wall-clock SP every 20 minutes.
- [configs/final_benchmark.yaml](configs/final_benchmark.yaml) — the same 6+1+1 layout with a 2 hour limit.

Scheduling, overlap, and log lines are in [docs/sp_sc_modes.md](docs/sp_sc_modes.md).

## Full benchmark

`scripts/run_final_benchmark.py` runs every `data/instances/xl/*.vrp` file for seeds 100, 200, and 300. Each job uses one 8-core slice and `configs/final_benchmark.yaml` (2 hours, async BG-AILS, async SP). On 32 cores that is 4 slices at a time. Finished `(instance, seed)` rows in `artifacts/final_benchmark/campaign.jsonl` are skipped on the next start.

```bash
python scripts/run_final_benchmark.py --dry-run
python scripts/run_final_benchmark.py --cpu-base 0 --total-cores 32
```

One slice of that campaign, same config and logging:

```bash
python scripts/run_final_benchmark.py --single \
  --instance data/instances/xl/XL-n2307-k34.vrp \
  --seed 100 --cpus 0-7
```

A smaller custom set, without the XL campaign layout:

```bash
python scripts/run_benchmark.py "data/instances/x/*.vrp" \
  --config configs/sync_example.yaml \
  --total-cores 8 \
  --cores-per-instance 8 \
  --time-limit 600
```

`--dry-run` prints the queue and exits.

## Where things live

`drispi/` is the installable package. A run enters through `drispi/pipeline/runner.py`, which loads YAML into `drispi/pipeline/config.py` and drives `drispi/pipeline/pipeline.py`.

| Package | Role |
|---------|------|
| `drispi/pipeline/` | Main loop, core pinning, sync/async workers for BG-AILS and SC/SP, run logs, post-run charts |
| `drispi/haos/` | Hierarchical adaptive operator selection: which k, clustering method, and solver to try next |
| `drispi/clustering/` | Vertex clustering (`kmeans`, agglomerative, k-medoids, fuzzy c-means) and route clustering |
| `drispi/solvers/` | Subproblem solvers: PyVRP in-process, FILO / FILO2 / AILS-II as subprocesses |
| `drispi/route_pool/` | Route pool, diversity, coverage, and the AILS pass that follows a successful SP |
| `drispi/sp/` | Set covering and set partitioning models solved with Gurobi |
| `drispi/improvement/` | Boundary-guided AILS-II and its time-budget predictor |
| `drispi/core/` | Instance, solution, and shared types |
| `drispi/dashboard/` | Streamlit view started with `--gui` |
| `drispi/ablation/` | BG-AILS ablation study (checkpoints, arm comparison, touch-counter jar). How to run it is in `drispi/ablation/README.md` |
| `drispi/utils/` | `.vrp` / `.sol` IO, BKS gaps, timestamps |

`configs/default.yaml` lists every flag. `sync_example.yaml`, `async_example.yaml`, and `final_benchmark.yaml` only override what they change.

`ext/` holds the COBRA, FILO, FILO2, and AILS-II sources as submodules. Build steps are in [docs/building.md](docs/building.md). Mode details are in [docs/sp_sc_modes.md](docs/sp_sc_modes.md).

`scripts/` holds the runners. `run_benchmark.py` and `run_final_benchmark.py` launch campaigns. `generate_bgails_checkpoints.py`, `run_bgails_ablation.py`, and `run_bgails_ablation_campaign.py` run the BG-AILS ablation. `build_final_benchmark_results.py` turns a campaign log into the summary CSV under `data/results/`.

`data/` has three parts:

- `data/instances/x/` and `data/instances/xl/` are the `.vrp` files.
- `data/bks/xl-bks.json` maps an instance name to a best-known cost.
- `data/results/` is the committed study output, not a place new runs write to. New runs go to `artifacts/`, which is gitignored. The two campaign tables are `finalBenchmarkResults_lagrange.csv` and `finalBenchmarkResults_dantzig.csv` (per instance, three seeds, cost, gap, and 30/60/90/120 minute checkpoints). The published benchmark is the Lagrange campaign. The Dantzig campaign is kept alongside it (it matched one BKS exactly and beat one initial BKS). Beside the tables are the comparison tables (`xl_solver_comparison_*.csv`), HAOS traces (`haos_campaign_lagrange.json`, `haos_wheel_state.csv`, credit CSVs), gap and paired-test JSON, and `bg_ails_ablation/*.json`. `dantzig_benchmark/` and `lagrange_benchmark/` keep the raw per-run logs those tables were built from.

`tests/` mirrors the package: `unit/`, `clustering/`, `haos/`, `route_pool/`, `sp/`, `improvement/`, `pipeline/`, `dashboard/`, and `ablation/`. `tests/integration/` runs a real pipeline on small and XL instances and is marked `integration` (some cases also `slow`). The default pytest invocation skips those markers:

```bash
pytest
pytest -m integration
```
