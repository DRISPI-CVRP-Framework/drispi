# DRISPI

Python research solver for capacitated VRP. Hierarchical adaptive operator
selection (HAOS) decomposes the instance, heterogeneous subcluster solvers
build routes, a route pool feeds set covering / partitioning (SC/SP), and
boundary-guided AILS-II (BG-AILS) improves the incumbent.

## Layout

| Path | What |
|------|------|
| `drispi/` | Installable package (pipeline, HAOS, clustering, solvers, SP, pool) |
| `scripts/` | Campaign runner and analysis CLIs |
| `configs/` | YAML profiles; merge on top of `configs/default.yaml` |
| `tests/` | Pytest suite |
| `data/` | Instances and BKS files (instances are not in the Docker image) |
| `ext/` | AILS-II, FILO, FILO2 sources |
| `docs/` | Mode and Docker notes |

## Setup

Python 3.11+. Create an environment and install with `pip install -e ".[dev]"`
(or `uv sync --dev`). Native solvers live under `ext/` and must be built for
`filo` / `filo2` / `ails2`.

For Gurobi + compiled solvers on a server, build the image from the repo-root
`Dockerfile` and mount `gurobi.lic`. See [DOCKER.md](DOCKER.md).

## Running

Single instance:

```bash
drispi-pipeline data/instances/xl/XL-n2307-k34.vrp \
  --config configs/async_example.yaml \
  --cpus 0-7
```

Equivalent: `python -m drispi.pipeline.runner …`. See `--help` for time limits,
`--gui`, and `--analysis`.

Final 100-XL × 3-seed campaign (6+1+1, 4 slices on 32 cores, ~150 h):

```bash
python scripts/run_final_benchmark.py --dry-run
python scripts/run_final_benchmark.py --cpu-base 0 --total-cores 32
```

Resume is the default: already-`ok` `(instance, seed)` rows in
`artifacts/final_benchmark/campaign.jsonl` are skipped. SP is on, so mount a
Gurobi license. Pin an exclusive 32-core island.

## Configuration

`configs/default.yaml` is the complete reference (every `DRISPIConfig` flag).
Profile YAMLs merge on top; keys they set win.

**Production 8-core slice** is 6 DRI + 1 BG-AILS + 1 SP, with both BG and SP
async. That layout is [`configs/async_example.yaml`](configs/async_example.yaml).
Uncomment the `cores:` block in `default.yaml` or pass that profile with
`--cpus 0-7`.

BG-AILS quality knobs currently in default: `pair_selection: greedy`,
`n_chains_mode: k`, `initial_omega: 10`. The time budget tracks the predicted
decompose-route wall of the current partition:
`max(floor_s, margin * scale * n_waves^wave_exponent * lockstep_sum)`
with knobs under `bg_ails.budget` in `default.yaml`. The hang detector uses
the same predictor, with slack `max(2x, x+120)`.

The HAOS level-1 k domain is no longer a fixed candidate list — it is computed
once per instance at init from `(n, K_min)` in the instance name (tunables
under `decomposition.k_domain`; see `drispi/haos/k_domain.py`). The computed
domain is logged on the run's init JSONL event. The imbalance law
`max_share = 1.062 * k^{-0.795}` is a recorded measurement only and is not
applied at runtime.

SC/SP modes, pinning, and YAML `off` quoting: [docs/sp_sc_modes.md](docs/sp_sc_modes.md).

**YAML 1.1:** always quote `sp_sc.mode: "off"`. Unquoted `off` becomes boolean
`False` and the pipeline silently treats it as sync.
