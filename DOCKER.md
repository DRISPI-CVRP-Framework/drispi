# Docker Guide

This guide covers building, testing, pushing, and running DRISPI with Docker. The workflow matches a plain `docker build` / `docker run` style; Docker Compose is optional convenience.

## Table of Contents

1. [Creating Docker Images](#1-creating-docker-images)
2. [Testing Images Locally](#2-testing-images-locally)
3. [Pushing Images to Docker Hub](#3-pushing-images-to-docker-hub)
4. [Tagging Images](#4-tagging-images)
5. [Docker Login on Server](#5-docker-login-on-server)
6. [Server Directory Layout](#6-server-directory-layout)
7. [Pulling Images on Server](#7-pulling-images-on-server)
8. [Running Images on Server](#8-running-images-on-server)
9. [CPU Pinning and Container Naming](#9-cpu-pinning-and-container-naming)
10. [Gurobi Token Server License](#10-gurobi-token-server-license)
11. [Understanding the Dockerfile](#11-understanding-the-dockerfile)
12. [Understanding .dockerignore](#12-understanding-dockerignore)
13. [Optional: Docker Compose](#13-optional-docker-compose)

---

## Path conventions

DRISPI uses these mount paths inside the container:

| Purpose | Host (typical) | Container |
|---------|----------------|-----------|
| Instance data | `./data` | `/app/data` |
| Run outputs | `./artifacts` | `/app/artifacts` |
| Gurobi license | path to `gurobi.lic` | `/root/gurobi.lic` |

Instance files live under `data/instances/…` (e.g. `data/instances/x/X-n106-k14.vrp`).

If you used `instances/` and `output/` in a previous project, the idea is the same — only the directory names differ.

### Non-root and entrypoint

The image default is `uv run python -m drispi.pipeline.runner`. **Do not** set
`--entrypoint uv` when running as a non-root `--user`: `uv` tries to rewrite
`/app/.venv` and fails with permission denied. For a script other than the
pipeline runner, use the venv interpreter:

```bash
docker run ... --user "$(id -u):$(id -g)" \
  --entrypoint /app/.venv/bin/python \
  IMAGE scripts/run_benchmark.py ...
```

Mount `data/` (instances are not in the image). Production 8-core slice is
6 DRI + 1 BG + 1 SP via `configs/async_example.yaml` and `--cpus` matching an
exclusive 8-core island. `--cpuset-cpus` must not be shared (AILS is time-stopped).

---

## 1. Creating Docker Images

Build from the repo root:

```bash
docker build -t drispi:0.1.0 .
```

**What this does:**
- `-t drispi:0.1.0` — local name and version tag (you choose the tag)
- `.` — build context (respects `.dockerignore`)

**Verify the image:**

```bash
docker images drispi
```

**Verify native solvers were copied into the image** (required for `filo`, `filo2`, `ails2`):

```bash
docker run --rm --entrypoint sh drispi:0.1.0 -c \
  'ls -la ext/filo/build/filo ext/filo2/build/filo2 ext/ails2/build/AILSII.jar'
```

Build the solvers on your host under `ext/` before `docker build` if any of these are missing.

---

## 2. Testing Images Locally

### Interactive shell

```bash
docker run -it --rm drispi:0.1.0 /bin/bash
```

### Single instance (default entrypoint)

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  drispi:0.1.0 \
  data/instances/x/X-n106-k14.vrp \
  --config configs/benchmark.yaml \
  --time-limit 60 \
  --output-dir artifacts/docker-smoke
```

Results appear on the host under `./artifacts/docker-smoke/`.

### Benchmark campaign (override entrypoint)

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  -e OMP_NUM_THREADS=1 \
  drispi:0.1.0 \
  run python scripts/run_benchmark.py \
  "data/instances/x/X-n106-k14.vrp" "data/instances/x/X-n110-k13.vrp" \
  --config configs/benchmark.yaml \
  --total-cores 8 \
  --cores-per-instance 4 \
  --time-limit 120 \
  --output-dir artifacts/benchmark-smoke
```

Note the `--entrypoint uv` prefix when using `docker run` directly:

```bash
docker run --rm --entrypoint uv ... drispi:0.1.0 run python scripts/run_benchmark.py ...
```

Or use Compose (see [section 12](#12-optional-docker-compose)), which sets the benchmark entrypoint for you.

---

## 3. Pushing Images to Docker Hub

### Login

```bash
docker login
```

### Tag with your Docker Hub username

```bash
docker tag drispi:0.1.0 yourusername/drispi:0.1.0
```

### Push

```bash
docker push yourusername/drispi:0.1.0
```

Verify at: `https://hub.docker.com/r/yourusername/drispi`

---

## 4. Tagging Images

Use semantic version tags you control:

```bash
docker tag drispi:0.1.0 yourusername/drispi:0.1.0
docker tag drispi:0.1.0 yourusername/drispi:latest   # optional

docker push yourusername/drispi:0.1.0
docker push yourusername/drispi:latest
```

**Best practice:** Pin production runs to a specific version (`0.1.0`), not `latest`.

---

## 5. Docker Login on Server

Same as any Docker Hub workflow:

```bash
ssh user@your-server
docker login
```

Install Docker on the server if needed: https://docs.docker.com/engine/install/

---

## 6. Server Directory Layout

You do **not** need to restructure `~/cvrp/` or move old runs. Create a **sibling directory** for DRISPI (on dantzig this is typically `~/drispi` = `/home/altendeitering/drispi`):

```
/home/altendeitering/
├── cvrp/                  # old project — leave as-is
└── drispi/                # new DRISPI runs
    ├── data/
    │   └── instances/     # .vrp files (copy, rsync, or symlink)
    ├── artifacts/         # run outputs (created automatically)
    ├── gurobi.lic         # token-server license file (see section 10)
    └── docker-compose.server.yml   # optional; copy from repo
```

**What you need on the server:**

| Item | Required? | Notes |
|------|-----------|-------|
| Pulled Docker image | Yes | e.g. `zorroy/drispi:0.2` |
| `data/instances/` | Yes | Large files — keep on disk, not in the image |
| `artifacts/` | Yes | Writable output directory |
| `gurobi.lic` | Yes | Same token-server file as your old project works |
| Full git clone | No | Only if you want compose file / configs on disk |
| Rebuild on server | No | Pull pre-built image only |

**Minimal setup:**

```bash
mkdir -p ~/drispi/{data/instances,artifacts}
# Copy or symlink instances and gurobi.lic
cp /path/to/existing/gurobi.lic ~/drispi/gurobi.lic
# rsync instances as needed
```

Configs (`configs/benchmark.yaml`, etc.) are **inside the image**. Override with CLI flags if needed.

---

## 7. Pulling Images on Server

```bash
cd ~/drispi

docker pull zorroy/drispi:0.2
docker images zorroy/drispi
```

---

## 8. Running Images on Server

DRISPI runs are **batch jobs**: the container starts, solves, writes to `artifacts/`, and exits. You typically do **not** use `--restart unless-stopped` (unlike a long-lived web service).

On a shared server, pin CPUs and encode the range in the container name — see [section 9](#9-cpu-pinning-and-container-naming).

Use `--user "$(id -u):$(id -g)"` so artifacts are owned by you. For a 6+1+1
slice, pass `--config configs/async_example.yaml` (or `configs/final_benchmark.yaml`)
and `--cpus` equal to the pinned 8-core island (not `--n-workers`).

Use `$(pwd)` or `/home/altendeitering/drispi` for volume paths (not `/altendeitering/drispi` unless that path exists on your host).

### Single instance

```bash
cd ~/drispi

docker run --rm --network host \
  --name altendeitering_drispi_x106_cpu0-15 \
  --cpuset-cpus="0-15" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  zorroy/drispi:0.2 \
  data/instances/x/X-n106-k14.vrp \
  --config configs/benchmark.yaml \
  --n-workers 16 \
  --time-limit 7200 \
  --output-dir artifacts/run1
```

`--n-workers 16` matches `--cpuset-cpus="0-15"` (16 cores). Subcluster parallelism is capped at `min(n_workers, n_clusters)`.

### Benchmark campaign

```bash
docker run --rm --network host \
  --name altendeitering_drispi_xl_benchmark_cpu0-15 \
  --cpuset-cpus="0-15" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  -e OMP_NUM_THREADS=1 \
  --entrypoint uv \
  zorroy/drispi:0.2 \
  run python scripts/run_benchmark.py \
  "data/instances/xl/*.vrp" \
  --config configs/benchmark.yaml \
  --total-cores 16 \
  --cores-per-instance 4 \
  --bks-file data/bks/xl-bks.json \
  --time-limit 7200 \
  --output-dir artifacts/benchmark
```

`--total-cores 16` must match the CPU set size. `--cores-per-instance 4` sets queue depth (`16 // 4 = 4` instances in parallel).

For a larger pinned run (10 instances, 32 cores, 8 cores per instance), see [section 9](#example-10-instances-2-h-each-cores-3263).

### Long run in background

For a multi-hour benchmark, detach and follow logs:

```bash
docker run -d --network host \
  --name altendeitering_drispi_xl_benchmark_cpu16-31 \
  --cpuset-cpus="16-31" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  -e OMP_NUM_THREADS=1 \
  --entrypoint uv \
  zorroy/drispi:0.2 \
  run python scripts/run_benchmark.py \
  "data/instances/xl/*.vrp" \
  --config configs/benchmark.yaml \
  --total-cores 16 \
  --cores-per-instance 4 \
  --time-limit 7200 \
  --output-dir artifacts/benchmark

docker logs -f altendeitering_drispi_xl_benchmark_cpu16-31
```

Remove the container when finished: `docker rm altendeitering_drispi_xl_benchmark_cpu16-31`.

---

## 9. CPU Pinning and Container Naming

On a shared university server you often run several jobs side by side, each on a disjoint CPU range. Pinning avoids oversubscription and makes it obvious which cores a container owns.

### Old project vs DRISPI

**Old project (`zorroy/cvrp`):**

```bash
docker run --rm --network host \
  --name altendeitering_cvrp_challenge_400_cpu0-15 \
  --cpuset-cpus="0-15" \
  -v /home/altendeitering/cvrp/challenge_output_final:/app/output \
  zorroy/cvrp:4.0 /app/output --max_workers 16
```

**DRISPI equivalent:**

| Old flag | DRISPI equivalent | Notes |
|----------|-------------------|-------|
| `--cpuset-cpus="0-15"` | same | Pins container to 16 cores |
| `--name …_cpu0-15` | same pattern | Encode CPU range in the name |
| `--max_workers 16` | `--n-workers 16` (single instance) | Subcluster worker cap |
| `--max_workers 16` | `--total-cores 16` (benchmark) | Shared core pool for `CoreManager` |

Always keep `OMP_NUM_THREADS=1` (and related env vars) so each worker uses one thread inside the pinned set.

### Naming convention

Use a descriptive, unique name that includes the CPU range:

```
altendeitering_drispi_<job>_<instance-or-set>_cpu<range>
```

Examples:

| Name | Meaning |
|------|---------|
| `altendeitering_drispi_x106_cpu0-15` | Single X instance on cores 0–15 |
| `altendeitering_drispi_xl_benchmark_cpu0-15` | XL benchmark on cores 0–15 |
| `altendeitering_drispi_xl_benchmark_cpu16-31` | Second benchmark on cores 16–31 |
| `altendeitering_drispi_10xl_benchmark_cpu32-63` | 10-instance XL benchmark on cores 32–63 |

This mirrors `altendeitering_cvrp_challenge_400_cpu0-15` from the old workflow.

### Matching cores: checklist

1. Count cores in `--cpuset-cpus` (e.g. `"0-15"` → 16 cores).
2. Set DRISPI to the same budget:
   - **Single instance:** `--n-workers 16` (or rely on `n_workers` in config if already 16).
   - **Benchmark:** `--total-cores 16`.
3. Pick `--cores-per-instance` for benchmark queue depth (e.g. `4` → up to 4 instances at once on a 16-core set).
4. Put the same range in `--name` (`cpu0-15`).
5. Use a **different** `--cpuset-cpus` and `--name` for each concurrent container.

### Running two jobs on one server

```bash
# Job A — cores 0–15
docker run -d --network host \
  --name altendeitering_drispi_xl_benchmark_cpu0-15 \
  --cpuset-cpus="0-15" \
  ... --total-cores 16 --cores-per-instance 4 ...

# Job B — cores 16–31 (no overlap)
docker run -d --network host \
  --name altendeitering_drispi_xl_benchmark_cpu16-31 \
  --cpuset-cpus="16-31" \
  ... --total-cores 16 --cores-per-instance 4 ...
```

Monitor:

```bash
docker ps --filter name=altendeitering_drispi
docker logs -f altendeitering_drispi_xl_benchmark_cpu0-15
htop   # verify load stays on the expected CPUs
```

### Example: 10 instances, 2 h each, cores 32–63

**Setup:** 10 XL instances, 2 h time limit per instance (`7200` s), Docker pinned to cores 32–63 (32 cores), `--cores-per-instance 8` → up to **4 instances in parallel** (`32 // 8`). Estimated wall time ≈ **6 h** (three waves: 4 + 4 + 2 instances × 2 h).

Dry-run first (no solves):

```bash
cd ~/drispi

docker run --rm --network host \
  --cpuset-cpus="32-63" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  -e OMP_NUM_THREADS=1 \
  --entrypoint uv \
  zorroy/drispi:0.2 \
  run python scripts/run_benchmark.py \
  data/instances/xl/XL-n10001-k1570.vrp \
  data/instances/xl/XL-n1048-k237.vrp \
  data/instances/xl/XL-n1094-k157.vrp \
  data/instances/xl/XL-n1141-k112.vrp \
  data/instances/xl/XL-n1188-k96.vrp \
  data/instances/xl/XL-n1234-k55.vrp \
  data/instances/xl/XL-n1281-k29.vrp \
  data/instances/xl/XL-n1328-k19.vrp \
  data/instances/xl/XL-n1374-k278.vrp \
  data/instances/xl/XL-n1421-k232.vrp \
  --config configs/benchmark.yaml \
  --total-cores 32 \
  --cores-per-instance 8 \
  --time-limit 7200 \
  --bks-file data/bks/xl-bks.json \
  --output-dir artifacts/benchmark_10xl_cpu32-63 \
  --dry-run
```

Full run (detached):

```bash
cd ~/drispi

docker run -d --network host \
  --name altendeitering_drispi_10xl_benchmark_cpu32-63 \
  --cpuset-cpus="32-63" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  --entrypoint uv \
  zorroy/drispi:0.2 \
  run python scripts/run_benchmark.py \
  data/instances/xl/XL-n10001-k1570.vrp \
  data/instances/xl/XL-n1048-k237.vrp \
  data/instances/xl/XL-n1094-k157.vrp \
  data/instances/xl/XL-n1141-k112.vrp \
  data/instances/xl/XL-n1188-k96.vrp \
  data/instances/xl/XL-n1234-k55.vrp \
  data/instances/xl/XL-n1281-k29.vrp \
  data/instances/xl/XL-n1328-k19.vrp \
  data/instances/xl/XL-n1374-k278.vrp \
  data/instances/xl/XL-n1421-k232.vrp \
  --config configs/benchmark.yaml \
  --total-cores 32 \
  --cores-per-instance 8 \
  --time-limit 7200 \
  --bks-file data/bks/xl-bks.json \
  --output-dir artifacts/benchmark_10xl_cpu32-63

docker logs -f altendeitering_drispi_10xl_benchmark_cpu32-63
```

**Flag summary:**

| Flag | Value | Meaning |
|------|-------|---------|
| `--cpuset-cpus` | `"32-63"` | 32 physical cores for this container |
| `--total-cores` | `32` | Must match cpuset size; `CoreManager` budget |
| `--cores-per-instance` | `8` | Queue slot size; `32 // 8 = 4` concurrent instances |
| `--time-limit` | `7200` | 2 h per instance |
| `--name` | `…_cpu32-63` | CPU range visible in `docker ps` |

Replace the ten instance paths with your chosen `.vrp` files. Swap instance paths for a glob only if it matches exactly the set you intend (e.g. `"data/instances/xl/XL-n1*.vrp"` — verify count with `--dry-run`).

---

## 10. Gurobi Token Server License

Your `gurobi.lic` is a **token-server** file (contains `TOKENSERVER=…` and `PORT=…`). That is the correct format — same as your old project.

**Important:** The license file is **not baked into the image**. The Dockerfile only sets where Gurobi looks for it:

```dockerfile
ENV GRB_LICENSE_FILE=/root/gurobi.lic
```

At runtime you **mount** your license file:

```bash
-v /path/to/gurobi.lic:/root/gurobi.lic:ro
```

The file tells Gurobi *which token server to contact*; the container still needs **network access** to that server.

### Why `--network host` on the server

Gurobi token servers often require the client to appear on the same network subnet as the server. Docker’s default bridge network puts the container on an isolated subnet, which can cause license errors even with a correct `gurobi.lic`.

On a **native Linux server** (your university server), use:

```bash
docker run --rm --network host ...
```

This matches your old project setup and lets the container reach the token server the same way the host does.

**Platform notes:**
- **Native Linux server** — `--network host` works; recommended.
- **Docker Desktop / WSL2** — `--network host` is unreliable; prefer running outside Docker or use a node-locked license for local dev.

Do **not** commit `gurobi.lic` to git (it is in `.gitignore`).

---

## 11. Understanding the Dockerfile

```dockerfile
FROM python:3.11-slim
```

Python 3.11 is required for compatibility with `scikit-learn-extra` (kmedoids clustering).

```dockerfile
WORKDIR /app
```

All paths in commands are relative to `/app`.

**System packages:** JRE (AILS2), build tools, cmake (native solvers).

**Python environment:** `uv sync --frozen` installs locked dependencies from `uv.lock`.

**Project files copied:** `drispi/`, `scripts/`, `configs/`, `data/bks/`, pre-built `ext/` solvers, `README.md`.

**Not copied:** instance files (`data/instances/`), run artifacts, tests, git history.

**Default entrypoint:**

```dockerfile
ENTRYPOINT ["uv", "run", "python", "-m", "drispi.pipeline.runner"]
```

Arguments after the image name are passed to the pipeline runner. Override with `--entrypoint` for benchmark mode.

---

## 12. Understanding .dockerignore

Excludes files from the build context to keep builds fast and images small:

| Excluded | Why |
|----------|-----|
| `.git/`, `tests/`, `docs/` | Not needed at runtime |
| `artifacts/` | Runtime output — use mounted volume |
| `.venv/`, `__pycache__/` | Rebuilt inside the image |
| `ext/**/results/`, `ext/**/instances/` | Large solver test data |

Instance `.vrp` files are also excluded from the context (they live under `data/` on the host and are mounted at run time).

**Security:** Never copy `gurobi.lic` into the image.

---

## 13. Optional: Docker Compose

Compose encodes the same volumes and environment as the `docker run` examples above.

### Log timestamps (`DRISPI_TZ`)

Log lines, run directory names, and dashboard snapshots use an explicit timezone
(default **UTC**). Set `DRISPI_TZ` to your IANA zone (e.g. `Europe/Berlin`) when
starting a container:

```bash
export DRISPI_TZ=Europe/Berlin
docker compose run --rm drispi data/instances/x/X-n106-k14.vrp --time-limit 60
```

`DRISPI_TZ` takes precedence over the standard `TZ` variable. The image ships
with `tzdata` so named zones resolve correctly.

### Local (build + run)

```bash
export DRISPI_IMAGE=yourusername/drispi:0.1.0

docker compose build
docker push "$DRISPI_IMAGE"

docker compose run --rm drispi \
  data/instances/x/X-n106-k14.vrp --time-limit 60 --output-dir artifacts/smoke

docker compose run --rm benchmark \
  "data/instances/x/*.vrp" \
  --total-cores 8 --cores-per-instance 4 --time-limit 120 \
  --output-dir artifacts/benchmark-smoke
```

### Server (pull only, no build)

Copy `docker-compose.server.yml` to the server, then:

```bash
export DRISPI_IMAGE=yourusername/drispi:0.1.0
export GRB_LICENSE_FILE=/altendeitering/drispi/gurobi.lic

docker compose -f docker-compose.server.yml pull

docker compose -f docker-compose.server.yml run --rm drispi \
  data/instances/x/X-n106-k14.vrp --config configs/benchmark.yaml --time-limit 7200
```

For token-server licenses on Linux, add `network_mode: host` to the service in a local override file, or prefer plain `docker run --network host` from section 8.

CPU pinning (`cpuset_cpus`, container name) is not set in the stock compose files — use `docker run` on the server when you need explicit core ranges.

---

## Complete Workflow Example

### Local machine

```bash
# 1. Build
docker build -t drispi:0.1.0 .

# 2. Test
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  drispi:0.1.0 \
  data/instances/x/X-n106-k14.vrp --time-limit 60 --output-dir artifacts/smoke

# 3. Tag and push
docker tag drispi:0.1.0 yourusername/drispi:0.1.0
docker login
docker push yourusername/drispi:0.1.0
```

### Server

```bash
mkdir -p ~/drispi/{data/instances,artifacts}
# copy gurobi.lic and instances

docker login
docker pull zorroy/drispi:0.2

docker run --rm --network host \
  --name altendeitering_drispi_x106_cpu0-15 \
  --cpuset-cpus="0-15" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/gurobi.lic:/root/gurobi.lic:ro" \
  -e GRB_LICENSE_FILE=/root/gurobi.lic \
  -e OMP_NUM_THREADS=1 \
  zorroy/drispi:0.2 \
  data/instances/x/X-n106-k14.vrp \
  --config configs/benchmark.yaml \
  --n-workers 16 \
  --time-limit 7200 \
  --output-dir artifacts/run1
```

---

## Useful Commands

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Shell inside a running container
docker exec -it <container-name> /bin/bash

# Follow logs
docker logs -f <container-name>
```
