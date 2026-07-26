#!/usr/bin/env bash
# Overnight Gurobi Threads sweep orchestrator for dantzig.
# Wave 1: n2307 on CPUs 0-15 + n3975 on 16-31 (parallel)
# Wave 2: n8389 on CPUs 0-15 (after wave 1)
#
# Usage (from ~/drispi on the server):
#   bash scripts/run_gurobi_sweeps_overnight.sh
#   # or background:
#   nohup bash scripts/run_gurobi_sweeps_overnight.sh > artifacts/sweeps/orchestrator.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/sweeps

IMAGE="${DRISPI_IMAGE:-altendeitering_sp-benchmark}"
THREADS="${SWEEP_THREADS:-1 2 4 8 16}"
REPEATS="${SWEEP_REPEATS:-3}"
MODE="${SWEEP_MODE:-sp}"
TIME_LIMIT="${SWEEP_TIME_LIMIT:-300}"

run_sweep () {
  local name="$1" cpus="$2" snaps="$3" out="$4"
  docker rm -f "$name" 2>/dev/null || true
  # Mount only the updated files — do NOT bind-mount the whole drispi/ tree
  # (that would hide the image's complete package behind a partial host copy).
  # shellcheck disable=SC2086
  docker run -d --network host \
    --name "$name" \
    --user "$(id -u):$(id -g)" \
    --cpuset-cpus="$cpus" \
    -v "${ROOT}/data:/app/data:ro" \
    -v "${ROOT}/artifacts:/app/artifacts" \
    -v "${ROOT}/configs:/app/configs:ro" \
    -v "${ROOT}/scripts/sweep_gurobi_threads.py:/app/scripts/sweep_gurobi_threads.py:ro" \
    -v "${ROOT}/drispi/sp/model.py:/app/drispi/sp/model.py:ro" \
    -v "${ROOT}/gurobi.lic:/tmp/gurobi.lic:ro" \
    -e GRB_LICENSE_FILE=/tmp/gurobi.lic \
    -e HOME=/tmp \
    -e DRISPI_TZ=Europe/Berlin \
    -e OMP_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    --entrypoint /app/.venv/bin/python \
    "$IMAGE" \
    scripts/sweep_gurobi_threads.py \
      --snapshots-dir "$snaps" \
      --threads $THREADS \
      --repeats "$REPEATS" \
      --mode "$MODE" \
      --time-limit "$TIME_LIMIT" \
      --output "$out"
  echo "$(date -Is) started $name on CPUs $cpus -> $out"
}

echo "$(date -Is) wave 1: n2307 on 0-15, n3975 on 16-31"
run_sweep altendeitering_sweep_n2307-k34 0-15 \
  artifacts/pool_snapshots/n2307-k34 \
  artifacts/sweeps/n2307-k34_gurobi_threads.jsonl

run_sweep altendeitering_sweep_n3975-k687 16-31 \
  artifacts/pool_snapshots/n3975-k687 \
  artifacts/sweeps/n3975-k687_gurobi_threads.jsonl

echo "$(date -Is) waiting for wave 1..."
docker wait altendeitering_sweep_n2307-k34
docker wait altendeitering_sweep_n3975-k687
echo "$(date -Is) wave 1 done"

echo "$(date -Is) wave 2: n8389 on 0-15"
run_sweep altendeitering_sweep_n8389-k2028 0-15 \
  artifacts/pool_snapshots/n8389-k2028 \
  artifacts/sweeps/n8389-k2028_gurobi_threads.jsonl

docker wait altendeitering_sweep_n8389-k2028
echo "$(date -Is) all sweeps finished"
