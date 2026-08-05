#!/usr/bin/env bash
# C3 long convergence/variance campaign on dantzig.
#
# Full 32 cores, 7+1 × 4 concurrent slices (real deployed contention).
# 3 instances × 4 seeds = 12 runs → 3 waves × 8h ≈ 24h wall.
# Intermediate marks (1h/2h/4h/6h/8h) logged into each run.jsonl + campaign.jsonl.
#
# Usage (from ~/drispi on the server):
#   bash scripts/run_c3_long_convergence_dantzig.sh --dry-run
#   bash scripts/run_c3_long_convergence_dantzig.sh
#   # background:
#   nohup bash scripts/run_c3_long_convergence_dantzig.sh \
#     > artifacts/c3_long_convergence/orchestrator.log 2>&1 &
#
# Env overrides:
#   DRISPI_IMAGE   docker image (default: zorroy/drispi:0.2)
#   C3_CPU_BASE    first CPU of the 32-core island (default: 0 → 0-31)
#   C3_NAME        docker container name prefix
#   GRB_LICENSE_FILE  host path to gurobi.lic (default: $ROOT/gurobi.lic)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/c3_long_convergence

IMAGE="${DRISPI_IMAGE:-zorroy/drispi:0.2}"
CPU_BASE="${C3_CPU_BASE:-0}"
CPU_END=$((CPU_BASE + 31))
CPUSET="${CPU_BASE}-${CPU_END}"
NAME="${C3_NAME:-altendeitering_drispi_c3_long_cpu${CPUSET}}"
LIC_HOST="${GRB_LICENSE_FILE:-${ROOT}/gurobi.lic}"

DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    EXTRA_ARGS+=("$arg")
  fi
done

if [[ ! -f "$LIC_HOST" ]]; then
  echo "ERROR: Gurobi license not found at $LIC_HOST" >&2
  echo "Set GRB_LICENSE_FILE or place gurobi.lic in $ROOT/" >&2
  exit 1
fi

echo "$(date -Is) C3 dantzig trigger"
echo "  image     = $IMAGE"
echo "  cpuset    = $CPUSET  (cpu_base=$CPU_BASE)"
echo "  container = $NAME"
echo "  license   = $LIC_HOST"
echo "  dry_run   = $DRY_RUN"

# Mount live pipeline + config so async/C3 land without rebuilding the image.
# data/artifacts stay on the host; do not mount a partial tree over /app.
COMMON_DOCKER_ARGS=(
  --network host
  --user "$(id -u):$(id -g)"
  --cpuset-cpus="$CPUSET"
  -v "${ROOT}/data:/app/data:ro"
  -v "${ROOT}/artifacts:/app/artifacts"
  -v "${ROOT}/configs:/app/configs:ro"
  -v "${ROOT}/scripts/run_c3_long_convergence.py:/app/scripts/run_c3_long_convergence.py:ro"
  -v "${ROOT}/drispi:/app/drispi:ro"
  -v "${LIC_HOST}:/tmp/gurobi.lic:ro"
  -e GRB_LICENSE_FILE=/tmp/gurobi.lic
  -e HOME=/tmp
  -e DRISPI_TZ=Europe/Berlin
  -e OMP_NUM_THREADS=1
  -e OPENBLAS_NUM_THREADS=1
  -e MKL_NUM_THREADS=1
  -e PYTHONUNBUFFERED=1
)

PY_ARGS=(
  scripts/run_c3_long_convergence.py
  --config configs/c3_long_convergence.yaml
  --cpu-base "$CPU_BASE"
  --total-cores 32
  --cores-per-slice 8
  --duration-hours 8
  --marks 60 120 240 360 480
  --seeds 42 43 44 45
  --output-dir artifacts/c3_long_convergence/runs
  --campaign-jsonl artifacts/c3_long_convergence/campaign.jsonl
  --bks-file data/bks/xl-bks.json
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  PY_ARGS+=(--dry-run)
  echo "$(date -Is) dry-run inside container (no solves)…"
  docker run --rm "${COMMON_DOCKER_ARGS[@]}" \
    --entrypoint /app/.venv/bin/python \
    "$IMAGE" \
    "${PY_ARGS[@]}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
  exit 0
fi

docker rm -f "$NAME" 2>/dev/null || true

echo "$(date -Is) starting detached container $NAME"
docker run -d "${COMMON_DOCKER_ARGS[@]}" \
  --name "$NAME" \
  --entrypoint /app/.venv/bin/python \
  "$IMAGE" \
  "${PY_ARGS[@]}" \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "$(date -Is) launched. Follow with:"
echo "  docker logs -f $NAME"
echo "  # campaign summary:"
echo "  tail -f artifacts/c3_long_convergence/campaign.jsonl"
