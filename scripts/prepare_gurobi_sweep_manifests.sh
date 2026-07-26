#!/usr/bin/env bash
# Prepare snapshot manifests for the Gurobi thread sweep (in-container VRP paths).
# Run once from ~/drispi on the server before the overnight sweep.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/pool_snapshots/n8389-k2028

cat > artifacts/pool_snapshots/n8389-k2028/snapshot_manifest.json << 'EOF'
{
  "instance": "XL-n8389-k2028",
  "instance_stem": "XL-n8389-k2028",
  "instance_path": "/app/data/instances/xl/XL-n8389-k2028.vrp",
  "seed": 42,
  "cores": 10,
  "marks_min": [20, 60, 90],
  "snapshots": [
    {"filename": "XL-n8389-k2028_20min_seed42.pkl", "mark_min": 20, "pool_size": 0, "unique_route_count": 0},
    {"filename": "XL-n8389-k2028_60min_seed42.pkl", "mark_min": 60, "pool_size": 0, "unique_route_count": 0},
    {"filename": "XL-n8389-k2028_90min_seed42.pkl", "mark_min": 90, "pool_size": 0, "unique_route_count": 0}
  ]
}
EOF

python3 - <<'PY'
import json
from pathlib import Path

pairs = [
    ("artifacts/pool_snapshots/n2307-k34", "/app/data/instances/xl/XL-n2307-k34.vrp"),
    ("artifacts/pool_snapshots/n3975-k687", "/app/data/instances/xl/XL-n3975-k687.vrp"),
    ("artifacts/pool_snapshots/n8389-k2028", "/app/data/instances/xl/XL-n8389-k2028.vrp"),
]
for d, vrp in pairs:
    p = Path(d) / "snapshot_manifest.json"
    m = json.loads(p.read_text())
    m["instance_path"] = vrp
    p.write_text(json.dumps(m, indent=2) + "\n")
    print("ok", p, "->", vrp)
PY

echo "manifests ready"
