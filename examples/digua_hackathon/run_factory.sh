#!/usr/bin/env bash
# Factory+Iris runner (GPU modes flock-wrapped like run.sh).
# Usage: ./run_factory.sh <smoke|record|preview> <out_dir>
set -euo pipefail

MODE="${1:-smoke}"
OUT="${2:-/root/gpufree-data/mingkang/digua-hackathon-drone/runs/factory}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/root/gpufree-data/.simready/isaac-python.sh
LOCK=/root/gpufree-data/.simready/gpu-coordination.lock

mkdir -p "$OUT"
cd "$HERE"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
RUN_LOG="$OUT/run_${MODE}_console.log"
{
  echo "[$(date -Is)] acquiring gpu lock for factory mode=$MODE pid=$$"
  exec 9>"$LOCK"
  flock 9
  echo "[$(date -Is)] gpu lock held by pid=$$; starting factory_$MODE"
  "$PY" -m digua_hackathon.factory_demo --mode "$MODE" --out "$OUT"
  status=$?
  echo "[$(date -Is)] finished status=$status; releasing lock"
  exit $status
} 2>&1 | tee "$RUN_LOG"
