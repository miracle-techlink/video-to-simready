#!/usr/bin/env bash
# Digua hackathon drone demo — server runner.
# Usage: ./run.sh <smoke|record|assets> <out_dir>
#
# GPU modes (smoke/record) run under the shared flock so every participant
# serializes on /root/gpufree-data/.simready/gpu-coordination.lock; start/end
# timestamps and the holder PID are logged next to the run output.
set -euo pipefail

MODE="${1:-smoke}"
OUT="${2:-$HOME/runs/digua-demo}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/root/gpufree-data/.simready/isaac-python.sh
LOCK=/root/gpufree-data/.simready/gpu-coordination.lock

mkdir -p "$OUT"

cd "$HERE"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

if [ "$MODE" = "assets" ]; then
  "$PY" -m digua_hackathon.run_demo --generate-assets --out "$OUT"
  exit 0
fi

if [ "$MODE" != "smoke" ] && [ "$MODE" != "record" ]; then
  echo "unknown mode: $MODE (assets|smoke|record)" >&2
  exit 2
fi

# nvidia-smi pre-check: report (do not kill) tasks outside the lock discipline
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

RUN_LOG="$OUT/run_${MODE}_console.log"
{
  echo "[$(date -Is)] acquiring gpu lock for mode=$MODE pid=$$"
  exec 9>"$LOCK"
  flock 9
  echo "[$(date -Is)] gpu lock held by pid=$$; starting"
  "$PY" -m digua_hackathon.run_demo --mode "$MODE" --out "$OUT"
  status=$?
  echo "[$(date -Is)] finished status=$status; releasing lock"
  exit $status
} 2>&1 | tee "$RUN_LOG"
