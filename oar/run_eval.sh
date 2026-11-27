#!/bin/bash
# Job on abaca (GPU or CPU node): unpack venv and model cache to node-local space, run loop_eval.py, validate, save JSON.
# Usage: run_eval.sh <smoke|full>   env: BATCH (default 8; use 16 on CPU), OAR_JOB_ID set by OAR.
set -euo pipefail
PRESET=${1:-smoke}
SHARD=${2:-0/1}   # k/N: run the k-th of N shards of each test set
ROOT="$HOME/arc-loops"
export PATH="$HOME/.local/bin:$PATH"
READY=$(ls "$ROOT"/env/READY-* 2>/dev/null | head -1 || true)
[ -n "$READY" ] || { echo "no READY environment on this site: run oar/env_build.sh first" >&2; exit 1; }
FP=${READY##*READY-}
LOCAL="/tmp/arc-run-${OAR_JOB_ID}"
SHM="/dev/shm/arc-hf-${OAR_JOB_ID}"
trap 'rm -rf "$LOCAL" "$SHM"' EXIT
mkdir -p "$LOCAL" "$SHM"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null; then
  :
else
  echo "no usable GPU on this node: CPU run (float32)"
fi
zstd -dc "$ROOT/env/venv-$FP.tar.zst" | tar -x -C "$LOCAL"
cp -r "$ROOT/hf-cache/." "$SHM/"
export HF_HOME="$SHM" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="$(nproc)"
LOG="$ROOT/results/logs/${PRESET}-${OAR_JOB_ID}.log"
"$LOCAL/venv/bin/python" -u "$ROOT/loop_eval.py" --preset "$PRESET" --shard "$SHARD" --batch-size "${BATCH:-8}" >"$LOG" 2>"${LOG%.log}.err"
if [ "$SHARD" = "0/1" ]; then
  "$LOCAL/venv/bin/python" "$ROOT/collect.py" "$LOG" --preset "$PRESET" --job "$OAR_JOB_ID" --out-dir "$ROOT/results"
fi
