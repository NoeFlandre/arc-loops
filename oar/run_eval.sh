#!/bin/bash
# GPU job on abaca: unpack venv and model cache to node-local space, run loop_eval.py, validate, save JSON.
set -euo pipefail
PRESET=${1:-smoke}
ROOT="$HOME/arc-loops"
export PATH="$HOME/.local/bin:$PATH"
READY=$(ls "$ROOT"/env/READY-* 2>/dev/null | head -1 || true)
[ -n "$READY" ] || { echo "no READY environment: run oar/env_build.sh first" >&2; exit 1; }
FP=${READY##*READY-}
LOCAL="/tmp/arc-run-${OAR_JOB_ID}"
SHM="/dev/shm/arc-hf-${OAR_JOB_ID}"
trap 'rm -rf "$LOCAL" "$SHM"' EXIT
mkdir -p "$LOCAL" "$SHM"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
zstd -dc "$ROOT/env/venv-$FP.tar.zst" | tar -x -C "$LOCAL"
cp -r "$ROOT/hf-cache/." "$SHM/"
export HF_HOME="$SHM" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
LOG="$ROOT/results/logs/${PRESET}-${OAR_JOB_ID}.log"
"$LOCAL/venv/bin/python" -u "$ROOT/loop_eval.py" --preset "$PRESET" --batch-size 8 >"$LOG" 2>"${LOG%.log}.err"
"$LOCAL/venv/bin/python" "$ROOT/collect.py" "$LOG" --preset "$PRESET" --job "$OAR_JOB_ID" --out-dir "$ROOT/results"
