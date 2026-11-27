#!/bin/bash
# CPU job on abaca: build the venv on node-local /tmp, pack it onto NFS, cache model + ARC once per site.
set -euo pipefail
ROOT="$HOME/arc-loops"
export PATH="$HOME/.local/bin:$PATH" UV_PYTHON_PREFERENCE=only-system
export UV_CACHE_DIR="/tmp/uv-cache-${OAR_JOB_ID}" HF_HOME="$ROOT/hf-cache" HF_HUB_DISABLE_PROGRESS_BARS=1
WORK="/tmp/arc-env-${OAR_JOB_ID}"
trap 'rm -rf "$WORK" "$UV_CACHE_DIR"' EXIT
MODEL=Qwen/Qwen3-0.6B-Base; MODEL_REV=da87bfb608c14b7cf20ba1ce41287e8de496c0cd
DATA=allenai/ai2_arc; DATA_REV=210d026faf9955653af8916fad021475a3f00453
FP=$( (cat "$ROOT/env/requirements.txt"; echo "$MODEL@$MODEL_REV"; echo "$DATA@$DATA_REV") | sha256sum | cut -c1-16)
READY="$ROOT/env/READY-$FP"
echo "fingerprint $FP; /tmp free: $(df -h /tmp | tail -1)"
if [ -f "$READY" ]; then echo "environment already ready"; exit 0; fi
mkdir -p "$WORK" "$ROOT/env" "$HF_HOME"
uv venv -q --python 3.13 "$WORK/venv"
uv pip install -q --python "$WORK/venv/bin/python" -r "$ROOT/env/requirements.txt"
uv pip freeze --python "$WORK/venv/bin/python" > "$ROOT/env/lock-$FP.txt"
"$WORK/venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download("Qwen/Qwen3-0.6B-Base", revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
for cfg in ("ARC-Easy", "ARC-Challenge"):
    load_dataset("allenai/ai2_arc", cfg, revision="210d026faf9955653af8916fad021475a3f00453")
print("cached model and ARC")
PY
tar -C "$WORK" -cf - venv | zstd -T0 -3 -q -f -o "$ROOT/env/venv-$FP.tar.zst"
find "$ROOT/env" -maxdepth 1 -name 'venv-*.tar.zst' ! -name "venv-$FP.tar.zst" -delete
find "$ROOT/env" -maxdepth 1 -name 'READY-*' -delete
touch "$READY"
echo "ready $FP; env size $(du -h "$ROOT/env/venv-$FP.tar.zst" | cut -f1); home $(du -sh "$ROOT" | cut -f1)"
