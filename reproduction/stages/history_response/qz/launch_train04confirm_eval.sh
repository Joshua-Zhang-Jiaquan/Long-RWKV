#!/bin/bash
set -euo pipefail
: "${CONFIRM_ROOT:?}" "${CONFIRM_EVAL_OUT:?}" "${CONFIRM_CHECKPOINT:?}"
cd "$CONFIRM_ROOT"
mkdir -p "$CONFIRM_EVAL_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04confirm.evaluate \
 --manifest "$CONFIRM_ROOT/manifest.json" \
 --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B \
 --model-root "$CONFIRM_ROOT/model_source" --checkpoint "$CONFIRM_CHECKPOINT" \
 --out "$CONFIRM_EVAL_OUT" > "$CONFIRM_EVAL_OUT/torchrun.log" 2>&1
