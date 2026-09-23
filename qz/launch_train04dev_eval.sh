#!/bin/bash
set -euo pipefail
: "${DEV_EVAL_ROOT:?}" "${DEV_EVAL_OUT:?}" "${DEV_CHECKPOINT:?}"
cd "$DEV_EVAL_ROOT"
mkdir -p "$DEV_EVAL_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04dev.evaluate \
 --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B \
 --model-root "$DEV_EVAL_ROOT/model_source" --checkpoint "$DEV_CHECKPOINT" \
 --out "$DEV_EVAL_OUT" > "$DEV_EVAL_OUT/torchrun.log" 2>&1
