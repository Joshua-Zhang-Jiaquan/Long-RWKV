#!/bin/bash
set -euo pipefail
: "${DEV_ROOT:?}" "${DEV_OUT:?}" "${DEV_OBJECTIVE:?}" "${DEV_STOP:?}"
cd "$DEV_ROOT"
mkdir -p "$DEV_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
extra=()
if [ -n "${DEV_RESUME:-}" ]; then extra+=(--resume "$DEV_RESUME"); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04dev.worker \
 --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B \
 --model-root "$DEV_ROOT/model_source" --objective "$DEV_OBJECTIVE" --stop-step "$DEV_STOP" \
 --out "$DEV_OUT" "${extra[@]}" > "$DEV_OUT/torchrun.log" 2>&1
