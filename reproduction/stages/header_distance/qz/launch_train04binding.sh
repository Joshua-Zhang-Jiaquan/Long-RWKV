#!/bin/bash
set -euo pipefail
: "${BIND_ROOT:?}" "${BIND_OUT:?}" "${BIND_OBJECTIVE:?}" "${BIND_PARENT:?}" "${BIND_STOP:?}"
cd "$BIND_ROOT"
mkdir -p "$BIND_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
extra=()
if [ -n "${BIND_RESUME:-}" ]; then extra+=(--resume "$BIND_RESUME"); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04binding.worker \
 --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B \
 --model-root "$BIND_ROOT/model_source" --objective "$BIND_OBJECTIVE" --mask-law "${BIND_MASK_LAW:?}" --stop-step "$BIND_STOP" \
 --parent "$BIND_PARENT" --out "$BIND_OUT" "${extra[@]}" > "$BIND_OUT/torchrun.log" 2>&1
