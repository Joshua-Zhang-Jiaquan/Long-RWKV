#!/bin/bash
set -euo pipefail
: "${CONFIRM_ROOT:?}" "${CONFIRM_OUT:?}" "${CONFIRM_SEED:?}" "${CONFIRM_PHASE:?}"
cd "$CONFIRM_ROOT"
mkdir -p "$CONFIRM_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--manifest "$CONFIRM_ROOT/manifest.json" --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B
      --model-root "$CONFIRM_ROOT/model_source" --out "$CONFIRM_OUT" --seed "$CONFIRM_SEED" --phase "$CONFIRM_PHASE")
if [[ "${CONFIRM_QUALIFICATION:-0}" == 1 ]]; then args+=(--qualification); fi
if [[ -n "${CONFIRM_PARENT:-}" ]]; then args+=(--parent "$CONFIRM_PARENT"); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04confirm.worker "${args[@]}" > "$CONFIRM_OUT/torchrun.log" 2>&1
