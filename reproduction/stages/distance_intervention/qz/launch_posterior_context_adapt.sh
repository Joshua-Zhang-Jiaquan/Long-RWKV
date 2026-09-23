#!/bin/bash
set -euo pipefail
: "${LCM_ROOT:?}" "${LCM_OUT:?}" "${LCM_QUALIFICATION:?}"
cd "$LCM_ROOT"
mkdir -p "$LCM_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B --model-root "$LCM_ROOT/model_source" --out "$LCM_OUT" --initial-checkpoint "${LCT_INITIAL_CHECKPOINT:?}" --initial-sha256 "${LCT_INITIAL_SHA256:?}")
if [[ "$LCM_QUALIFICATION" == 1 ]]; then args+=(--qualification); else args+=(--qualification-out "${LCM_QUALIFICATION_OUT:?}"); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.posterior_context_adapt.worker "${args[@]}" > "$LCM_OUT/torchrun.log" 2>&1
