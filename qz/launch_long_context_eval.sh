#!/bin/bash
set -euo pipefail
: "${LCM_ROOT:?}" "${LCM_EVAL_OUT:?}" "${LCM_CHECKPOINT:?}"
cd "$LCM_ROOT"
mkdir -p "$LCM_EVAL_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
diagnostic_args=()
if [[ "${LCM_DIAGNOSTIC_STEP200:-0}" == "1" ]]; then diagnostic_args+=(--diagnostic-step200); fi
if [[ "${LCM_COST_ONLY:-0}" == "1" ]]; then diagnostic_args+=(--cost-only); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.long_context_eval.worker \
 --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B \
 --model-root "$LCM_ROOT/model_source" --checkpoint "$LCM_CHECKPOINT" --out "$LCM_EVAL_OUT" "${diagnostic_args[@]}" > "$LCM_EVAL_OUT/torchrun.log" 2>&1
