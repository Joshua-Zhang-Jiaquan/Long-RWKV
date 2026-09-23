#!/bin/bash
set -euo pipefail
: "${LCM_ROOT:?}" "${LCM_EVAL_OUT:?}" "${LCM_CHECKPOINT:?}"
cd "$LCM_ROOT"
mkdir -p "$LCM_EVAL_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cost_args=()
if [[ "${LCM_COST_ONLY:-0}" == "1" ]]; then cost_args+=(--cost-only); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.long_context_baseline_eval.worker \
 --base "${LCM_BASE:?}" \
 --kind "${LCM_KIND:?}" --checkpoint "$LCM_CHECKPOINT" --out "$LCM_EVAL_OUT" "${cost_args[@]}" > "$LCM_EVAL_OUT/torchrun.log" 2>&1
