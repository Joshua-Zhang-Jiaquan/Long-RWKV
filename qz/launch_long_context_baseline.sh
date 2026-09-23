#!/bin/bash
set -euo pipefail
: "${LCM_ROOT:?}" "${LCM_OUT:?}" "${LCM_QUALIFICATION:?}"
cd "$LCM_ROOT"
mkdir -p "$LCM_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--base "${LCM_BASE:?}" --kind "${LCM_KIND:?}" --out "$LCM_OUT")
if [[ "$LCM_QUALIFICATION" == 1 ]]; then args+=(--qualification); else args+=(--qualification-out "${LCM_QUALIFICATION_OUT:?}"); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.long_context_baselines.worker "${args[@]}" > "$LCM_OUT/torchrun.log" 2>&1
