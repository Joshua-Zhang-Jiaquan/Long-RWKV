#!/bin/bash
set -euo pipefail
: "${COST_ROOT:?}" "${COST_OUT:?}" "${COST_CHECKPOINT:?}" "${COST_KIND:?}" "${COST_BASE:?}"
cd "$COST_ROOT"
mkdir -p "$COST_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.distance_cost.worker \
 --base "$COST_BASE" --model-root "$COST_ROOT/model_source" --checkpoint "$COST_CHECKPOINT" \
 --kind "$COST_KIND" --out "$COST_OUT" > "$COST_OUT/torchrun.log" 2>&1
