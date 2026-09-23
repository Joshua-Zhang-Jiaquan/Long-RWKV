#!/bin/bash
set -euo pipefail
: "${PROBE_ROOT:?}" "${PROBE_OUT:?}" "${PROBE_CHECKPOINT:?}"
cd "$PROBE_ROOT"
mkdir -p "$PROBE_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.posterior_context_probe.worker \
 --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B \
 --model-root "$PROBE_ROOT/model_source" --checkpoint "$PROBE_CHECKPOINT" \
 --out "$PROBE_OUT" > "$PROBE_OUT/torchrun.log" 2>&1
