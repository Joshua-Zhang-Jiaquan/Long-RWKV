#!/bin/bash
set -euo pipefail
: "${MIX_ROOT:?}" "${MIX_OUT:?}" "${MIX_OBJECTIVE:?}" "${MIX_PARENT:?}" "${MIX_STOP:?}"
cd "$MIX_ROOT"
mkdir -p "$MIX_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
extra=()
if [ -n "${MIX_RESUME:-}" ]; then extra+=(--resume "$MIX_RESUME"); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04mix.worker \
 --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B \
 --model-root "$MIX_ROOT/model_source" --objective "$MIX_OBJECTIVE" --stop-step "$MIX_STOP" \
 --parent "$MIX_PARENT" --out "$MIX_OUT" "${extra[@]}" > "$MIX_OUT/torchrun.log" 2>&1
