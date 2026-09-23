#!/bin/bash
set -euo pipefail
: "${PAIR_ROOT:?}" "${PAIR_OUT:?}" "${PAIR_PARENT:?}" "${PAIR_MODE:?}" "${PAIR_UPDATES:?}"
cd "$PAIR_ROOT"
mkdir -p "$PAIR_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
extra=()
if [ -n "${PAIR_RESUME:-}" ]; then extra+=(--resume "$PAIR_RESUME"); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04paired.worker \
 --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B \
 --model-root "$PAIR_ROOT/model_source" --mode "$PAIR_MODE" --updates "$PAIR_UPDATES" \
 --parent "$PAIR_PARENT" --out "$PAIR_OUT" "${extra[@]}" > "$PAIR_OUT/torchrun.log" 2>&1
