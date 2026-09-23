#!/bin/bash
set -euo pipefail
: "${TRAIN04_ROOT:?stage required}"
: "${TRAIN04_OUT:?output required}"
: "${TRAIN04_MODE:?mode required}"
: "${TRAIN04_SEED:?seed required}"
cd "$TRAIN04_ROOT"
mkdir -p "$TRAIN04_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
base=/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B
if [ "$TRAIN04_MODE" = probe ]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04.runtime_probe \
    --base "$base" --model-root "$TRAIN04_ROOT/model_source" --checkpoint "${TRAIN04_CHECKPOINT:?}" \
    --out "$TRAIN04_OUT" > "$TRAIN04_OUT/torchrun.log" 2>&1
fi
if [ "$TRAIN04_MODE" = exact ]; then
  exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04.exact_audit \
    --base "$base" --model-root "$TRAIN04_ROOT/model_source" --manifest "$TRAIN04_ROOT/exact_manifest.json" \
    --out "$TRAIN04_OUT" > "$TRAIN04_OUT/torchrun.log" 2>&1
fi
if [ "$TRAIN04_MODE" = eval ]; then
  exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04.evaluate \
    --base "$base" --model-root "$TRAIN04_ROOT/model_source" --checkpoint "${TRAIN04_CHECKPOINT:?}" \
    --out "$TRAIN04_OUT" > "$TRAIN04_OUT/torchrun.log" 2>&1
fi
extra=()
if [ "$TRAIN04_MODE" = train ]; then extra+=(--smoke-receipt "${TRAIN04_SMOKE:?}"); fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.train04.worker \
  --mode "$TRAIN04_MODE" --seed "$TRAIN04_SEED" --init-seed 0 --grad-accum 4 \
  --lr 0.0001 --weight-decay 0.01 --checkpoint "$base" --model-root "$TRAIN04_ROOT/model_source" \
  --out "$TRAIN04_OUT" "${extra[@]}" > "$TRAIN04_OUT/torchrun.log" 2>&1
