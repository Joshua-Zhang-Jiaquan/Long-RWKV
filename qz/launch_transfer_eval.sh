#!/bin/bash
set -euo pipefail
: "${TRANSFER_ROOT:?}" "${TRANSFER_OUT:?}" "${TRANSFER_SPLIT:?}" "${TRANSFER_CHECKPOINT:?}"
cd "$TRANSFER_ROOT"
mkdir -p "$TRANSFER_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 10 > "$TRANSFER_OUT/gpu_usage.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.predictive_transfer.evaluate --base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B --model-root "$TRANSFER_ROOT/model_source" --out "$TRANSFER_OUT" --checkpoint "$TRANSFER_CHECKPOINT" --split "$TRANSFER_SPLIT" > "$TRANSFER_OUT/torchrun.log" 2>&1
