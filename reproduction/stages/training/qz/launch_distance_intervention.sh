#!/bin/bash
set -euo pipefail
: "${DISTANCE_ROOT:?}" "${DISTANCE_OUT:?}" "${DISTANCE_ARM:?}" "${DISTANCE_SEED:?}" "${DISTANCE_STEPS:?}"
cd "$DISTANCE_ROOT"
mkdir -p "$DISTANCE_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--base /inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B --model-root "$DISTANCE_ROOT/model_source" --out "$DISTANCE_OUT" --initial-checkpoint "${DISTANCE_INITIAL:?}" --initial-sha256 "${DISTANCE_INITIAL_SHA256:?}" --arm "$DISTANCE_ARM" --seed "$DISTANCE_SEED" --steps "$DISTANCE_STEPS")
if [[ "${DISTANCE_QUALIFICATION:?}" == 1 ]]; then args+=(--qualification); else args+=(--qualification-out "${DISTANCE_QUALIFICATION_OUT:?}"); fi
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 10 > "$DISTANCE_OUT/gpu_usage.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.distance_intervention.worker "${args[@]}" > "$DISTANCE_OUT/torchrun.log" 2>&1
