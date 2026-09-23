#!/bin/bash
set -euo pipefail
: "${E3_ROOT:?qualified source snapshot required}"
: "${E3_OUT:?output required}"
: "${E3_LENGTH:?length required}"
: "${E3_MODEL:?f2 or r0 required}"
cd "$E3_ROOT"
mkdir -p "$E3_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BIRWKV_LOAD_SLOTS=8 BIRWKV_META_CONSTRUCT=0 PYTHONUNBUFFERED=1
task_args=(--mode confirm
  --panel /inspire/hdd/global_user/zhangjiaquan-253108540222/capability_eval_data/lc_grid324
  --out "$E3_OUT" --length "$E3_LENGTH" --cpu-threads 4
  --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF)
if [[ "$E3_MODEL" == f2 ]]; then
  : "${E3_PROTOCOL:?measured decoder freeze required}"
  exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.e3.gpu_runner \
    "${task_args[@]}" --protocol "$E3_PROTOCOL" --seed 17 --reuse-unchanged-logits \
    --checkpoint /inspire/hdd/global_user/zhangjiaquan-253108540222/research/lacesmm_assets/birwkv_f2_step14000 \
    > "$E3_OUT/torchrun.log" 2>&1
elif [[ "$E3_MODEL" == r0 ]]; then
  : "${E3_QUALIFICATION:?measured causal qualification required}"
  exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.e3.causal_runner \
    "${task_args[@]}" --qualification "$E3_QUALIFICATION" \
    > "$E3_OUT/torchrun.log" 2>&1
else
  echo 'unsupported model' >&2
  exit 2
fi
