#!/bin/bash
set -euo pipefail
: "${E3_ROOT:?immutable staged source required}"
: "${E3_OUT:?output directory required}"
: "${E3_MODEL:?f2 or r0 required}"
cd "$E3_ROOT"
mkdir -p "$E3_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BIRWKV_LOAD_SLOTS=8 BIRWKV_META_CONSTRUCT=0 PYTHONUNBUFFERED=1
control_args=(--model "$E3_MODEL" --panel "$E3_ROOT/results/lc_dev_panel" --out "$E3_OUT"
  --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF
  --cpu-threads 4 --seed 17)
if [[ "$E3_MODEL" == f2 ]]; then
  : "${E3_PROTOCOL:?frozen protocol required}"
  control_args+=(--protocol "$E3_PROTOCOL"
    --checkpoint /inspire/hdd/global_user/zhangjiaquan-253108540222/research/lacesmm_assets/birwkv_f2_step14000)
elif [[ "$E3_MODEL" == r0 ]]; then
  : "${E3_QUALIFICATION:?R0 qualification required}"
  control_args+=(--qualification "$E3_QUALIFICATION")
else
  exit 2
fi
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.e3.control_runner \
  "${control_args[@]}" > "$E3_OUT/torchrun.log" 2>&1
