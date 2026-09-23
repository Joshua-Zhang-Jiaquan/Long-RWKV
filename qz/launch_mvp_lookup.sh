#!/bin/bash
set -euo pipefail
: "${MVP_ROOT:?immutable MVP stage required}"
: "${MVP_OUT:?separate MVP output required}"
cd "$MVP_ROOT"
mkdir -p "$MVP_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BIRWKV_LOAD_SLOTS=8 BIRWKV_META_CONSTRUCT=0 PYTHONUNBUFFERED=1
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.mvp.lookup_pilot \
  --out "$MVP_OUT" \
  --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF \
  --checkpoint /inspire/hdd/global_user/zhangjiaquan-253108540222/research/lacesmm_assets/birwkv_f2_step14000 \
  > "$MVP_OUT/torchrun.log" 2>&1
