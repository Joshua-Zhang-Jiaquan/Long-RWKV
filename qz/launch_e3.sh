#!/bin/bash
set -euo pipefail
: "${E3_ROOT:?immutable staged source required}"
: "${E3_OUT:?output path required}"
cd "$E3_ROOT"
mkdir -p "$E3_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BIRWKV_LOAD_SLOTS=8 BIRWKV_META_CONSTRUCT=0
export PYTHONUNBUFFERED=1
exec torchrun --standalone --nproc_per_node=8 -m lrwkv_evidence.e3.gpu_runner \
  --mode dev --panel "$E3_ROOT/results/lc_dev_panel" --out "$E3_OUT" \
  --checkpoint /inspire/hdd/global_user/zhangjiaquan-253108540222/research/lacesmm_assets/birwkv_f2_step14000 \
  --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF \
  --cpu-threads 4 --reuse-unchanged-logits > "$E3_OUT/torchrun.log" 2>&1
