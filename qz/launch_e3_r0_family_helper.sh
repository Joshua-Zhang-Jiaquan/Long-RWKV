#!/bin/bash
set -euo pipefail
: "${E3_ROOT:?original qualified R0 source snapshot required}"
: "${E3_FAMILY_HELPER_CODE:?absolute helper_family.py path required}"
: "${E3_PRIMARY_OUT:?original primary output required}"
: "${E3_OUT:?separate family helper output required}"
: "${E3_LENGTH:?32768 or 65536 required}"
: "${E3_QUALIFICATION:?frozen R0 dev qualification required}"
cd "$E3_ROOT"
mkdir -p "$E3_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BIRWKV_LOAD_SLOTS=8 BIRWKV_META_CONSTRUCT=0 PYTHONUNBUFFEREDWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec torchrun --standalone --nproc_per_node=8 "$E3_FAMILY_HELPER_CODE" \
  --model r0 --qualified-root "$E3_ROOT" --primary-out "$E3_PRIMARY_OUT" --out "$E3_OUT" \
  --length "$E3_LENGTH" --family code_dataflow --qualification "$E3_QUALIFICATION" \
  --panel /inspire/hdd/global_user/zhangjiaquan-253108540222/capability_eval_data/lc_grid324 \
  --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF \
  > "$E3_OUT/torchrun.log" 2>&1
