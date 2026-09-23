#!/bin/bash
set -euo pipefail
: "${E3_ROOT:?original qualified source snapshot required}"
: "${E3_HELPER_CODE:?absolute helper_suffix.py path required}"
: "${E3_OUT:?separate helper output required}"
: "${E3_PRIMARY_OUT:?primary output required}"
: "${E3_MODEL:?f2 or r0 required}"
cd "$E3_ROOT"
mkdir -p "$E3_OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export BIRWKV_LOAD_SLOTS=8 BIRWKV_META_CONSTRUCT=0 PYTHONUNBUFFERED=1
helper_args=(--qualified-root "$E3_ROOT" --model "$E3_MODEL"
  --primary-out "$E3_PRIMARY_OUT" --out "$E3_OUT"
  --panel /inspire/hdd/global_user/zhangjiaquan-253108540222/capability_eval_data/lc_grid324
  --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF)
if [[ "$E3_MODEL" == f2 ]]; then
  : "${E3_PROTOCOL:?frozen execution protocol required}"
  helper_args+=(--protocol "$E3_PROTOCOL" --seed 17
    --checkpoint /inspire/hdd/global_user/zhangjiaquan-253108540222/research/lacesmm_assets/birwkv_f2_step14000)
elif [[ "$E3_MODEL" == r0 ]]; then
  : "${E3_QUALIFICATION:?causal dev qualification required}"
  helper_args+=(--qualification "$E3_QUALIFICATION")
else
  echo 'unsupported helper model' >&2
  exit 2
fi
exec torchrun --standalone --nproc_per_node=8 "$E3_HELPER_CODE" "${helper_args[@]}" > "$E3_OUT/torchrun.log" 2>&1
