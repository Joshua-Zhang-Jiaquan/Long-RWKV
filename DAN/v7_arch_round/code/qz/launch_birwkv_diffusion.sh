#!/usr/bin/env bash
# launch_birwkv_diffusion.sh -- token-level BiRWKV masked-diffusion training/parity.
#
# Modes (MODE env):
#   parity  - run scale/train/parity_birwkv_warmstart.py (1 GPU, exits 0/1)
#   smoke   - 20-step single-node training smoke (checkpoint + resume check)
#   train   - full run (NNODES>1 uses GPFS rendezvous, else torchrun --standalone)
#
# Env:
#   MODE (parity|smoke|train)  MODEL_DIR  TOKEN_DIR  SAVE_ROOT  RUN_NAME
#   NNODES (default 1)  NGPUS (default 8)  MICROBATCH  GRAD_ACCUM  STEPS
#   EXTRA_ARGS (e.g. "--force-forward" for the causal control arm)
#   TOKEN (rdzv token, multi-node only)  LOGDIR

set -uo pipefail

SCALE_DIR="${DAN_SCALE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RDZV_SCRIPT="$SCALE_DIR/qz/rendezvous_gpfs.sh"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${MODE:-smoke}"
NNODES="${NNODES:-1}"
NGPUS="${NGPUS:-8}"
MODEL_DIR="${MODEL_DIR:-/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/base_models/rwkv7-0.4B-world}"
TOKEN_DIR="${TOKEN_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/preprocessed_data/fineweb_4096_packed_full}"
SAVE_ROOT="${SAVE_ROOT:-/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion}"
RUN_NAME="${RUN_NAME:-birwkv-diff-dev}"
MICROBATCH="${MICROBATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
STEPS="${STEPS:-4000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOGDIR="${LOGDIR:-$SAVE_ROOT/logs}"
HOST="$(hostname)"
mkdir -p "$LOGDIR" "$SAVE_ROOT"
# NOTE (2026-08-16): keying these on $HOST.$MODE means two arms launched
# sequentially in ONE job silently truncate each other's logs -- observed in
# job-02ed602a, where the conditioned arm's banner erased the baseline arm's
# entire step history. The fix is NOT to change these paths while a
# fault-tolerant job is in flight: round-4b (job-aa61cc21) has
# auto_fault_tolerance with 2 retries, and a retry would re-exec this script and
# start writing to a different filename, orphaning its monitors mid-run.
# Instead, callers running multiple arms in one job must pass a distinct LOGDIR
# per arm (see birwkv_m0_latent_smoke_job.json). Revisit renaming these once no
# long fault-tolerant run is active.
BOOTLOG="$LOGDIR/${HOST}.${MODE}.boot.txt"
RUNLOG="$LOGDIR/${HOST}.${MODE}.run.log"

{
  echo "==== BOOT(birwkv-diffusion:$MODE) $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "host=$HOST NNODES=$NNODES NGPUS=$NGPUS MODEL_DIR=$MODEL_DIR"
  echo "TOKEN_DIR=$TOKEN_DIR SAVE_ROOT=$SAVE_ROOT RUN_NAME=$RUN_NAME"
  echo "MICROBATCH=$MICROBATCH GRAD_ACCUM=$GRAD_ACCUM STEPS=$STEPS EXTRA_ARGS=$EXTRA_ARGS"
  nvidia-smi -L 2>&1 || echo "<nvidia-smi failed>"
  echo "---- PREFLIGHT: fla kernel import ----"
  python -c "import torch; from fla.ops.rwkv7 import chunk_rwkv7; from fla.layers.rwkv7 import RWKV7Attention; print('PREFLIGHT OK: torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
  echo "---- PREFLIGHT: model + trainer import ----"
  python -c "
import sys; sys.path.insert(0, '$SCALE_DIR')
from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion, MASK_TOKEN_ID
print('PREFLIGHT OK: birwkv7_diffusion importable, mask_id', MASK_TOKEN_ID)" 2>&1
  echo "---- PREFLIGHT: TRAINER imports (catches missing staged modules) ----"
  # Importing the model alone is not enough. The two trees have diverged badly --
  # the repo tree carries 59 models/*.py that staging does not -- and TRAINING RUNS
  # FROM STAGING. A module the trainer imports but nobody staged (this happened with
  # models/state_hijacking_cache.py) otherwise fails only after the 4B load and FSDP
  # init. Importing the trainer module itself walks its whole import graph, so any
  # such gap surfaces here, in seconds, before a single GPU is touched.
  python -c "
import sys; sys.path.insert(0, '$SCALE_DIR')
import importlib
importlib.import_module('train.train_birwkv_diffusion')
print('PREFLIGHT OK: trainer module and its full import graph resolve')" 2>&1 \
    || echo "PREFLIGHT FAIL: trainer import failed (a module is probably unstaged)"
  echo "---- PREFLIGHT: EXTRA_ARGS are all real trainer flags ----"
  # A misspelled or removed flag in EXTRA_ARGS otherwise surfaces only AFTER the
  # 4B model has loaded and FSDP has initialised -- minutes of 8xH200 wasted, and
  # in the worst case the run proceeds with the flag silently absent. AST-parse the
  # trainer's own argparse calls and reject any --flag it does not define.
  python - "$SCALE_DIR" $EXTRA_ARGS <<'PYFLAGCHK' 2>&1
import ast, sys
scale, args = sys.argv[1], sys.argv[2:]
tree = ast.parse(open(f"{scale}/train/train_birwkv_diffusion.py").read())
known = {
    a.value
    for n in ast.walk(tree)
    if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
    for a in n.args
    if isinstance(a, ast.Constant) and isinstance(a.value, str)
}
bad = [a for a in args if a.startswith("--") and a not in known]
if bad:
    print(f"PREFLIGHT FAIL: unknown trainer flag(s) {bad}")
else:
    flags = [a for a in args if a.startswith("--")]
    print(f"PREFLIGHT OK: {len(flags)} EXTRA_ARGS flags all defined in argparse")
PYFLAGCHK
  FIRST_TOKEN_DIR="${TOKEN_DIR%%,*}"
  ls "$FIRST_TOKEN_DIR"/*.pkl >/dev/null 2>&1 && echo "PREFLIGHT OK: pkl shards present" || echo "PREFLIGHT FAIL: no pkl shards in $FIRST_TOKEN_DIR"
  echo "==== END BOOT ===="
} > "$BOOTLOG" 2>&1
cat "$BOOTLOG"
grep -q "PREFLIGHT FAIL" "$BOOTLOG" && { echo "[launch] FATAL: preflight failed"; exit 2; }

cd "$SCALE_DIR"

case "$MODE" in
  parity)
    python train/parity_birwkv_warmstart.py --model-dir "$MODEL_DIR" 2>&1 | tee "$RUNLOG"
    exit "${PIPESTATUS[0]}"
    ;;
  smoke)
    torchrun --standalone --nproc_per_node="$NGPUS" train/train_birwkv_diffusion.py \
      --model-dir "$MODEL_DIR" --token-dir "$TOKEN_DIR" \
      --save-root "$SAVE_ROOT" --run-name "${RUN_NAME}-smoke" \
      --steps 20 --microbatch "$MICROBATCH" --grad-accum 1 \
      --save-every 10 --log-every 5 --val-every 20 --sampler-every 20 \
      --val-samples 32 --max-samples 2000 $EXTRA_ARGS 2>&1 | tee "$RUNLOG"
    exit "${PIPESTATUS[0]}"
    ;;
  train)
    TRAIN_ARGS=(train/train_birwkv_diffusion.py
      --model-dir "$MODEL_DIR" --token-dir "$TOKEN_DIR"
      --save-root "$SAVE_ROOT" --run-name "$RUN_NAME"
      --steps "$STEPS" --microbatch "$MICROBATCH" --grad-accum "$GRAD_ACCUM")
    if [ "$NNODES" -gt 1 ]; then
      TOKEN="${TOKEN:?TOKEN required for multi-node}"
      RDZV_BASE="${RDZV_BASE:-/inspire/hdd/global_user/zhangjiaquan-253108540222/rdzv}"
      export RDZV_DIR="$RDZV_BASE/rdzv_$TOKEN" RDZV_RUN_ID="$TOKEN" NNODES NGPUS
      mkdir -p "$RDZV_DIR"
      find "$RDZV_DIR" -maxdepth 1 \( -name "host_*" -o -name "name_*" \) -mmin +20 -delete 2>/dev/null || true
      bash "$RDZV_SCRIPT" "${TRAIN_ARGS[@]}" $EXTRA_ARGS 2>&1 | tee "$RUNLOG"
    else
      torchrun --standalone --nproc_per_node="$NGPUS" "${TRAIN_ARGS[@]}" $EXTRA_ARGS 2>&1 | tee "$RUNLOG"
    fi
    exit "${PIPESTATUS[0]}"
    ;;
  *)
    echo "[launch] FATAL: unknown MODE=$MODE"; exit 2 ;;
esac
