#!/usr/bin/env bash
# launch_lm_eval.sh -- per-pod entrypoint for the BiRWKV LM-likelihood eval (phase 1).
#
# Scores causal next-token NLL/ppl and diffusion masked-CE (denoising pseudo-NLL)
# on pre-tokenized .npz corpora (wikitext103 / lambada / pg19) for one checkpoint.
# CKPT_DIR=base scores the frozen HF warm-start as the causal control arm.
#
# One pod = one multi-GPU node. Each (corpus, shard) writes a per-shard JSON to a
# per-corpus subdir of OUTDIR on GPFS; merge_eval folds each corpus separately so
# the three corpora never mix into one bootstrap group.
#
# Conventions mirror launch_capability_eval.sh (self-written BOOT log, GPU
# telemetry, PREFLIGHT/PROBE fail-closed, per-GPU logs, multinode-safe striding).
#
# Env (set at submit time via the CreateJob command):
#   CKPT_DIR   (required) step_*/ dir holding model.pt, OR the literal "base"
#   MODEL_DIR  (required) HF dir supplying geometry + tokenizer
#   LM_CORPORA (required) comma-sep "name:/abs/npz/dir" entries
#   OUTDIR     (required) GPFS dir for per-corpus shard + merged JSONs
#   RATIOS     (default 0.15,0.3,0.5,0.7,0.9) mask ratios for the diffusion-MC arms
#   MAX_SAMPLES (default unset) per-shard doc cap
#   BATCH      (default 16)
#   NUM_SHARDS (default NGPUS)
#   NGPUS      (default 8)   NNODES (default 1)
#   PROBE_ONLY (default 0)   PROBE_N (default 4)   SEED (default 42)
#   ALPHA_ONE  (default 0) set 1 to add a paired mc<R>_fwdonly arm per ratio with the
#              reverse stream disabled (algebraically alpha:=1.0). Fills the missing
#              (masked objective x forward-only stream) cell so the reverse stream can
#              be isolated; see v5.2 Phase 0a. Roughly doubles diffusion-arm runtime.
#
# Repo rule: local work is CPU-only; this script requires CUDA and runs ONLY
# inside an approved qz job.

set -uo pipefail

SCALE_DIR="${DAN_SCALE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NGPUS_HINT="${NGPUS:-8}"   # needed by the capacity sizing below (NGPUS default set later)

# --- Environment: mirror the training launcher (offline HF). ---
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Host-RAM / thread guards (2026-08-15: fixed the 8-way load hang). ---
# Each shard builds the ~4.1B denoiser in fp32 and then casts to bf16, and the
# ckpt arms additionally hold the fp32 model.pt state dict AND its bf16 copy:
# ~32.8 GB peak per process (ckpt) / ~24.6 GB (base warm-start). Eight at once is
# ~262 GB / ~197 GB against this pod's 300 GB cgroup — under the cap, but close
# enough that combined with 8 x 64 OMP threads (512-way oversubscription on 64
# cores) during the fp32 construction, all 8 shards wedged silently in load with
# 0% GPU util and no traceback. LOAD_SLOTS serializes the *load* phase (node-local
# flock ticket) so peak host RAM is bounded by LOAD_SLOTS x per-proc peak; the GPU
# compute that follows still runs fully in parallel.
# Sized from the pod's ACTUAL capacity, not a fixed guess: the H100 eval pods
# (64 CPU / 300 GB cgroup) and the H200 pods (192 CPU / 1930 GB) differ ~6x in RAM,
# and one hardcoded value either oversubscribes the small pod or starves the big one.
_CPUS="$(nproc 2>/dev/null || echo 8)"
_CGROUP_BYTES="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)"
case "$_CGROUP_BYTES" in (''|*[!0-9]*) _CGROUP_BYTES=0;; esac
# ~33 GB peak host RAM per concurrent load (measured after the in-place bf16 cast).
if [ "$_CGROUP_BYTES" -gt 0 ]; then
  _MEM_SLOTS=$(( _CGROUP_BYTES / 36000000000 ))   # 36 GB/slot leaves headroom
  [ "$_MEM_SLOTS" -lt 1 ] && _MEM_SLOTS=1
  [ "$_MEM_SLOTS" -gt "$NGPUS_HINT" ] && _MEM_SLOTS="$NGPUS_HINT"
else
  _MEM_SLOTS=3
fi
LOAD_SLOTS="${LOAD_SLOTS:-$_MEM_SLOTS}"
_THREADS=$(( _CPUS / NGPUS_HINT ))
[ "$_THREADS" -lt 1 ] && _THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$_THREADS}"

# --- Defaults (overridable at submit time). ---
NGPUS="${NGPUS:-8}"
NNODES="${NNODES:-1}"
NUM_SHARDS="${NUM_SHARDS:-$NGPUS}"
RATIOS="${RATIOS:-0.15,0.3,0.5,0.7,0.9}"
ALPHA_ONE="${ALPHA_ONE:-0}"
ALPHA_ONE_ARG=()
[[ "$ALPHA_ONE" == "1" ]] && ALPHA_ONE_ARG+=(--alpha-one-ablation)
BATCH="${BATCH:-16}"
SEED="${SEED:-42}"
PROBE_ONLY="${PROBE_ONLY:-0}"
PROBE_N="${PROBE_N:-4}"
WINDOW="${WINDOW:-0}"
PACK="${PACK:-0}"   # >0: concatenate consecutive chunks into PACK-token seqs (phase-2a long-context)
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.88}"  # target GPU-memory fraction when BATCH=0 (autotune)

: "${CKPT_DIR:?CKPT_DIR is required (step_*/ dir with model.pt, or 'base')}"
: "${MODEL_DIR:?MODEL_DIR is required (HF dir with geometry + tokenizer)}"
: "${LM_CORPORA:?LM_CORPORA is required (comma-sep name:/abs/npz/dir)}"
: "${OUTDIR:?OUTDIR is required (GPFS dir for shard + merged JSONs)}"

mkdir -p "$OUTDIR"
HOST="$(hostname)"
BOOTLOG="$OUTDIR/${HOST}.lm.boot.txt"
RUNLOG="$OUTDIR/${HOST}.lm.run.log"

# --- Self-written boot log (qz GetJobLog is unreliable; GPFS is source of truth). ---
{
  echo "==== LM-EVAL BOOT $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "hostname=$HOST"
  echo "CKPT_DIR=$CKPT_DIR  MODEL_DIR=$MODEL_DIR  OUTDIR=$OUTDIR"
  echo "LM_CORPORA=$LM_CORPORA  RATIOS=$RATIOS  MAX_SAMPLES=${MAX_SAMPLES:-<unset>}  PACK=$PACK"
  echo "ALPHA_ONE=$ALPHA_ONE (alpha:=1.0 paired ablation arm)"
  echo "NGPUS=$NGPUS  NNODES=$NNODES  NUM_SHARDS=$NUM_SHARDS  BATCH=$BATCH (0=autotune to $GPU_MEM_FRAC)"
  echo "LOAD_SLOTS=$LOAD_SLOTS  OMP_NUM_THREADS=$OMP_NUM_THREADS  cpus=$_CPUS  cgroup=$_CGROUP_BYTES (auto-sized)"
  echo "---- nvidia-smi -L ----"
  nvidia-smi -L 2>&1 || echo "<nvidia-smi -L failed>"
  echo "---- deps ----"
  python -c "import torch,transformers,fla;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'n_gpu',torch.cuda.device_count())" 2>&1 || echo "<dep import failed>"
  echo "---- ckpt ----"
  if [[ "$CKPT_DIR" == "base" ]]; then
    echo "base sentinel — will load frozen HF warm-start (causal-only control)"
  else
    ls -la "$CKPT_DIR/model.pt" 2>&1 || echo "<model.pt missing>"
  fi
  echo "==== END BOOT ===="
} > "$BOOTLOG" 2>&1
echo "[lm-eval] wrote boot log: $BOOTLOG"

cd "$SCALE_DIR"

# --- Telemetry sampler (mirrors launch_capability_eval.sh). ---
TL="${OUTDIR}/telemetry-${HOST}.lm"
python "$SCALE_DIR/qz/gpu_telemetry.py" collect --output-dir "$TL" &
telemetry_pid=$!
cleanup() {
  status=$?
  kill "$telemetry_pid" >/dev/null 2>&1 || true
  wait "$telemetry_pid" >/dev/null 2>&1 || true
  if [[ -n "${watchdog_pid:-}" ]]; then
    kill "$watchdog_pid" >/dev/null 2>&1 || true
    wait "$watchdog_pid" >/dev/null 2>&1 || true
  fi
  [[ -n "${LM_LOAD_LOCK_DIR:-}" ]] && rm -rf "$LM_LOAD_LOCK_DIR" 2>/dev/null || true
  python "$SCALE_DIR/qz/gpu_telemetry.py" summarize \
    --telemetry-csv "$TL/gpu-telemetry.csv" \
    --processes-csv "$TL/gpu-processes.csv" \
    --output "$TL/utilization-summary.json" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

# --- PREFLIGHT: validate every flag this script passes against lm_eval's own
# argparse, before a single GPU second is spent (fail-closed on a flag typo). ---
echo "[lm-eval] PREFLIGHT: checking argv against lm_eval argparse"
preflight_err="$(python - "$SCALE_DIR" <<'PYEOF' 2>&1
import ast, pathlib, re, sys
scale = pathlib.Path(sys.argv[1])
launcher = scale / "qz" / "launch_lm_eval.sh"
runner = scale / "eval" / "capability" / "lm_eval.py"
accepted = set()
for node in ast.walk(ast.parse(runner.read_text())):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"):
        for a in node.args:
            if isinstance(a, ast.Constant) and str(a.value).startswith("-"):
                accepted.add(a.value)
code_lines = [ln for ln in launcher.read_text().splitlines()
              if not ln.lstrip().startswith("#")]
used = {m for m in re.findall(r"(?<![\w-])--[A-Za-z0-9][\w-]*", "\n".join(code_lines))}
merge_only = {"--shards-glob", "--output-dir", "--bootstrap-samples",
              "--telemetry-csv", "--processes-csv", "--help",
              # not lm_eval flags: merge_eval / gpu_telemetry / py-spy (watchdog)
              "--pid"}
bad = sorted(f for f in used - merge_only if f not in accepted)
if bad:
    print("unknown lm_eval flags in launcher: " + " ".join(bad))
PYEOF
)"
if [[ -n "$preflight_err" ]]; then
  echo "[lm-eval] PREFLIGHT FAILED: $preflight_err" | tee -a "$RUNLOG"
  exit 3
fi
echo "[lm-eval] PREFLIGHT OK." | tee -a "$RUNLOG"

# --- PROBE: fail-closed smoke on the first corpus before the full sweep. ---
first_spec="${LM_CORPORA%%,*}"
probe_name="${first_spec%%:*}"
probe_dir="${first_spec#*:}"
probe_out="$OUTDIR/${probe_name}/lm.${probe_name}.probe.json"
echo "[lm-eval] PROBE ($PROBE_N samples, corpus=$probe_name) -> $probe_out"
MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then MAX_SAMPLES_ARG+=(--max_samples "$MAX_SAMPLES"); fi
python -m eval.capability.lm_eval \
  --ckpt_dir "$CKPT_DIR" --model_dir "$MODEL_DIR" \
  --corpus "$probe_name" --corpus_dir "$probe_dir" \
  --ratios "$RATIOS" --batch "$BATCH" --gpu_mem_frac "$GPU_MEM_FRAC" --seed "$SEED" --window "$WINDOW" --pack "$PACK" \
  "${ALPHA_ONE_ARG[@]}" \
  --shard 0 --num_shards 1 \
  --max_samples "$PROBE_N" --output "$probe_out" 2>&1 | tee -a "$RUNLOG"
probe_rc="${PIPESTATUS[0]}"
if [[ "$probe_rc" -ne 0 ]]; then
  echo "[lm-eval] PROBE FAILED rc=$probe_rc — aborting sweep (fail-closed)." | tee -a "$RUNLOG"
  exit "$probe_rc"
fi
echo "[lm-eval] PROBE OK." | tee -a "$RUNLOG"

if [[ "$PROBE_ONLY" == "1" ]]; then
  echo "[lm-eval] PROBE_ONLY=1 — stopping after probe." | tee -a "$RUNLOG"
  exit 0
fi

# --- Node rank from hostname (multinode-safe; NNODES=1 reproduces single pod). ---
if [[ "$NNODES" -gt 1 ]]; then
  NODE_RANK="${NODE_RANK:-$(echo "$HOST" | grep -oE '[0-9]+$' || echo 0)}"
else
  NODE_RANK=0
fi
WORLD_GPUS=$((NNODES * NGPUS))
echo "[lm-eval] sweep topology: NNODES=$NNODES NODE_RANK=$NODE_RANK NGPUS=$NGPUS (world $WORLD_GPUS, $NUM_SHARDS shards)" | tee -a "$RUNLOG"

rc=0
# Load-gate lock dir: node-local (/dev/shm), NEVER GPFS — a shared-FS lock would
# serialize loads across every pod in the job instead of within this one.
export LM_LOAD_SLOTS="$LOAD_SLOTS"
export LM_LOAD_LOCK_DIR="${LM_LOAD_LOCK_DIR:-/dev/shm/lm_eval_locks.$$}"
mkdir -p "$LM_LOAD_LOCK_DIR"
echo "[lm-eval] load gate: LM_LOAD_SLOTS=$LM_LOAD_SLOTS dir=$LM_LOAD_LOCK_DIR" | tee -a "$RUNLOG"

# --- Progress watchdog: the 2026-08-15 hang produced 0 bytes of new output for
# ~55 min with no traceback. Fail loudly instead: if no shard JSON appears and no
# per-GPU log grows for WATCHDOG_STALL_SEC, dump diagnostics to the run log so the
# next run has evidence rather than silence.
WATCHDOG_STALL_SEC="${WATCHDOG_STALL_SEC:-1800}"
(
  last_sig=""; last_change=$(date +%s)
  while true; do
    sleep 120
    sig="$(find "$OUTDIR" -type f \( -name '*.json' -o -name '*.log' \) -printf '%s' 2>/dev/null | md5sum)"
    now=$(date +%s)
    if [[ "$sig" != "$last_sig" ]]; then last_sig="$sig"; last_change=$now; continue; fi
    if (( now - last_change >= WATCHDOG_STALL_SEC )); then
      {
        echo "==== WATCHDOG: no output growth for $((now - last_change))s ===="
        date -u +%Y-%m-%dT%H:%M:%SZ
        echo "---- nvidia-smi ----"; nvidia-smi 2>&1 | head -25
        echo "---- host mem ----"; free -g 2>&1 | head -3
        echo "---- python procs ----"; ps -eo pid,rss,etime,stat,cmd 2>/dev/null | grep -E "lm_eval|[p]ython" | head -15
        echo "---- py-spy dump (if available) ----"
        for p in $(pgrep -f eval.capability.lm_eval 2>/dev/null | head -3); do
          py-spy dump --pid "$p" 2>&1 | head -25 || echo "<py-spy unavailable for $p>"
        done
        echo "==== END WATCHDOG ===="
      } >> "$RUNLOG" 2>&1
      last_change=$now  # re-arm so it reports again if the stall persists
    fi
  done
) &
watchdog_pid=$!
# --- Sweep: one corpus at a time (each a full multi-GPU shard fan-out). ---
IFS=',' read -ra CORPUS_SPECS <<< "$LM_CORPORA"
unset IFS  # restore default word-splitting; a leaked IFS=',' would collapse `for g in $(seq ...)`
for spec in "${CORPUS_SPECS[@]}"; do
  [ -z "$spec" ] && continue
  cname="${spec%%:*}"
  cdir="${spec#*:}"
  if [ -z "$cname" ] || [ -z "$cdir" ]; then
    echo "[lm-eval] bad corpus spec '$spec' (want name:/abs/dir)" | tee -a "$RUNLOG"
    rc=1; continue
  fi
  mkdir -p "$OUTDIR/$cname"
  # --- Stale-shard guard: shard JSONs are globbed by name, so leftovers from an
  # EARLIER job in this OUTDIR would be merged together with this run's output —
  # silently mixing results from different code versions (this bit twice on
  # 2026-08-15: once as a phantom "hang", once as v2-vs-v3 scorer contamination).
  # Anything older than this pod's boot log is quarantined, never merged.
  stale=0
  for old in "$OUTDIR/$cname"/lm.${cname}.shard*of${NUM_SHARDS}.json; do
    [ -f "$old" ] || continue
    if [[ "$old" -ot "$BOOTLOG" ]]; then
      mkdir -p "$OUTDIR/$cname/stale_preboot"
      mv "$old" "$OUTDIR/$cname/stale_preboot/" 2>/dev/null && stale=$((stale+1))
    fi
  done
  if [[ "$stale" -gt 0 ]]; then
    echo "[lm-eval] quarantined $stale pre-boot shard JSON(s) for $cname -> $cname/stale_preboot/" | tee -a "$RUNLOG"
  fi
  echo "[lm-eval] corpus=$cname dir=$cdir" | tee -a "$RUNLOG"

  declare -a pids=()
  for g in $(seq 0 $((NGPUS - 1))); do
    ( export CUDA_VISIBLE_DEVICES=$g
      w=$((NODE_RANK * NGPUS + g))
      SHARDLOG="$OUTDIR/$cname/${HOST}.${cname}.gpu${g}.log"
      for (( s=w; s<NUM_SHARDS; s+=WORLD_GPUS )); do
        out="$OUTDIR/$cname/lm.${cname}.shard${s}of${NUM_SHARDS}.json"
        MAX_SAMPLES_ARG=()
        if [[ -n "${MAX_SAMPLES:-}" ]]; then MAX_SAMPLES_ARG+=(--max_samples "$MAX_SAMPLES"); fi
        python -m eval.capability.lm_eval \
          --ckpt_dir "$CKPT_DIR" --model_dir "$MODEL_DIR" \
          --corpus "$cname" --corpus_dir "$cdir" \
          --ratios "$RATIOS" --batch "$BATCH" --gpu_mem_frac "$GPU_MEM_FRAC" --seed "$SEED" --window "$WINDOW" --pack "$PACK" \
          "${ALPHA_ONE_ARG[@]}" \
          --shard "$s" --num_shards "$NUM_SHARDS" \
          "${MAX_SAMPLES_ARG[@]}" \
          --output "$out" 2>&1 | tee -a "$SHARDLOG"
      done
    ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
  done

  # --- MERGE this corpus (only when this pod sees all shards). ---
  shard_count="$(ls "$OUTDIR/$cname"/lm.${cname}.shard*of${NUM_SHARDS}.json 2>/dev/null | wc -l)"
  if [[ "$shard_count" -ge "$NUM_SHARDS" && "$rc" -eq 0 ]]; then
    python -m eval.capability.merge_eval \
      --shards-glob "$OUTDIR/$cname/lm.${cname}.shard*of${NUM_SHARDS}.json" \
      --output-dir "$OUTDIR/$cname/merged" --bootstrap-samples 1000 2>&1 | tee -a "$RUNLOG" || true
  else
    echo "[lm-eval] merge skipped for $cname on $HOST: $shard_count/$NUM_SHARDS shards (rc=$rc)" | tee -a "$RUNLOG"
  fi
done

echo "[lm-eval] sweep exited rc=$rc (run log: $RUNLOG)" | tee -a "$RUNLOG"
exit "$rc"
