#!/usr/bin/env bash
# launch_sampler_eval.sh -- per-pod entrypoint for the Gate-0 iterative-gain check.
#
# Settles the OPEN Gate-0 item in v4_plan section 1.2: during round-4b the
# in-training probe showed `em@32 - em@1` NEGATIVE for steps 500-3000 (as low as
# -0.0170), recovering to +0.0047 only in the final 500 steps. That inverts the
# property section 1.1 certifies ("iterative beats single-shot with disjoint
# CIs"). The in-training probe cannot settle it: it runs at ONE mask ratio, on the
# TRAINING corpus, with a handful of sequences and no CIs.
#
# `eval/capability/sampler_eval_offline.py` has existed for a while but had NO
# launcher, which is precisely why this check was never run. It sweeps
# mask_ratio x {1,8,16,32,64} denoise steps and emits
# qz_capability_sampler_shard_v1 records so merge_eval produces bootstrap CIs per
# arm.
#
# ---- THE HOLDOUT TRAP (read before changing PER_DIR) ----
# The trainer holds out exactly `--val-samples` (default 256) sequences from the
# TAIL of the FIRST token dir:
#     val_ds = Subset(probe, range(first_total - 256, first_total))
# and trains on everything before that. sampler_eval_offline's `_load_tail_sequences`
# also reads from the tail, but its default `--per_dir` is 512 -- so HALF of its
# sequences were trained on. A gate that exists to measure held-out behaviour must
# not be fed training data, so PER_DIR defaults to 256 here and the script REFUSES
# to run above that unless FORCE_PER_DIR=1 is set explicitly.
#
# Env:
#   CKPT_DIR   (required) step_*/ dir holding model.pt
#   MODEL_DIR  (required) HF dir supplying geometry + tokenizer
#   TOKEN_DIRS (required) comma-separated packed-pkl dirs; the FIRST one must be
#              the dir the run trained on (that is the one with a real holdout)
#   OUTDIR     (required) GPFS dir for per-shard + merged JSON
#   PER_DIR    (default 256) tail sequences per dir -- see the holdout trap above
#   WINDOW     (default 1024)  BATCH (default 8)
#   SELF_CORRECTION (default 0) set 1 to add the sc arms
#   COMMIT_GROUPS (default 0) comma-separated commit_group_size sweep (v5.2 B2/Q1-U2).
#              0 = legacy single-forward multi-commit; g>=1 commits <=g positions per
#              forward and refreshes the canvas between groups. Arms gain a _g<G> suffix,
#              and g=0 keeps the LEGACY arm name so old merges stay comparable.
#              Cost note: g=1 runs ~one forward per committed token, so a full ratio x
#              step grid at g=1 is far more expensive than the legacy sweep.
#   NGPUS (default 8)  NNODES (default 1)  NODE_RANK (default 0)
#   NUM_SHARDS (default NGPUS*NNODES)  SEED (default 42)
#
# Repo rule: local work is CPU-only; this requires CUDA and runs ONLY inside an
# approved qz job.

set -uo pipefail

SCALE_DIR="${DAN_SCALE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SCALE_DIR" || exit 4

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${CKPT_DIR:?CKPT_DIR required}"
: "${MODEL_DIR:?MODEL_DIR required}"
: "${TOKEN_DIRS:?TOKEN_DIRS required}"
: "${OUTDIR:?OUTDIR required}"
PER_DIR="${PER_DIR:-256}"
WINDOW="${WINDOW:-1024}"
BATCH="${BATCH:-8}"
SELF_CORRECTION="${SELF_CORRECTION:-0}"
COMMIT_GROUPS="${COMMIT_GROUPS:-0}"
STEPS_GRID="${STEPS_GRID:-}"
# Empty = the harness default grid (0.3/0.5/0.7/0.9). Set e.g. "0.95,1.0" to
# measure the FREE-GENERATION regime, which the default grid never reaches.
MASK_RATIOS="${MASK_RATIOS:-}"
NGPUS="${NGPUS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_SHARDS="${NUM_SHARDS:-$((NGPUS * NNODES))}"
WORLD_GPUS=$((NGPUS * NNODES))
SEED="${SEED:-42}"
HOST="$(hostname)"

# Thread caps: 8 concurrent 4B loads oversubscribed a pod to 512 threads and wedged
# for 55 minutes (memory: birwkv-8way-load-hang-fix). Size from the cgroup.
CPUS="$(nproc 2>/dev/null || echo 32)"
THREADS=$(( CPUS / NGPUS )); [[ "$THREADS" -lt 1 ]] && THREADS=1
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTDIR"
BOOTLOG="$OUTDIR/${HOST}.sampler.boot.txt"
RUNLOG="$OUTDIR/${HOST}.sampler.run.log"

{
  echo "==== BOOT(sampler-eval) $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "host=$HOST NNODES=$NNODES NGPUS=$NGPUS NUM_SHARDS=$NUM_SHARDS NODE_RANK=$NODE_RANK"
  echo "CKPT_DIR=$CKPT_DIR"
  echo "MODEL_DIR=$MODEL_DIR"
  echo "TOKEN_DIRS=$TOKEN_DIRS OUTDIR=$OUTDIR"
  echo "PER_DIR=$PER_DIR WINDOW=$WINDOW BATCH=$BATCH SELF_CORRECTION=$SELF_CORRECTION SEED=$SEED"
  echo "COMMIT_GROUPS=$COMMIT_GROUPS (v5.2 B2 grouped sequential commit; 0=legacy)"
  echo "STEPS_GRID=${STEPS_GRID:-<harness default 1,8,16,32,64>}"
  echo "MASK_RATIOS=${MASK_RATIOS:-<harness default 0.3,0.5,0.7,0.9>}"
  echo "OMP_NUM_THREADS=$THREADS (cpus=$CPUS)"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | head -10
} > "$BOOTLOG" 2>&1

# --- PREFLIGHT: fail closed on the ways this number can be meaningless. ---
{
  echo "---- PREFLIGHT: holdout size ----"
  # The whole point of this gate is held-out data. See the holdout trap above.
  if [[ "$PER_DIR" -gt 256 && "${FORCE_PER_DIR:-0}" != "1" ]]; then
    echo "PREFLIGHT FAIL: PER_DIR=$PER_DIR exceeds the trainer's 256-sequence holdout."
    echo "  The trainer keeps only range(n-256, n) out of training, so sequences"
    echo "  beyond that were TRAINED ON and this would not be a held-out measurement."
    echo "  Set PER_DIR<=256, or FORCE_PER_DIR=1 if you are deliberately measuring"
    echo "  train-set behaviour and will say so in the writeup."
    exit 3
  fi
  echo "PREFLIGHT OK: PER_DIR=$PER_DIR within the 256-sequence holdout"

  echo "---- PREFLIGHT: checkpoint present ----"
  if [[ ! -f "$CKPT_DIR/model.pt" ]]; then
    echo "PREFLIGHT FAIL: $CKPT_DIR/model.pt missing"
    exit 3
  fi
  echo "PREFLIGHT OK: model.pt present"

  echo "---- PREFLIGHT: imports via the REAL invocation path ----"
  # Must be `python -m` from cwd=$SCALE_DIR, exactly as the shards run. An earlier
  # launcher imported via a hand-built sys.path and gave a FALSE PASS, missing that
  # a local module shadowed stdlib `code` and broke torch's own import.
  python -m eval.capability.sampler_eval_offline --help >/dev/null || exit 3
  # Assert on the EFFECTIVE grid, not the module constant. Checking STEP_GRID alone would
  # pass no matter what STEPS_GRID overrides it to -- a preflight that cannot fail is not a
  # preflight. Same class of defect as an env var absent from the boot log.
  STEPS_GRID="$STEPS_GRID" MASK_RATIOS="$MASK_RATIOS" python -c "
import os
from eval.capability.sampler_eval_offline import MASK_RATIOS, STEP_GRID
import code as _c, math as _m
assert '/usr/lib' in _c.__file__, f'stdlib code shadowed by {_c.__file__}'
ov = os.environ.get('STEPS_GRID', '').strip()
eff = tuple(int(x) for x in ov.split(',') if x.strip()) if ov else STEP_GRID
assert eff, 'effective step grid is empty'
assert min(eff) >= 1, f'step counts must be >= 1, got {eff}'
assert 1 in eff, f'effective grid {eff} must contain 1 -- em@1 is the single-shot baseline the gate compares against'
assert max(eff) >= 32, f'effective grid {eff} must reach 32 to match the section 1.1 claim'
rov = os.environ.get('MASK_RATIOS', '').strip()
reff = tuple(float(x) for x in rov.split(',') if x.strip()) if rov else MASK_RATIOS
assert reff, 'effective ratio grid is empty'
assert all(0.0 < r <= 1.0 for r in reff), f'ratios must be in (0,1], got {reff}'
print('PREFLIGHT OK: effective ratios', reff, '(override)' if rov else '(harness default)',
      '| effective steps', eff, '(override)' if ov else '(harness default)')
" || exit 3
} >> "$BOOTLOG" 2>&1
tail -6 "$BOOTLOG"
grep -q "PREFLIGHT FAIL" "$BOOTLOG" && { echo "[sampler] PREFLIGHT FAILED, see $BOOTLOG"; exit 3; }
echo "==== END BOOT ====" >> "$BOOTLOG"

SC_ARG=()
[[ "$SELF_CORRECTION" == "1" ]] && SC_ARG+=(--self_correction)
RATIO_ARG=()
[[ -n "$MASK_RATIOS" ]] && RATIO_ARG+=(--mask_ratios "$MASK_RATIOS")
GROUP_ARG=()
[[ -n "$COMMIT_GROUPS" ]] && GROUP_ARG+=(--commit_groups "$COMMIT_GROUPS")
[[ -n "$STEPS_GRID" ]] && GROUP_ARG+=(--steps_grid "$STEPS_GRID")

# Quarantine shard JSONs older than this pod's boot log: a reused OUTDIR would
# otherwise merge results across code versions (learned 2026-08-15).
stale=0
for old in "$OUTDIR"/sampler.shard*.json; do
  [[ -e "$old" ]] || continue
  if [[ "$old" -ot "$BOOTLOG" ]]; then
    mkdir -p "$OUTDIR/stale_preboot"
    mv "$old" "$OUTDIR/stale_preboot/" 2>/dev/null && stale=$((stale+1))
  fi
done
[[ "$stale" -gt 0 ]] && echo "[sampler] quarantined $stale pre-boot shard JSON(s)" | tee -a "$RUNLOG"

rc=0
declare -a pids=()
for g in $(seq 0 $((NGPUS - 1))); do
  ( export CUDA_VISIBLE_DEVICES=$g
    w=$((NODE_RANK * NGPUS + g))
    SHARDLOG="$OUTDIR/${HOST}.gpu${g}.log"
    for (( s=w; s<NUM_SHARDS; s+=WORLD_GPUS )); do
      out="$OUTDIR/sampler.shard${s}of${NUM_SHARDS}.json"
      python -m eval.capability.sampler_eval_offline \
        --ckpt_dir "$CKPT_DIR" --model_dir "$MODEL_DIR" \
        --token_dirs "$TOKEN_DIRS" --per_dir "$PER_DIR" \
        --window "$WINDOW" --batch "$BATCH" --seed "$SEED" \
        "${SC_ARG[@]}" "${RATIO_ARG[@]}" "${GROUP_ARG[@]}" \
        --shard "$s" --num_shards "$NUM_SHARDS" \
        --output "$out" 2>&1 | tee -a "$SHARDLOG"
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
echo "[sampler] STAGE=shards_complete rc=$rc" | tee -a "$RUNLOG"

# STAGE breadcrumbs to GPFS. The phase1a run (job-5eb82a49) wrote all 8 shards with
# 576 records each and then exited without ever creating RUNLOG, so the script died
# between the shard loop and here -- and `qz GetJobLog` returns InternalError, leaving
# no way to see the launcher's own stderr. Four shell hypotheses were falsified by
# direct test, so the next occurrence must be diagnosable from the output dir alone.
shard_count="$(ls "$OUTDIR"/sampler.shard*of${NUM_SHARDS}.json 2>/dev/null | wc -l)" || shard_count=0
echo "[sampler] STAGE=counted shard_count=$shard_count/$NUM_SHARDS rc=$rc" | tee -a "$RUNLOG"
if [[ "$shard_count" -lt "$NUM_SHARDS" ]]; then
  echo "[sampler] MERGE SKIPPED: only $shard_count/$NUM_SHARDS shards" | tee -a "$RUNLOG"
elif [[ "$rc" -ne 0 ]]; then
  # rc != 0 with a full shard set means a shard process failed AFTER writing, or the
  # wait captured a signal. The shards may still be complete and mergeable, so say so
  # explicitly rather than leaving a silent gap: a human can merge by hand.
  echo "[sampler] MERGE SKIPPED on rc=$rc although all $NUM_SHARDS shards exist." \
       "Inspect then merge manually: python -m eval.capability.merge_eval" \
       "--shards-glob '$OUTDIR/sampler.shard*of${NUM_SHARDS}.json'" \
       "--output-dir '$OUTDIR/merged' --bootstrap-samples 1000" | tee -a "$RUNLOG"
fi
if [[ "$shard_count" -ge "$NUM_SHARDS" && "$rc" -eq 0 ]]; then
  # NOT `|| true`. The Gate-0 claim is about DISJOINT CIs, so a silent merge
  # failure would leave shard-local point estimates looking like a verdict. That
  # exact failure already happened once (d0_eval wrote "shard" where merge_eval
  # requires "shard_index": every shard was filtered out and the job exited 0).
  if ! python -m eval.capability.merge_eval \
    --shards-glob "$OUTDIR/sampler.shard*of${NUM_SHARDS}.json" \
    --output-dir "$OUTDIR/merged" --bootstrap-samples 1000 2>&1 | tee -a "$RUNLOG"
  then
    echo "[sampler] MERGE FAILED -- shard-local numbers are NOT a verdict" | tee -a "$RUNLOG"
    rc=5
  fi
  if ! ls "$OUTDIR/merged"/*.json >/dev/null 2>&1; then
    echo "[sampler] MERGE produced no output -- refusing to imply CIs exist" | tee -a "$RUNLOG"
    rc=5
  fi
  # Record-count audit. merge_eval keys on document_id, so DUPLICATE ids across
  # shards are silently DEDUPLICATED -- not flagged. That exact bug shipped here
  # once: sampler_eval_offline labelled rows with the shard-LOCAL batch offset, so
  # all 8 shards emitted ":0".. ":31" for different sequences and the merge kept
  # 640 of 5120 records while exiting 0. A missing-shard check cannot catch it
  # (all 8 shards were present); only comparing counts can.
  python - "$OUTDIR" "$NUM_SHARDS" <<'PYAUDIT' 2>&1 | tee -a "$RUNLOG"
import glob, json, sys
outdir, nshards = sys.argv[1], sys.argv[2]
shard_total = 0
for f in glob.glob(f"{outdir}/sampler.shard*of{nshards}.json"):
    shard_total += len(json.load(open(f)).get("records", []))
merged = glob.glob(f"{outdir}/merged/*.json")
if not merged:
    print("[sampler] AUDIT SKIP: no merged file"); sys.exit(0)
m = json.load(open(merged[0]))
kept = m.get("record_count", len(m.get("records", [])))
print(f"[sampler] AUDIT shard_records={shard_total} merged_records={kept}")
if shard_total and kept < shard_total:
    lost = 100.0 * (1 - kept / shard_total)
    print(f"[sampler] AUDIT FAIL: merge kept {kept}/{shard_total} ({lost:.1f}% lost) "
          "-- almost certainly duplicate document_id across shards")
    sys.exit(7)
print("[sampler] AUDIT OK: no records lost in merge")
PYAUDIT
  audit_rc=${PIPESTATUS[0]}
  if [[ "$audit_rc" -ne 0 ]]; then
    echo "[sampler] record-count audit failed (rc=$audit_rc) -- numbers are NOT a verdict" | tee -a "$RUNLOG"
    rc=7
  fi
else
  echo "[sampler] merge skipped on $HOST: $shard_count/$NUM_SHARDS shards (rc=$rc)" | tee -a "$RUNLOG"
fi

echo "[sampler] SAMPLER_EVAL_DONE rc=$rc" | tee -a "$RUNLOG"
exit "$rc"
