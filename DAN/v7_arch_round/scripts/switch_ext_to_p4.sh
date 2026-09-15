#!/usr/bin/env bash
# Launch the loop extension on the PRIORITY-4 high lane (user directive).
# The prio-3 attempt (job-21820297) went job_failed — killed by low-lane
# preemption ~18 min in (NCCL up, first forward, then silent stop; the same
# signature as mHC r1). It is terminal, so no stop is needed.
# Shape: 2x8 H100 + mb4 x ga4 = 256 rows/step = exact N2 batch semantics
# (mHC 16gpu_hp precedent; the high lane caps at 16 running GPU).
# Idempotent: guarded by the sentinel; safe to re-run on submission failure.
set -euo pipefail

P=/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/DAN/v7_arch_round
SPEC=$P/specs/m4_loop_ext_9500_p4_job.json
SENT=$P/specs/.ext_submitted_job_id

# refuse to double-submit the p4 shape
if [ -f "$SENT" ]; then
  PREV=$(cat "$SENT")
  if [ "$PREV" != "job-21820297-02f9-4f0e-b79a-094e4f4724a9" ]; then
    echo "already submitted: $PREV — nothing to do"; exit 0
  fi
  echo "sentinel holds the dead prio-3 job id; proceeding with the p4 resubmit"
fi

echo "=== submitting m4-loop-2p9b-16h100-ext-p4 ==="
cd "$(dirname "$SPEC")/../../.."
OUT=$(qz train CreateJob --data "$(cat "$SPEC")" -o yaml 2>&1) || { echo "$OUT" | head -5; exit 1; }
echo "$OUT" | grep -E "job_id|Error|Message" || echo "$OUT" | head -5
JID=$(echo "$OUT" | grep -oE "job-[a-f0-9-]+" | head -1)
if [ -n "$JID" ]; then
  echo "$JID" > "$SENT"
  echo "SUBMITTED: $JID (prio 4; recorded in $SENT)"
else
  echo "submission FAILED — re-run this script to retry"
  exit 1
fi
