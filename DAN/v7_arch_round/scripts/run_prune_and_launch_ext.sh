#!/usr/bin/env bash
# v7_arch_round — Step 0 + Step 1 (v2): prune checkpoints (best+last kept,
# incl. the CLOSED W-L1 latent lane's dead weight) and launch the full-budget
# loop extension (m4-loop-2p9b, resume s4750 -> 9500 steps).
#
# Safe to re-run: pruning skips dirs that are already gone; the job submission
# is guarded by a sentinel file so it cannot double-submit.
set -euo pipefail

S=/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion
P=/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/DAN/v7_arch_round
SPEC=$P/specs/m4_loop_ext_9500_job.json
SENT=$P/specs/.ext_submitted_job_id

echo "=== STEP 0: checkpoint pruning (best-performed + last only) ==="
# Round runs:  kept = loop s4750 (last=gated endpoint) · mHC s9500 (last=gated
# endpoint) · N2 s6000_probe_copy (best performed, MMLU 48.0, bracket) + s9500 (last)
# W-L1 latent arm: CLOSED 2026-09-06 (gate-stopped by design). Best ckpt s2000
# is preserved as step_00002000_probe_copy + e_target_manifest.json. The raw
# s2000 is a duplicate of that copy; s2500/s3000 are the gate-failed tail.
TARGETS=(
  "$S/m4-loop-2p9b/step_00004000"
  "$S/m4-loop-2p9b/step_00004500"
  "$S/m4-mhc-2p9b/step_00008500"
  "$S/m4-mhc-2p9b/step_00009000"
  "$S/n2-knowpt-2p9b/step_00002000_probe_copy"
  "$S/n2-knowpt-2p9b/step_00004000_probe_copy"
  "$S/n2-knowpt-2p9b/step_00008000_probe_copy"
  "$S/wl1-latentv2-arm/step_00002000"
  "$S/wl1-latentv2-arm/step_00002500"
  "$S/wl1-latentv2-arm/step_00003000"
)
# safety: every target must be a step_ dir under exactly these four runs
for t in "${TARGETS[@]}"; do
  case "$t" in
    "$S/m4-loop-2p9b/step_"*|"$S/m4-mhc-2p9b/step_"*|"$S/n2-knowpt-2p9b/step_"*|"$S/wl1-latentv2-arm/step_"*) ;;
    *) echo "REFUSING unexpected path: $t"; exit 1;;
  esac
done
# safety: the checkpoints we keep must exist and be intact
for keep in "$S/wl1-latentv2-arm/step_00002000_probe_copy/model.pt" \
            "$S/m4-loop-2p9b/step_00004750/model.pt" \
            "$S/m4-mhc-2p9b/step_00009500/model.pt" \
            "$S/n2-knowpt-2p9b/step_00006000_probe_copy/model.pt" \
            "$S/n2-knowpt-2p9b/step_00009500/model.pt"; do
  [ -f "$keep" ] || { echo "ABORT: kept checkpoint missing: $keep"; exit 1; }
done
for t in "${TARGETS[@]}"; do
  if [ -d "$t" ]; then
    sz=$(du -sh "$t" | cut -f1); rm -rf "$t"
    echo "  deleted $t ($sz)"
  else
    echo "  already gone: $t"
  fi
done
echo "--- remaining checkpoints ---"
for run in m4-loop-2p9b m4-mhc-2p9b n2-knowpt-2p9b wl1-latentv2-arm; do
  echo "  $run: $(ls "$S/$run" | tr '\n' ' ')"
done
FREE_GB=$(df -BG --output=avail /inspire/hdd/global_user/zhangjiaquan-253108540222 | tail -1 | tr -dc '0-9')
echo "  disk free now: ${FREE_GB}G"
if [ "$FREE_GB" -lt 150 ]; then
  echo "ABORT: less than 150G free after pruning — not launching the extension."
  exit 1
fi

echo
echo "=== STEP 1: launch full-budget loop extension (s4750 -> 9500) ==="
if [ -f "$SENT" ]; then
  echo "  already submitted: $(cat "$SENT") — skipping submission"
else
  cd "$(dirname "$SPEC")/../../.."
  OUT=$(qz train CreateJob --data "$(cat "$SPEC")" -o yaml 2>&1)
  echo "$OUT" | grep -E "job_id|name|Error|Message" || echo "$OUT" | head -5
  JID=$(echo "$OUT" | grep -oE "job-[a-f0-9-]+" | head -1)
  if [ -n "$JID" ]; then
    echo "$JID" > "$SENT"
    echo "  SUBMITTED: $JID (recorded in $SENT)"
  else
    echo "  submission FAILED — see output above; re-run this script to retry"
    exit 1
  fi
fi

echo
echo "=== DONE ==="
echo "Extension: m4-loop-2p9b resumes from step_00004750 to 9500 (10B tokens, N2-matched lr)."
echo "Watch: logs_m4_loop/ boot + run logs; MMLU probes/panels fire automatically at 500/2000-step boundaries."
