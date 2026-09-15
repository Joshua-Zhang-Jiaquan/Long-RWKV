#!/usr/bin/env bash
# m4_ability_tracker.sh <RUN_NAME> <TAG> -- v7 arch-round in-flight eval battery.
#
# usage:  m4_ability_tracker.sh m4-mhc-2p9b mhc
#         m4_ability_tracker.sh m4-loop-2p9b loop
#
# Watches outputs_birwkv_diffusion/<RUN>/step_* as the run produces them. For
# every 2000-step boundary ckpt:
#   1. wait for model.pt to be READABLE, salvage to step_XXXX_probe_copy
#      (the proven fresh-inode cp route around GPFS visibility lag),
#   2. submit the MMLU-proxy probe (200-set, chat_frame 0) AND the fixed
#      5-corpus cap_lm panel against the probe copy,
#   3. collect the MMLU score into ability_curve_m4<TAG>.csv.
#
# V26 discipline (the n2/t3 column bugs): specs are instantiated by PYTHON with
# a placeholder-free assertion BEFORE submission -- a raw __PLACEHOLDER__ left
# in a spec is a hard abort, not a silent mis-path; the MMLU column pattern is
# WARNed loudly when it matches nothing in the summary header.
#
# Idempotent: state files under m2_baseline_triangle/m4<TAG>_tracker_state/.
set -u
RUN=${1:?usage: m4_ability_tracker.sh RUN_NAME TAG}
TAG=${2:?usage: m4_ability_tracker.sh RUN_NAME TAG}
S=/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion
O=/inspire/hdd/global_user/zhangjiaquan-253108540222/m2_baseline_triangle
Q=/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/scale/qz
ST=$O/m4${TAG}_tracker_state
PROBELOG=$O/m4${TAG}_probes.log
CURVE=$O/ability_curve_m4${TAG}.csv
mkdir -p "$ST"
PARAMS=4090000000  # 4.09B params (dual-attention 2.9B geometry); FLOPs ~ 6*N*tokens

while true; do
  # 1. salvage new readable 2000-boundary ckpts and submit both evals
  for d in $S/$RUN/step_*/; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    step=${b#step_}
    case $step in
      000|""|*[!0-9]*) continue ;;
      *probe_copy*) continue ;;
    esac
    n=$((10#$step))
    [ $((n % 2000)) -ne 0 ] && continue
    [ -f "$ST/sent_$step" ] && continue
    [ -f "$d/model.pt" ] || continue   # not yet readable; retry next poll
    cp "$d/meta.json" $ST/meta_$step.tmp 2>/dev/null || continue
    tokens=$(python3 -c "import json;print(json.load(open('$ST/meta_$step.tmp'))['tokens_seen'])" 2>/dev/null)
    rm -f $ST/meta_$step.tmp
    [ -z "${tokens:-}" ] && continue
    D=$S/$RUN/step_${step}_probe_copy
    mkdir -p "$D"
    cp "$d/meta.json" "$D/" 2>/dev/null || continue
    cp "$d/model.pt" "$D/model.pt" 2>/dev/null || continue

    COL="m4${TAG}_step${n}_raw_mmlu200"
    python3 - "$Q/m4_probe_TEMPLATE.json" "$ST/spec_$step.json" \
      "$D" "$O/tasks_${COL}" "$O/m4${TAG}_probe_step${n}.log" \
      "m4${TAG}_probe step=${n}" "$n" "$S/$RUN" "$PROBELOG" <<'PYEOF'
import json, sys
tpl, out, ck, taskdir, logname, mark, n, rundir, probelog = sys.argv[1:10]
spec = json.load(open(tpl))
cmd = spec["command"]
for k, v in [("__CKPT__", ck), ("__TASKDIR__", taskdir), ("__LOGNAME__", logname),
             ("__MARK__", mark), ("__STEP__", n), ("__RUNDIR__", rundir),
             ("__PROBELOG__", probelog)]:
    cmd = cmd.replace(k, v)
spec["command"] = cmd
json.dump(spec, open(out, "w"))
PYEOF
    # set the job name + placeholder-free assertion (V26: abort, never submit raw)
    python3 - "$ST/spec_$step.json" "m4-${TAG}-probe-step${n}" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
spec["name"] = sys.argv[2]
raw = json.dumps(spec)
bad = [t for t in ("__CKPT__", "__TASKDIR__", "__LOGNAME__", "__MARK__",
                   "__STEP__", "__RUNDIR__", "__PROBELOG__", "__PLACEHOLDER__")
       if t in raw]
assert not bad, f"unsubstituted placeholders {bad} -- aborting submission"
json.dump(spec, open(sys.argv[1], "w"))
PYEOF
    if [ $? -ne 0 ]; then
      echo "TRACKER WARN: probe placeholder assertion failed for step $n -- SKIPPING this boundary (template bug), continuing"
      touch "$ST/failed_$step"
      continue
    fi
    jid=$(qz train CreateJob --data "$(cat $ST/spec_$step.json)" -o yaml 2>/dev/null | grep -oE 'job_id: job-[a-f0-9-]+' | head -1 | awk '{print $2}')

    python3 - "$Q/m4_caplm_TEMPLATE.json" "$ST/caplm_$step.json" \
      "$D" "/inspire/hdd/global_user/zhangjiaquan-253108540222/cap_lm_m4${TAG}_s${n}" \
      "m4-${TAG}-lmeval-s${n}" <<'PYEOF'
import json, sys
tpl, out, ck, outdir, name = sys.argv[1:6]
spec = json.load(open(tpl))
spec["command"] = spec["command"].replace("__CKPT__", ck).replace("__OUTDIR__", outdir)
spec["name"] = name
spec["description"] = spec["description"] + f" [instantiated for {name}]"
raw = json.dumps(spec)
bad = [t for t in ("__CKPT__", "__OUTDIR__", "__NAME__") if t in raw]
assert not bad, f"unsubstituted placeholders {bad} -- aborting"
json.dump(spec, open(out, "w"))
PYEOF
    if [ $? -ne 0 ]; then
      echo "TRACKER WARN: caplm placeholder assertion failed for step $n -- SKIPPING this boundary (template bug), continuing"
      touch "$ST/failed_$step"
      continue
    fi
    cjid=$(qz train CreateJob --data "$(cat $ST/caplm_$step.json)" -o yaml 2>/dev/null | grep -oE 'job_id: job-[a-f0-9-]+' | head -1 | awk '{print $2}')
    echo "$step $tokens $jid $cjid" > "$ST/sent_$step"
    echo "TRACKER: step $n salvaged; MMLU probe $jid; cap_lm $cjid"
  done

  # 2. collect finished MMLU probes into the curve
  for f in $ST/sent_*; do
    [ -f "$f" ] || continue
    step=$(basename "$f"); step=${step#sent_}
    [ -f "$ST/done_$step" ] && continue
    n=$((10#$step))
    grep -q "m4${TAG}_probe step=${n} exit=" $PROBELOG 2>/dev/null || continue
    read _ tokens jid cjid < "$f"
    score=$(python3 - <<PYEOF 2>/dev/null
import csv
with open("$O/accuracy_summary.csv") as fh:
    hdr=None
    for row in csv.reader(fh):
        if hdr is None and row and row[0]=="task":
            hdr=row; continue
        if hdr and row and row[0]=="mmlu":
            for name,val in zip(hdr,row):
                if name == "m4${TAG}_step${n}_raw_mmlu200": print(val)
            break
PYEOF
)
    flops=$(python3 -c "print(f'{6*$PARAMS*$tokens:.3e}')")
    if [ -z "$score" ]; then
      score=ERR
      echo "TRACKER WARN: column m4${TAG}_step${n}_raw_mmlu200 NOT FOUND in accuracy_summary header -- check before trusting the curve"
    fi
    echo "$n,$tokens,$flops,$score" >> $CURVE
    touch "$ST/done_$step"
    echo "TRACKER: step $n -> mmlu=$score (tokens=$tokens)"
  done
  sleep 600
done
