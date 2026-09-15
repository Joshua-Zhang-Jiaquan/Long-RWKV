#!/usr/bin/env bash
# Progressive-ckpt MMLU prober for the mHC run: salvage every 500-step ckpt before
# the trainer's keep-3 prune eats it, submit the MMLU probe, write the tracker-format
# sent_ marker so the existing collection loop records the score in the ability curve.
# Cleanup: rm non-2000-boundary probe copies once their probe reported exit=;
# 2000-boundary copies persist until the cap_lm merged/ output exists (panel reads them).
set -u
S=/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion
RUN=m4-mhc-2p9b
O=/inspire/hdd/global_user/zhangjiaquan-253108540222/m2_baseline_triangle
ST=$O/m4mhc_tracker_state
Q=/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/scale/qz
LOG=/root/.claude/jobs/6f32d845/tmp/m4_progressive_prober.log
mkdir -p "$ST"
while true; do
  for d in $S/$RUN/step_*/; do
    [ -d "$d" ] || continue
    b=$(basename "$d"); step=${b#step_}
    case $step in ""|*[!0-9]*) continue ;; esac
    n=$((10#$step))
    [ $((n % 500)) -ne 0 ] && continue
    # 2000-multiples belong to the m4 tracker (probe+caplm with shared sent markers) --
    # the prober touching them caused duplicate submissions + marker-write races (s4000/s6000)
    [ $((n % 2000)) -eq 0 ] && continue
    m="$step"   # RAW dir suffix (8-digit, tracker convention); %06d normalization caused duplicate s4000 submission
    [ -f "$ST/sent_$m" ] && continue
    [ -f "$d/model.pt" ] && [ -f "$d/meta.json" ] || continue
    tokens=$(python3 -c "import json;print(json.load(open('$d/meta.json'))['tokens_seen'])" 2>/dev/null)
    [ -z "${tokens:-}" ] && continue
    D=$S/$RUN/step_${m}_probe_copy
    if [ ! -d "$D" ]; then
      mkdir -p "$D"
      cp "$d/meta.json" "$D/" && cp "$d/model.pt" "$D/model.pt" || { rm -rf "$D"; continue; }
    fi
    jid=$(python3 - "$Q/m4_probe_TEMPLATE.json" "$D" "$n" "$S/$RUN" "$ST" "$tokens" >> "$LOG" 2>&1 <<'PYEOF'
import json, sys, re, subprocess
tpl, ck, n, rundir, st, tokens = sys.argv[1:7]
n = int(n)
O = '/inspire/hdd/global_user/zhangjiaquan-253108540222/m2_baseline_triangle'
spec = json.load(open(tpl))
cmd = spec['command']
for k, v in [("__CKPT__", ck), ("__TASKDIR__", f"{O}/tasks_m4mhc_step{n}_raw_mmlu200"),
             ("__LOGNAME__", f"{O}/m4mhc_probe_step{n}.log"), ("__MARK__", f"m4mhc_probe step={n}"),
             ("__STEP__", str(n)), ("__RUNDIR__", rundir), ("__PROBELOG__", f"{O}/m4mhc_probes.log")]:
    cmd = cmd.replace(k, v)
spec['command'] = cmd
spec['name'] = f'm4-mhc-probe-step{n}'
raw = json.dumps(spec)
bad = [t for t in ('__CKPT__','__TASKDIR__','__LOGNAME__','__MARK__','__STEP__','__RUNDIR__','__PROBELOG__','__PLACEHOLDER__','UNSET') if t in raw]
assert not bad, f'unsubstituted {bad}'
r = subprocess.run(['qz','train','CreateJob','--data',json.dumps(spec),'-o','yaml'], capture_output=True, text=True)
mm = re.search(r'job_id: (job-[a-f0-9-]+)', r.stdout + r.stderr)
if mm:
    import os as _os
    _tmp = f'{st}/.sent_{n:08d}.tmp'
    with open(_tmp, 'w') as _f:
        _f.write(f'{n:08d} {tokens} {mm.group(1)} -\n'); _f.flush(); _os.fsync(_f.fileno())
    _os.replace(_tmp, f'{st}/sent_{n:08d}')  # atomic replace (an earlier patch split open().write() leaving 0-byte markers: s5500/s6500/s7000)
    print(f'PROBER step={n} probe={mm.group(1)}')
else:
    print(f'PROBER step={n} FAILED: {(r.stdout+r.stderr)[:200]}')
PYEOF
)
    # Belt and suspenders: the python's marker write was lost twice (s5500, s6500 --
    # 0-byte file after open('w')). If the marker is missing/empty, the shell rewrites it
    # from the captured submission line.
    if [ ! -s "$ST/sent_$m" ]; then
      pj=$(grep "PROBER step=$n " "$LOG" 2>/dev/null | tail -1 | grep -oE 'job-[a-f0-9-]+' | head -1)
      [ -n "$pj" ] && echo "$m $tokens $pj -" > "$ST/sent_$m"
    fi
  done
  # cleanup: probe copies whose work is done
  for D in $S/$RUN/step_*_probe_copy/; do
    [ -d "$D" ] || continue
    b=$(basename "$D"); step=${b#step_}; step=${step%_probe_copy}
    n=$((10#$step))
    if ! grep -q "m4mhc_probe step=$n exit=" "$O/m4mhc_probes.log" 2>/dev/null; then continue; fi
    if [ $((n % 2000)) -eq 0 ]; then
      # ALL FIVE corpora must be merged before the copy is deletable -- panels load the ckpt
      # lazily per corpus, and "any corpus merged" deleted the s6000 copy mid-panel (0/8 shards
      # for pg19/owt/lambada/lm1b, job-06e2499d).
      nm=$(ls -d /inspire/hdd/global_user/zhangjiaquan-253108540222/cap_lm_m4mhc_s$n/*/merged/ 2>/dev/null | wc -l)
      [ "$nm" -ge 5 ] && rm -rf "$D"
    else
      rm -rf "$D"
    fi
  done
  sleep 600
done
