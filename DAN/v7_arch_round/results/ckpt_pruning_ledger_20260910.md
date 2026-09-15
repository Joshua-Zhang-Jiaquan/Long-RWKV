# Checkpoint pruning ledger (user directive: keep only best-performed + last) — 2026-09-10

## Kept
- `outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750` (46G, model+optim) — last AND best-gated endpoint (all PRIMARY/decode/W-A5 gates read on it); neutral copy at `m2_baseline_triangle/m4loop_endpoint_ckpt`
- `outputs_birwkv_diffusion/m4-mhc-2p9b/step_00009500` (16G) — last + gated endpoint; neutral copy at `m2_baseline_triangle/m4mhc_endpoint_ckpt`
- `outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00006000_probe_copy` (16G) — best performed (MMLU 48.0) + pre-registered loop-arm bracket
- `outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00009500` (46G) — last (mHC gate endpoint)

## Deleted (target list — executed by the user/command below)
- `outputs_birwkv_diffusion/m4-loop-2p9b/step_00004000` (46G)
- `outputs_birwkv_diffusion/m4-loop-2p9b/step_00004500` (46G)
- `outputs_birwkv_diffusion/m4-mhc-2p9b/step_00008500` (16G)
- `outputs_birwkv_diffusion/m4-mhc-2p9b/step_00009000` (16G)
- `outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00002000_probe_copy` (16G)
- `outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00004000_probe_copy` (16G)
- `outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00008000_probe_copy` (16G)
Total: ~172G freed.

All panel/sampler DATA kept: every `cap_lm_*` and `sampler_gate_*` dir untouched (the gates' raw per-document outputs remain independently reproducible from the kept checkpoints where applicable).

## Command (run if not yet executed)
```
S=/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion
rm -rf $S/m4-loop-2p9b/step_00004000 $S/m4-loop-2p9b/step_00004500 \
       $S/m4-mhc-2p9b/step_00008500 $S/m4-mhc-2p9b/step_00009000 \
       $S/n2-knowpt-2p9b/step_00002000_probe_copy \
       $S/n2-knowpt-2p9b/step_00004000_probe_copy \
       $S/n2-knowpt-2p9b/step_00008000_probe_copy
```
=== /inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/m4-loop-2p9b/step_00008500/00009000 pruned (kept s9500 + neutral m4loop_ext_endpoint_ckpt)
