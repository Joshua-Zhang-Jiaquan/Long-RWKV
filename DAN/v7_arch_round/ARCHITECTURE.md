# Model architecture — v7 architecture round

## Base model: BiRWKV7ForMaskedDiffusion (2.9B params)

The frozen base is **RWKV7-Goose-World3-2.9B-HF**, converted to a
bidirectional masked-diffusion denoiser (`convert-2p9b-bidir` in the lineage
history):

```
input_ids [B,T]                     T = 4096 (packed docs)
  └─ embeddings
       └─ 32 × BiRWKV7Block
            │   each block holds TWO RWKV7Attention instances:
            │     attn_fwd — pretrained weights, causal scan
            │     attn_bwd — the SAME weights run on torch.flip(x)
            │   fused: o = α·o_fwd + (1−α)·o_bwd
            │          α = sigmoid(fuse_proj(x) + fuse_bias)
            │          fuse_bias init +4.0  → α ≈ 0.982 at init
            │          (the untrained bidirectional model IS numerically its
            │           causal warm start — the "earned strength" philosophy
            │           every later mechanism in this repo copies)
       └─ norm → lm_head (65,536 vocab)
```

Training objective: absorbing-state masked diffusion. Random token spans are
replaced by `mask_token_id = 65535` at ratios {0.15, 0.3, 0.5, 0.7, 0.9};
the model predicts the original tokens at masked positions (mask_ce), plus a
causal-replay CE term (lam_c = 0.10) that anchors the forward stream.

## Mechanism under test 1 — mHC (manifold hyperconnections) [REFUTED]

`ResidualStreamMixer` (`code/models/residual_streams.py`): replaces the single
residual stream with a small bank of coupled streams H ∈ [B,T,n,C] (n=4):

```
per layer ℓ:  x  = Σ_i w_pre[i]·H[i]           (learned input mixing)
              o  = block_ℓ(x)
              Δ  = o − x
              H' = A_res·H + a_post·Δ           (learned recurrent mixing)
```

Identity at init (w_pre = e_0, A_res = I, a_post = e_0) — attaching the
mechanism is a no-op until gradient opens the mixing. Only 768 new parameters
at n=4. Attached via `model.attach_residual_streams(n_streams=)`; state-dict
keys under `residual_streams.*`.

## Mechanism under test 2 — weight-tied loop [POSITIVE, see report.md]

`BackboneLoopControl` (`code/models/residual_streams.py`): re-runs the SAME
block modules of layers [16,32) one or more extra times (weight-tying by
construction — zero new block parameters, only one tanh-bounded scalar gate
per extra pass, exactly 0 at init):

```
after the main 0→32 pass:
  for p in range(reps):                       # reps = 1 in training
      g_p = tanh(gates_raw[p]) · loop_scale   # 0 at init
      h_pass = run_layers(16..32, h)
      h = h + g_p · (h_pass − h)
      # v_first highway is CARRIED between passes (more depth, not reset
      # depth); post-loop v_first updates are discarded → h/logits
      # unchanged at init even for reps > trained (extrapolation-safe)
```

Attached via `model.attach_backbone_loop(loop_range, loop_reps)`; trainer
flags `--loop-range 16:32 --loop-reps 1`. Trained NFE: 48 blocks-passes
(32 + 16). The forward accepts `loop_reps_override` for eval-time depth
extrapolation — extra passes index `gates[min(p, len−1)]`, i.e. they reuse
the LAST TRAINED gate (the conservative choice; validated in
`code/train/test_backbone_loop.py` i2: override 4 and 9 are bit-identical to
legacy at init, override 0 disables the loop).

## Trainer + eval integration

- Trainer (`code/train/train_birwkv_diffusion.py`): FSDP full-shard,
  gradient checkpointing, mechanism attach flags, step-0 mechanism-parity
  probe (identity-at-init assert; resume-aware — trained gates make bf16-ULP
  drift legitimate on resume, so the assert skips when start_step>0).
- Checkpoints: mechanisms live at the FSDP root (`residual_streams.*`,
  `loop.*`) so their state lands in model.pt under stable prefixes.
- Eval loader (`code/eval/birwkv_diffusion_model.py`): sniffs mechanism keys
  in the checkpoint and attaches BEFORE load_state_dict; supports the
  `LOOP_REPS_OVERRIDE` env for W-A5 (wraps model.forward; works for both the
  PPL harness and the sampler harness).
- CPU safety cases (`code/train/test_*.py`): exact identity at init, grads
  flow to gates AT init (no one-step delay), bounded gates, weight-tying
  (loop adds ZERO new block modules), range/reps validation, state-dict
  round-trip.

## File map (this folder = snapshot; live copies in scale/ of both trees)

| file | role |
|---|---|
| code/models/birwkv7_diffusion.py | the model: blocks, fusion, mechanism attach, forward + loop_reps_override |
| code/models/residual_streams.py | ResidualStreamMixer (mHC), BackboneLoopControl (loop) |
| code/train/train_birwkv_diffusion.py | FSDP trainer (mechanism flags, parity probe, val arms) |
| code/train/test_{backbone_loop,residual_streams}.py | mechanisms' CPU test cases |
| code/eval/birwkv_diffusion_model.py | eval loader (sniff + attach + LOOP_REPS_OVERRIDE) |
| code/eval/lm_eval.py | 5-corpus PPL panel harness (paired per-doc nll) |
| code/eval/sampler_eval_offline.py | decode harness (em/tau/residue/repetition, mask×steps grid) |
| code/eval/merge_shards.py | shard merge |
| code/qz/launch_*.sh | job launchers (train, lm panel, sampler) |
| code/qz/m4_ability_tracker.sh | 2000-boundary MMLU+panel auto-submitter |
| scripts/m4_loop_prober.sh | per-500-step MMLU prober (marker-deduped) |
| scripts/wl_endpoint_battery.py | endpoint battery submitter (10 jobs, dup-checked) |
| scripts/wl_gate_analysis.py | paired gate analysis (PPL + decode + W-A5) |
