# Actual staged-source/API inspection

All inspection was read-only. The 16.37 GB checkpoint was **not unpickled or loaded** on CPU; only ZIP metadata and `pickletools` opcode strings were read. HF safetensor counts/shapes came directly from the standard-library parse of each safetensors JSON header.

## Constructor mismatch (deterministic GPU-job failure)

- Reviewed staged source: `/inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale/models/birwkv7_diffusion.py`, SHA-256 `e6d90ce843f722f235c3e88b894ffe060847e2b5cdda5b793af801f03f865858`.
- AST inspection found `BiRWKV7ForMaskedDiffusion.from_hf_pretrained` at line 1256 with `loop_range=None, loop_reps=0` among its defaults.
- Payload call at `probe.py:72` supplies `dtype=torch.bfloat16, loop_reps=1` and omits `loop_range`.
- Staged source lines 1328-1332 raise `ValueError("loop_reps > 0 requires loop_range")` for exactly that call.
- Training provenance in `outputs_birwkv_diffusion/logs_m4_loop/qb-prod-gpu084.train.run.log:7` identifies this checkpoint as `--loop-range 16:32 --loop-reps 1`.
- Safe opcode-string scan of `model.pt` found 1,955 state-key candidates and the actual keys `loop.gates_raw`, `loop.hi`, `loop.lo`; it is a loop checkpoint and cannot be loaded into a bare model without attaching that architecture first.

AST command output:

```text
staged_method_line=1256
signature=cls=<required>, model_dir=<required>, dtype=torch.bfloat16, gate_bias_init=4.0, gradient_checkpointing=False, n_streams=1, mix_scale=1.0, loop_range=None, loop_reps=0, loop_scale=1.0
payload_call_line=72
payload_keywords=dtype=torch.bfloat16, loop_reps=1
loop_range_supplied=False
```

## Actual model geometry and safe weight-header inspection

- `config.json`: hidden size 2,560; 32 layers; vocabulary 65,536; head dimension 64; derived heads 40; FFN intermediate size 10,240; BF16 attention mode `chunk`.
- HF weights are two BF16 safetensor shards: 1,059 tensors, 2,947,735,040 elements, index metadata `total_size=5,895,470,080` bytes.
- Header-derived HF attention subset: 829 tensors / 934,049,280 elements. The custom bidirectional copies therefore contain 1,868,098,560 attention elements.
- Fusion projections/biases add 209,797,120 elements. Expected custom model total before loop control is 4,091,581,440 elements; the one trained loop gate adds one parameter element.
- The qualification currently reports only Python tensor-object counts (`unique_parameter_tensors`, `directional_parameter_tensors`), not element counts or expected exact geometry.
- Qualification checkpoint identity: `model.pt`, 16,367,167,378 bytes, SHA-256 `ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07`; flat state dict (no outer `model` key), 1,955 candidate tensor keys, with loop keys and no residual/latent/block-t keys. `meta.json` says step 4,750 and hashes to `321ce0465b2cedb13c870c1ad4ec2c7c36d14cd37670e9caa018b9e63954e7ef`.
- The checkpoint is a PyTorch ZIP/pickle archive, not safetensors. The GPU-only loader does use `torch.load(..., weights_only=True, map_location="cpu")`; CPU review did not execute it. A non-pickle safetensors conversion or an explicitly accepted restricted-unpickler risk decision remains required before calling the format intrinsically safe.

## Actual available FLA/Transformers cache API

- Locally available package metadata (without importing FLA/CUDA): `flash-linear-attention=0.5.0`, `fla-core=0.5.0`, `transformers=5.3.0`, `torch=2.8.0a0+5228986c39.nv25.6`, `safetensors=0.5.3`, `pytorch-triton=3.3.0+git96316ce52.nvinternal`.
- In the inspected FLA 0.5.0 source, `fla.models.utils.Cache` aliases `FLACache` for Transformers newer than 4.53.3. Its constructor detects Transformers' `layer_class_to_replicate` parameter; inspected Transformers 5.3.0 `Cache.__init__` has that parameter. Thus `Cache()` is source-compatible in this available environment.
- `RWKV7Attention.forward` accepts `past_key_values`/`use_cache`; it always computes and updates `conv_state`, and updates recurrent state when returned.
- `RWKV7FeedForward.forward` separately accepts `state`, consumes `state[layer_idx]['ffn_state']`, and updates `ffn_state`. The staged custom block calls `self.ffn(ffn_in)` with no cache (`birwkv7_diffusion.py:772`), so its whole-model prefix/suffix comparison is **not** a valid full causal-cache equivalence check: FFN token-shift state is absent.
- Re-running tied loop layers uses the same `layer_idx` and same cache object (`birwkv7_diffusion.py:1211-1214`), aliasing logical pass state. Base causal-cache qualification must disable the loop; any loop-cache claim needs independently indexed per-pass state or an explicit unsupported result.
- Package/source versions are not yet runtime-bound to mutable image tag `relay2:v2`; see `sourcehashes/runtime-binding.md`.

## GPU-name evidence

- Existing H100 run boot logs on this pool repeatedly report `NVIDIA H100 80GB HBM3` (for example `cap_lm_m4loop_s4750_reps3/...lm.boot.txt:9-16`).
- Payload requires both substrings `"H100"` and `"SXM"` (`probe.py:59`), so it rejects the observed authorized H100 SXM hardware name because the driver-visible product string omits `SXM`.
