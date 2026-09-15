# Commands and exits

## Owned CPU tests

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m pytest "scale/tests/nonlatent_iclr/test_architecture_contract.py" -q -p no:cacheprovider
exit: 0
output: 5 passed in 1.73s
```

Classification: CPU engineering proof. The five tests execute real `torch.nn` objects in a purpose-built tiny harness and a purpose-built state owner, but do not instantiate the staged FLA model, its cache, a real caller, or a checkpoint. The harness assertions prove its own construction and therefore are tautological if presented as actual-model proof.

## Standalone task functions

All commands set `CUDA_VISIBLE_DEVICES=""` and `PYTHONDONTWRITEBYTECODE=1`.

| Root | Function | Exit | Status | Detail |
|---|---|---:|---|---|
| marker-only evidence sandbox | `prepare` | 0 | `ARCHITECTURE_COMPLETE` | task-3 architecture contract is source-grounded |
| marker-only evidence sandbox | `verify` | 0 | `ARCHITECTURE_COMPLETE` | contract validates; external runtime claims remain unavailable |
| marker-only evidence sandbox | `analyze` | 0 | `BLOCKED_EXTERNAL` | needs FLA GPU cache caller, missing caller modules, and real checkpoint shape artifact |
| real repository, read-only | `verify` | 0 | `ARCHITECTURE_COMPLETE` | contract validates; external runtime claims remain unavailable |
| real repository, read-only | `analyze` | 0 | `BLOCKED_EXTERNAL` | needs FLA GPU cache caller, missing caller modules, and real checkpoint shape artifact |

The sandbox `prepare` result shows that four literal marker strings are sufficient for `ARCHITECTURE_COMPLETE`; this is not source-hash or actual-model grounding.

## Planned CLI surface

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m scale.experiments.nonlatent_iclr.cli verify --task 3 --case happy --repo-root "." --evidence-root "<independent evidence>"
exit: 2
status: UNSUPPORTED
detail: task 3 is not implemented
```

The standalone module is not routed through the plan's CLI contract.

## Isolated adversarial contract probe

The probe created a temporary root under this evidence directory, copied only the labeled marker fixtures, exercised `verify`, and removed the temporary root in `finally`.

```text
exit: 0
malformed_json=MALFORMED|JSONDecodeError detail
wrong_tiedness=MALFORMED|attention relationship must be clone_not_tied
stale_contract=ARCHITECTURE_COMPLETE|contract validates; external runtime claims remain unavailable
wrong_source_hash=ARCHITECTURE_COMPLETE|contract validates; external runtime claims remain unavailable
source_drift_markers_retained=ARCHITECTURE_COMPLETE|contract validates; external runtime claims remain unavailable
source_marker_removed=MALFORMED|source_marker_missing:...
cleanup=removed:adversarial-dfqqjddi
```

The stale payload changed `contract_version` to `0` and the trained total from `48` to `999`. The wrong-hash payload supplied all-zero/all-`f` SHA-256 values. Both were accepted. Literal-marker removal is detected, but content drift retaining markers is not.

## Caller-state probe

```text
exit: 0
valid_carry=PASS|same_object=True|value=(4, 5)
wrong_session=REJECTED|session_mismatch
wrong_prefix=REJECTED|prefix_mismatch
stale_canvas_after_edit=REJECTED|stale_canvas
```

Classification: behavior of the purpose-built `CanvasStateOwner` only; no actual FLA/cache caller module is exercised.

## Actual snapshot-loop seam

Direct command:

```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python "DAN/v7_arch_round/code/train/test_backbone_loop.py"
exit: 1
ModuleNotFoundError: models.state_hijacking_dit_torch_types
```

That module is absent everywhere in the repository and is not listed in the contract's external requirements.

A second run preloaded only `TypedTorchModule = torch.nn.Module` in memory, then ran the unmodified snapshot test through `runpy`:

```text
exit: 0
BACKBONE-LOOP SUITE: ALL PASS
```

Classification: dependency-stubbed engineering proof. It executes the actual `BiRWKV7ForMaskedDiffusion` class body extracted from the snapshot AST and validates its loop control flow/module reuse, but substitutes `_FakeBlock` for `BiRWKV7Block` and does not import FLA, run a real attention kernel, or load a checkpoint.

## Claim-escalation boundary

An isolated payload changed cache equivalence to `verified_on_actual_fla_gpu`, masks to `verified_actual_model`, state scope to `actual_fla_kernel_cache`, and removed every external requirement:

```text
exit: 0
claim_escalation=ARCHITECTURE_COMPLETE|contract validates; external runtime claims remain unavailable
cleanup=removed:claim-escalation-53r1r8ji
```

Thus unavailable facts can masquerade as actual-model proof in the machine-readable artifact even though the returned prose still says they remain unavailable.

The one specifically guarded unavailable fact does fail closed:

```text
exit: 0
checkpoint_count_escalation=REJECTED|full checkpoint count must remain unavailable
```

This protection is narrow; it does not cover cache equivalence, mask semantics, state verification scope, or removal of external requirements.

## Default blocker discovery and no-GPU import

```text
exit: 0
fla_spec=/usr/local/lib/python3.12/dist-packages/fla/__init__.py
fla_loaded=False
cuda_visible_devices=''
cuda_initialized=False
latent_plan_file=False
state_hijacking_cache_file=False
state_hijacking_types_file=False
vendored_rwkv7_layer=False
v7_round_safetensors=0
v7_round_pt=0
```

The installed FLA package does contain `fla/layers/rwkv7.py`, `fla/models/rwkv7/*`, and `fla/ops/rwkv7/*`; it was inspected as files, not imported. No staged-v7 checkpoint is present under `DAN/v7_arch_round/`. Repository-level release/base safetensors exist, but no artifact binds one to this staged architecture/checkpoint lineage.

An isolated root then supplied all three missing model-module filenames and a checkpoint-shape file. `analyze` still emitted the same missing-assets detail:

```text
exit: 0
modules_and_shape_present=BLOCKED_EXTERNAL|needs FLA GPU cache caller, missing caller modules, and real checkpoint shape artifact
cleanup=removed:reason-discovery-q0acp5hy
```

Therefore blocker prose is constant, not discovered. It happens to match two current missing caller modules and the absent staged-v7 checkpoint, but omits the missing type module that blocks the snapshot CPU test.

## Real-file SHA-256 ledger

```text
2591f58b1dfb7b567b979fb1f05cbb8c1e9a60490fc9fba293fcb2da48e0154a  DAN/v7_arch_round/code/models/birwkv7_diffusion.py
0bee15b5af05a785b70ddfeffa3064161f04beccea36b44e9bfd01029e286b60  DAN/v7_arch_round/code/train/train_birwkv_diffusion.py
71168c3217d6937916ff366e50d385fec579b7b8a749d85e53f2cda9f0251117  scale/experiments/nonlatent_iclr/architecture_contract.py
26c0b6354826b5dd48822bc213c62dde824545ba9a2ee32c59858f7d9f9a8a58  scale/experiments/nonlatent_iclr/architecture_checks.py
fba1dce74989cc9e397fb96c970bb4ee85967aef4f9a276ef8a6ced2dd56d6f1  scale/experiments/nonlatent_iclr/architecture_task.py
d5e25bf85b446f58824879ec314167237c6549928374a2f9aed1bccb9ff09235  scale/experiments/nonlatent_iclr/cli.py
b6762b72e1939a6a08466dcf2a97ff28d2e88ecbe45b5612a14daf7e87b13667  scale/tests/nonlatent_iclr/test_architecture_contract.py
733058e21f63f7a5c13512ea6e5d2c25dd9e51343d786ff2f93521f955dce8f6  DAN/nonlatent_iclr/architecture_contract.json
1f28257a50bf3dca444cef9523e971742436b9605495b06ed7fa2b0533a932d8  DAN/nonlatent_iclr/architecture_contract.md
```

None of these hashes is stored or checked by the task-3 contract.

Additional current-source identities:

```text
flash-linear-attention distribution version: 0.5.0
de4109b25df8162810f8b45abfc48cc8149b93a02a06cfb51ea3b36717404ffa  /usr/local/lib/python3.12/dist-packages/fla/layers/rwkv7.py
96717da020bca677e9fc672d2bec75866c1b001e1fec067105c551900e39af19  /usr/local/lib/python3.12/dist-packages/fla/models/utils.py
9d8d1951d80871440567f4653b2b903022fc5128240798347466db949ff0105f  DAN/v7_arch_round/specs/m4_loop_32gpu_half.json
```

Static file inspection confirms the installed FLA attention reads a layer cache, passes its recurrent state as `initial_state`, requests a final state only when `use_cache`, and updates the cache without replacing a recurrent state when the returned value is `None`. This supports source-level cache plumbing only. The files/version are not bound by the contract, and no kernel was imported or executed.

The loop spec supplies `--loop-range 16:32 --loop-reps 1`, hence the source-level 32+16 calculation. Its identity is likewise absent from the contract.

## LSP scope

`lsp_status` found Pyright installed; basedpyright was unavailable and not installed by policy. Diagnostics were run on the three task-3 modules, the owned test file, the CLI consumer, and both retained fixture files.

```text
architecture_contract.py: clean
architecture_checks.py: clean
architecture_task.py: clean
cli.py: clean
fixture birwkv7_diffusion.py: clean
fixture train_birwkv_diffusion.py: clean
test_architecture_contract.py: 7 Pyright errors
```

The test errors are one `reportIndexIssue` at the nested payload mutation and six type/attribute errors around `harness.layers[0]` and `.weight`.
