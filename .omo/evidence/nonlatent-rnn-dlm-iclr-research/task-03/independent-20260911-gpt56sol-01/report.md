# Independent task-3 verdict

## Verdict

**NEEDS_FIX.** Retain the CPU subset only as engineering evidence. Do not promote task 3 beyond `[~]`: the machine contract accepts stale and invented actual-runtime claims, its success status can be mistaken for whole-task completion, and the planned CLI does not expose task 3.

## Confirmed partial behavior

1. The five owned CPU tests pass (`5 passed in 1.73s`).
2. `CanvasStateOwner` rejects cross-session reuse, prefix mismatch, and carry after a canvas edit. This proves only the purpose-built state owner.
3. Current staged source at SHA-256 `2591f58b...e0154a` separately constructs `attn_fwd` and `attn_bwd`; its key remap initializes both from each HF attention tensor. This is current-source support for clone-not-tied, not runtime identity proof.
4. Current staged source re-executes the same `self.layers` modules. The hashed run spec fixes layers `[16,32)` and one repetition, supporting the source-level `32+16=48` calculation, distinct from outer sampler forwards.
5. The actual snapshot class-body loop suite passes only after an in-memory stand-in for the missing `TypedTorchModule`; it also replaces every real RWKV block with `_FakeBlock`. It is engineering proof of the Python loop seam, not FLA/checkpoint proof.
6. Static inspection of installed `flash-linear-attention==0.5.0` supports the described cache plumbing. FLA was not imported, CUDA remained uninitialized, and no kernel/cache equivalence was executed.
7. Malformed JSON, a removed literal source marker, false tiedness, and promotion of the full checkpoint count from `unavailable` are rejected.

## Blocking defects

1. No source hashes are carried by `architecture_contract.json`; `validate_source_markers` checks only four literal strings. A contract with version `0`, total `999`, false SHA-256 fields, or drifted source retaining those strings still returns `ARCHITECTURE_COMPLETE`.
2. A payload claiming actual FLA cache equivalence, actual-model mask verification, and actual-kernel state scope while deleting all external requirements also returns `ARCHITECTURE_COMPLETE`. Unavailable facts can therefore masquerade as actual proof.
3. `prepare` returns `ARCHITECTURE_COMPLETE` and “source-grounded” for marker-only fixture files. `verify` returns the same status while acknowledging external facts are unavailable; `blocked_claims` remains `0`. Only `analyze` returns `BLOCKED_EXTERNAL`.
4. The plan's CLI rejects task 3 as `UNSUPPORTED` with exit `2`; callers cannot reach the standalone task implementation through the registered interface.
5. Blocker text is constant, not discovered. It remains unchanged after isolated module/shape files are supplied. Current facts do include absent `latent_plan.py`, absent `state_hijacking_cache.py`, and no checkpoint under `DAN/v7_arch_round`, but the text omits absent `state_hijacking_dit_torch_types.py`, which currently prevents the snapshot CPU suite from starting.
6. The owned tiny Torch harness is a separately authored container whose module-identity and count assertions follow directly from its own constructor/loops. It does not exercise the staged model class, attention pair, optimizer groups, actual caller state, cache equivalence, or masks.
7. Plan happy-path coverage is incomplete: no actual optimizer/gradient-membership test and no cached-versus-recomputed test in a claimed valid real mode.
8. Pyright is clean for the three task modules and CLI, but `test_architecture_contract.py` has seven diagnostics: one unsafe nested mapping mutation and six unresolved `ModuleList`/weight types. Basedpyright is unavailable and was not installed by policy.

## Exact repairs and pending assets

1. Add a strict, closed contract schema with exact `contract_version`, canonical payload/freshness validation, source/spec/FLA path plus SHA-256 identities, and rejection of unknown or missing proof fields.
2. Validate every claim's allowed value/classification/scope. In particular, prevent cache, masks, state scope, and external requirements from being escalated or deleted; add adversarial regressions for each.
3. Replace whole-task `ARCHITECTURE_COMPLETE` with an explicitly partial status such as `ARCHITECTURE_CPU_SUBSET_VERIFIED`; propagate `BLOCKED_EXTERNAL` and a nonzero/listed blocked-claim count through prepare/verify/analyze and machine exit policy.
4. Route task 3 through `scale.experiments.nonlatent_iclr.cli` with real happy/failure cases and test its JSON status/exit contract.
5. Discover blockers from current paths/runtime metadata rather than fixed prose; include `models.state_hijacking_dit_torch_types` and distinguish “FLA installed but runtime unexecuted” from “kernel/module absent.”
6. Supply the actual non-latent caller/cache modules (`models.latent_plan`, `models.state_hijacking_cache`, and the missing torch type module), then test real session/revision/prefix/canvas invalidation and cached-versus-recomputed equivalence.
7. Supply an identity-bound staged-v7 checkpoint config plus safetensors tensor-shape/index artifact and lineage. Existing unrelated release/base safetensors are not substitutes.
8. Add actual-model attention identity/initialization and optimizer-membership checks, and bind the input/output mask contract. FLA GPU kernel equivalence remains an external authorized-runtime asset.
9. Fix the test typing diagnostics and the truncated sentence in `DAN/nonlatent_iclr/architecture_contract.md`.

## Isolation and cleanup

- Real repository inputs and shared `DAN/nonlatent_iclr` outputs were read only; `prepare` ran only in this evidence sandbox.
- Every temporary adversarial root was removed in `finally`; the retained marker sandbox is explicitly fixture-labeled.
- No qz, CUDA execution, network, installation, git, or unpickling occurred.
- Full commands, exits, hashes, and classifications are in `commands.md`.
