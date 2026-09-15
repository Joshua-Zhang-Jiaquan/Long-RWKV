# Superseding bounded runtime review

## Verdict

**PASS** for promotion of the bounded `PARTIAL_QUALIFICATION` milestone only. Task3 remains incomplete.

This review supersedes only the Torch-version reason in the immutable prior rejection at `../runtime-review/verdict.json` (SHA-256 `774eeba0a809beaeddbaff9c940121f58eb00a1f62cc13d66b7bd009d8831990`). The original rejection and all source receipts remain unchanged.

## Resolution

The difference is metadata normalization, not evidence of environment drift:

1. The worker-bound preflight producer (`manifest.py`, SHA-256 `1c9bd38b…ce65`) and runtime-environment producer (`runtime_environment.py`, SHA-256 `aeb15fe3…3a69`) call `importlib.metadata.version`. They record `2.8.0a0+5228986c39.nv25.6`.
2. The worker-bound rank producer (`runtime_checks.py`, SHA-256 `c34753e6…6dde`) records `str(torch.__version__)`. All eight hash-bound ranks record `2.8.0a0+5228986c39.nv25.06`.
3. Installed packaging `23.2` parses both strings as equal `Version` values. Both canonicalize to `2.8.0a0+5228986c39.nv25.6`; both local segment tuples are `5228986c39`, `nv25`, `6`.
4. A CPU-only `/usr/bin/python` observation with `CUDA_VISIBLE_DEVICES=""` and `torch.cuda.is_initialized() == false` reproduced both spellings simultaneously from one installed Torch distribution: distribution metadata gave `nv25.6`, while `torch.__version__` gave `nv25.06`.
5. The observed `torch/serialization.py` SHA-256 was `8263ad671446bb287b1d343309dfa7dcd4aa12ec37e89498b2522bf840e905e0`, exactly matching the worker runtime-environment observation and the bound runtime manifest. This independently supports the same-package formatting mechanism.

This proves normalized version identity for the disputed fields. It does **not** prove that every Torch source file, shared object, wheel, or container byte is identical; no such broader claim is made.

## Preserved scope

- The scientific result remains `PARTIAL_QUALIFICATION`, not full Task3 qualification.
- Full-model prefix cache remains `NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED`.
- Loop prefix cache remains `NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES`.
- The observed image digest remains unknown.
- `GetJobLog` remains unavailable with business `InternalError`.
- The source hash ledgers remain unsigned provenance records.
- No full-model cache, long-context, performance/goodput, billing, exact-duration, or arbitrary binary-identity claim follows.

## Exact next prerequisite

For Task3, wire and qualify FFN token-shift state for full-model prefix caching and prove tied-layer state-alias equivalence for loop caching under the same identity-bound runtime evidence. Preserve the image-digest limitation until a cryptographic digest is observed.
