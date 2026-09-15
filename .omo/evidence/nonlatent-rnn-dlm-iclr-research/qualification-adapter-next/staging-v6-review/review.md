# Adapter-v6 focused CPU launch-preflight review

## Verdict

**REJECT** immutable adapter-v6 staging under the mandatory five-lane fail-fast review rule. This is not a GPU, live-runtime, or Task3 verdict.

The candidate itself passed every substantive bounded check. The rejection is caused by review admissibility and future-identity findings, not by a demonstrated adapter implementation defect.

## Candidate identity and substantive checks

- Manifest: `692d1cc88d6b2c6335736d47492aa933bf57b9a23a4d0dcc9e5e3eeda165cf32`, 111/111 unique entries with matching hashes and sizes.
- Tree: `5b86080fba3c7ed11c65c9b7045349f0ad074973921ff04c22347ac4db8690f9`.
- Launcher: `8b7970c27a5a99b18a251e7c84b680dce6fa74f336866c9ff22ab871900b73c5`.
- Approved `full_canvas.py`: `1e5400e4283744d6c29ce533f0138b2b3e2b7bcd78b1a64b668a79f97c6b3e22`.
- Inherited bytecode guard: `b5bcff58019e09a2acee0bf0ea6b4839371f91c3e25c32982d56f47f5c3b8a09`.
- V5 origin validation, lazy imports, and `ParameterEvidence.model_validate` parsing remain present.
- The candidate caller performs adapter `open`, then `forward`, and unconditional `reset` in `finally`; it passes `force_forward=False`, `z_slots=None`, `state_cache=None`, and `use_cache=False`.
- External package/model/checkpoint identities and all 72 owned model-source identities match v5.
- Post-execution topology remained zero writable paths, symlinks, and bytecode paths.

## Fresh review executions

- Candidate lifecycle/parameter scope: 5 passed, pytest 18.62s, wall 20.15s, exit 0.
- Release/manifest/preservation scope: 3 passed, pytest 0.93s, wall 2.50s, exit 0.
- Approved regression scope: 55 passed plus 11 passed, 66/66 disjoint with no duplicates.
- Inherited controls: 11 v5 bytecode/freshness plus 6 v4 origin controls, 17/17 passed.
- Strict CPU preflight: `PREFLIGHT_VERIFIED`, 111 files, exit 0.
- Runtime environment: `MATCH`, 8/8 package sources, 6/6 versions, and no runtime model modules loaded.

The original staging timings and basedpyright receipt were pre-existing evidence. The independent QA lane reran the behavioral and preflight scopes; the exact basedpyright result was reused rather than rerun. Workspace engineering files returned no LSP diagnostics. Global immutable candidate paths could not be queried by LSP because the service rejects files outside the request working directory.

## Blocking issues

1. The independent QA lane used one unprefixed `/usr/bin/python -c` timing helper before its corrected commands. Although this did not write into either immutable release and all candidate commands subsequently used external caches, it violates the literal rule that every Python invocation be source-clean. Minimum fix: repeat the complete CPU QA pass with the environment prefix on every Python command and publish the exact commands plus post-run tree/bytecode/cache checks.
2. No fresh adapter-v6 controller request, nonce, or permit exists. The consumed qualification05 request/nonce/permit were not reused. Before any later live action, independently verify a fresh v6-bound controller identity against the launcher's retained run-id contract.

One code lane also returned FAIL because it could not discover `changed-files.json`; that premise is invalidated by the pre-lane read and its staging ledger hash. It is not a candidate defect, but the five-lane panel was not unanimously passing.

## Preserved boundaries

These tests use a recording CPU model. They do not prove real-model state isolation, checkpoint loading, CUDA, GPU lifecycle, full-model caching, or loop caching. No job or live action occurred, and Task3 remains incomplete.
