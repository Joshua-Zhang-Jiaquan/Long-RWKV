# Task 03e: source-bound CPU partial contract

## Outcome

The whole task remains `BLOCKED_EXTERNAL`. The only successful CPU result is
`ENGINEERING_ONLY`; it never describes the full architecture as complete.

## Red / green

- Baseline: `python -m pytest scale/tests/nonlatent_iclr/test_architecture_contract.py -q`
  reported `5 passed`.
- Red: after adding the source/receipt/status regressions, collection failed on
  missing `validate_contract_json` before the repair existed.
- Green: the repaired suite reported `8 passed` twice. It includes stale source
  hash, schema version, blocked-count, false runtime claim, removed blocker,
  partial/stale receipt, and ownership invalidation rejection.
- Standalone happy: `python -m scale.experiments.nonlatent_iclr.architecture_task
  --case happy --repo-root .` emitted `BLOCKED_EXTERNAL`, `ENGINEERING_ONLY`,
  and `blocked_claims: 6` with exit 0.
- Standalone failure emitted `EXPECTED_FAILURE_CONFIRMED` for the malformed
  contract rejection with exit 0.

## Bound identities

- Contract SHA-256:
  `7f04bf824dd0acec3f4b2c54f949d3722b224d14a369a0e1824445e8eec2337e`.
- Receipt SHA-256:
  `cd17dc51c55a4f04042d63bdc3588aabed6527f3831ccbdcb6d91b887b57b616`.
- Staged model SHA-256:
  `2591f58b1dfb7b567b979fb1f05cbb8c1e9a60490fc9fba293fcb2da48e0154a`.
- Installed `flash-linear-attention` 0.5.0 source is hash-bound in the contract
  without importing FLA or initializing CUDA. This is source inspection only.

## Engineering controls and cleanup

The real Torch harness proves only its own copied-parameter and shared-loop
controls. The original snapshot loop suite also executes through an explicit
test-only `TypedTorchModule`/fake-block loader. Neither proves FLA cache kernels,
actual caller state ownership, optimizer membership, masks, or checkpoint shape.

The prior owned contract was archived using `write_once` before fresh publication.
No temporary source mutation, CUDA, qz, network, install, git, or checkpoint
unpickling occurred.
