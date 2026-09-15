# Independent Task 2 verdict: REJECT

Task 2 is not acceptance-complete. Its 7 owned tests and the 62-test Task 1 regression pass, and the published five pooled `max_run_frac` values reproduce from current raw records, but the implementation does not satisfy the plan's semantic, cell-table, verification-truth, or recovery contracts.

## Decisive reproductions

1. `sequence_metrics((1,2,3,4,0), (9,9,9,4,0), ...)` returns `max_run_frac=0.25`; the generated prediction's longest run is `3/4=0.75`.
2. The hand-computable tau-a fixture `[1,1,2]` is `2/3`, but Task 2 exposes no tau computation function.
3. Two active tokens in neither condition nor prediction mask remain visible and are accepted; different seeds produce identical results because the sampled random value is discarded.
4. A panel declaring eight shards, 999 records, and four grid cells passes with one shard and one pair. `failure="OOM"` plus a finite value is averaged as success.
5. Pooled tables hide required cells: loop vs N2-s6000 reports `-0.04907417` (better), while `r100_s32` is `+0.0478515625` (worse); reps3/reps1 reports `+0.06706352` while `r100_s32` is `-0.009765625`.
6. Tampering a correction to `delta=12345`, `pair_count=999`, and the opposite interpretation still verifies `METRICS_COMPLETE`; so do empty corrections, schema 999, absent receipt, and absent Markdown.
7. Injected report/receipt write failure leaves an immutable partial publication: retry says `PUBLICATION_EXISTS`, and verification says `METRICS_COMPLETE` with no terminal receipt.

## Exact repair list

1. In `metrics.py::sequence_metrics`, calculate repetition on non-pad **prediction** tokens. Reject length/pad inconsistencies and represent zero prediction positions with a typed unavailable outcome or input error, never NaN in a qualifying record.
2. Implement and export the documented tau-a commit-order metric; lock ascending, descending, tied (`[1,1,2] -> 2/3`), too-short, and masked-only controls. Keep it diagnostic-only.
3. In `prompt_preserving_corruption`, require every non-pad position to satisfy exactly one of condition/prediction and every pad position neither; this rejects unclassified future observations. Remove `seed`, or specify and test a real seeded operation.
4. Parse sampler shards with a strict schema: reject booleans for integer/float fields; validate filename/index/`num_shards`, per-shard and total counts, consistent metadata/grid, complete expected arms/cells, and duplicate keys across all shards/panels.
5. Model failed cells explicitly. Retain failure reason/count in every table; never average failed records as successes and never erase a whole panel merely because a retained failure has no metric.
6. Recompute and publish raw-bound rows per `(mask_ratio, steps, repetitions, comparison)` for `max_run_frac`, `em` renamed `masked_token_accuracy`, tau-a, and available diagnostics. Keep pooled all-arm averages secondary only; label legacy normal CIs descriptive, and treat tiny floating residuals as neutral rather than printing `0.00000000 worse`.
7. Do not publish sequence exact match for these historical panels: decoded target/prediction sequences are absent. Report it explicitly as unavailable/N/A until fresh decoded sequences exist.
8. Give the correction payload a strict versioned schema and attempt identity. `verify_task_two` must require the exact comparison/cell set, recompute every value from current records, compare counts/deltas/directions/CI labels, regenerate or digest-check Markdown, and validate the exact terminal receipt's attempt, payload hash, code hash, and status.
9. Make Task 2 publication recover like Task 1: validate an existing JSON, safely finish a missing Markdown/receipt, and only report complete after the terminal receipt exists. Distinguish source-read failure from publication interruption.
10. Expand CLI failure verification to prove all plan-named failures in one bound result: reversed polarity, NaN/nonfinite value, missing pair/coverage, and future-answer leakage. Add regressions for every reproduction above.

## Evidence index

- Commands/exits/counts: `test-runs.txt`, `cli-runs.txt`, `verification.txt`.
- Numeric and schema repros: `numeric-controls.json`, `validation-outcomes.json`, `validation_probe.py`.
- Full raw controls: `raw-cell-controls.json`, `raw-cell-command.txt`, `raw_cell_probe.py`, `raw_cell_models.py`.
- Verifier/recovery repros: `correction-outcomes.json`, `correction-command.txt`, `correction_probe.py`.
- Integrity: `baseline-sha256.txt`, `final-sha256.txt`.
