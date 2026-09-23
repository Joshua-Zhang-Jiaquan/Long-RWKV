# Completion checks for the frozen confirmation campaign

The nine training jobs and six endpoint audits must all finish before the paper
can report the complete confirmation. Keep every seed and both history arms.
The frozen recipe, panel, partitions, staged runtime, and terminal steps remain
unchanged by these reporting tools.

1. Run `python -m theory_mvp.train04confirm.training_audit` after all three
   parents and six branches complete. This verifies full contiguous update
   intervals, finite loss/gradient records, prescribed learning rates and token
   counts, same-seed parent provenance, and actual terminal checkpoint hashes.
   The expected total is 45,182,400 training input tokens. Its GPU-hour figure
   covers the worker's measured training interval only; it is not total billed
   compute and excludes initialization, evaluation, and development.
2. Confirm the live orchestration process has collected all six terminal audits.
   Run `python -m theory_mvp.train04confirm.report`. This revalidates raw endpoint
   shards, the exact fixed panel, inference precision, all terminal checkpoint
   hashes, and every seed/arm combination before aggregation.
3. Inspect all families, policies, per-structure values, and paired seed
   contrasts. Retain adverse findings. Describe means and ranges across these
   three data/corruption seeds without population significance claims or
   independent-pretraining claims.
4. Update the paper from the resulting files; replace pending confirmation
   language, preserve development results as development, and distinguish exact
   oracle controls from trained-model measurements. Rebuild and inspect the
   rendered result tables and relevant main-text claims.
5. Finish the claim/evidence and provenance audit before declaring completion.

The training-audit tests exercise all prescribed phase budgets and refusal of
missing/duplicate updates, wrong parent provenance, nonfinite losses, wrong
token counts, wrong learning rates, reversed clocks, and incomplete execution.
Eleven focused tests passed on 2026-09-22. These tests validate the checker;
actual nine-job completion is still pending.

## JSON reporting repair

Use `python -m theory_mvp.train04confirm.collect` for future collections. The
frozen collector compared JSON lists against native tuples and rejected all
otherwise identical support vectors. The reporting-only replacement normalizes
container types while retaining exact order and all validation. Training and
inference sources remain frozen. See
`results/train04confirm/collector_schema_repair.json`.
