# Debug Journal — task-01 final blockers

Started: 2026-09-11
Goal: Repair only exact external/claim coverage, initial source-race handling, and canonical default receipt/CLI state.

## Environment snapshot

- Runtime: CPython via `python`; pytest; standard-library CLI.
- Entry: `python -m experiments.nonlatent_iclr.cli` from `scale/`.
- Repository has no git metadata; no git operation is permitted.
- References read: programming Python README/type-patterns/data-modeling/error-handling; debugging Python runtime and methodology setup/investigate/fix.
- Existing repro authority: `repair-final-20260911-a/review-goal.md`, `review-code-quality.md`, `review-context.md`.

## Hypotheses

1. [CONFIRMED BY PRIOR REVIEW] Schema compares only snapshot tuples, so external omissions/extras remain semantically valid — distinguishing evidence: digest-consistent 13/3 removal returns `AUDIT_COMPLETE` — fix: exact 32-tuple contract.
2. [CONFIRMED BY PRIOR REVIEW] Claim validation checks shape but not identity/bindings/classification — distinguishing evidence: one generic observed claim returns `AUDIT_COMPLETE` — fix: machine claim IDs.
3. [CONFIRMED BY PRIOR REVIEW] Panel file stats occur outside the hash error boundary — distinguishing evidence: deletion after enumeration raises `FileNotFoundError` — fix: typed observation error.
4. [CONFIRMED BY PRIOR REVIEW] The current receipt was written beneath a non-default evidence root — distinguishing evidence: root-flag-free happy verify returns `PUBLICATION_PENDING` — fix: canonical receipt recovery after identity checks.

## Artifacts to revert or clean

- [ ] `scale/experiments/nonlatent_iclr/contract.py` — new versioned machine contract.
- [ ] `scale/experiments/nonlatent_iclr/{schema,claims,inventory,models,service}.py` — bounded production edits.
- [ ] `scale/tests/nonlatent_iclr/{test_schema_contract,test_service_inventory,test_cli_contract,test_audit_cli}.py` — failing-first regressions.
- [ ] `archive-before-replacement/` — immutable copies of current generated outputs and relevant receipts; preserve as evidence.
- [ ] Temporary pytest/manual-QA fixtures — remove through context managers and verify absent.
- [ ] `DAN/nonlatent_iclr/evidence_ledger.json` and audit output — archive before schema-version replacement.
- [ ] Canonical default receipt for the replacement attempt — preserve as final publication evidence.

## Findings

- Prior independent reviewers reproduced all four requested blockers; no broad rediscovery is required.
- RED schema coverage: seven digest-shape-valid external/claim omissions or fabrications returned `None` from validation instead of rejection; see `red-schema-coverage.txt`.
- RED machine contract: the complete version-2 ledger with eight stable `claim_id` values was rejected as `invalid schema_version`; focused pytest failed 1/1 in 0.16s.
- RED initial-inventory race: deterministic deletion after panel listing escaped `prepare_task_one` as `FileNotFoundError` from `panel_artifact` size collection; focused pytest failed 1/1 in 0.20s.
- RED default CLI: root-flag-free happy verification from `scale/` returned `PUBLICATION_PENDING`, incorrectly reported `blocked_claims=0`, and exited 2; see `red-default-cli.txt`.
- GREEN: 62/62 tests and all changed-file LSP diagnostics passed; see `green-pytest-lsp.txt`.
- Published schema-version-2 attempt `6626a347e9434fc18acd0d2ffd6ca461` to the canonical default evidence root. Ledger, audit, and receipt hashes are bound in `changed-file-sha256.txt`.
- Default CLI QA passed happy/failure/analyze/fresh-render routes; repeat prepare correctly returned immutable `PUBLICATION_EXISTS` with eight blocked claims.
- Nine-class manual QA passed all categories and removed its temporary fixture; see `manual-qa-summary.json` and `manual-qa-commands.json`.

## Final fix

Pending red → green proof.
