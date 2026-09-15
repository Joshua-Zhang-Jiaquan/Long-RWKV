# Task-01 Repair — Five-Lane Independent Review

## DoneClaim

The repair implementation, generated ledger/audit, executor test suite, real CLI exercise, nine-class manual QA, hash binding, evidence reporting, and five requested independent review lanes are complete. **Task-01 is not accepted:** three of five independent lanes rejected it. No rejection fix or task-2 work was started.

Plan task 1 remains unchecked at `.omo/plans/nonlatent-rnn-dlm-iclr-research.md:217`.

## Reviewer verdicts

| Lane | Verdict | Confidence/severity | Session identity | Local evidence |
|---|---|---|---|---|
| Goal and constraints | **FAIL** | high | `ses_f7044295cffelLG5LaBZq1PTqf` | `review-goal.md` |
| Hands-on QA | **PASS** | high | `ses_f7044260fffeEuC5H49ob4bToL` | `review-qa.md` |
| Code quality/correctness | **FAIL** | high | `ses_f704427c8ffePpiu656Z9hG0gw` | `review-code-quality.md` |
| Security | **PASS** | maximum MEDIUM, non-blocking | `ses_f70442702ffejEO5UbNGqNGyGK` | `review-security.md` |
| Context and historical bindings | **FAIL** | high | `ses_f704426e8ffe8Z6rQ7SrooEceg` | `review-context.md` |

Aggregate verdict: **FAILED — criteria not met**. Review policy requires every lane to pass.

## Blocking findings and repair status

1. **External and claim coverage enforcement — not repaired.** `schema.py:166-191` enforces exact coverage only for 16 snapshot tuples. Goal, context, and quality reviewers independently reproduced `AUDIT_COMPLETE` for a digest-consistent ledger missing the 13 panel and 3 metadata rows. They also reproduced acceptance after replacing eight historical claim mappings with one generic/fabricated claim. The generated ledger is complete; validation of replacement ledgers is not.
2. **Prepare-time file race — not repaired.** Code-quality review reproduced an uncaught `FileNotFoundError` when a panel file disappeared between enumeration and stat, violating the structured CLI failure contract.
3. **Default receipt location — not repaired.** The current terminal receipt is under the explicit repair evidence root. With default evidence-root arguments, the shipped ledger returns `PUBLICATION_PENDING`, exit 2, and `blocked_claims=0` rather than passing the documented happy verification.

Security additionally reported non-blocking availability/robustness findings: unbounded ledger/receipt reads, analyze’s double-read race, traceback escapes for deep JSON/NUL output, and panel enumeration before its cap.

## Verification receipts

- Executor suite: `green-pytest.txt` — `53 passed in 1.33s`, exit 0.
- Independent QA: `review-qa.md` — 53 tests passed twice under hash seeds 101 and 202; 23/23 scenario groups passed.
- Goal review: `review-goal.md` — 53 tests passed independently under hash seed 17.
- Code-quality review: `review-code-quality.md` — 53 tests passed twice.
- Context and security reviewers also independently ran all 53 tests successfully.
- Executor manual QA: `manual-qa-summary.json` — all 9 classes passed.
- CLI command receipts: `generated-commands.txt` and `manual-qa-commands.json`.
- Changed-source LSP receipt: `lsp-diagnostics.txt`; independent QA also ran Pyright 1.1.411 with zero errors, warnings, or diagnostics.

No implementation test, real CLI route, or executor QA class remains pending. Only the reported blocking fixes and a later independent re-review remain; neither was begun here.

## Generated artifact and source binding

- Attempt: `af0d67ded56848369a2e21a0f5b4cbd2`.
- Ledger: `DAN/nonlatent_iclr/evidence_ledger.json` — SHA-256 `901e21579e5ff4f77011611485ab637235b4c0662d2440f4851dab2e4c3755eb`.
- Audit: `DAN/nonlatent_iclr/evidence_audit.md` — SHA-256 `e58e004a658a39aaf3e21cbfb75403c731e15b0512e71252faed7313e43ea90b`.
- Repair-root receipt SHA-256: `11e2423af9416566ed151b10c7ba14b8b34d5a3918aa0d4f8027fe38648d9891`; its ledger digest matches.
- Generated universe: 32 SHA-256 identities = 16 snapshot + 13 panel + 3 metadata; eight historical claims remain task-2 blocked.
- Exact final source/test/generated hash binding: `changed-file-sha256.txt`. Reviewers made no source changes, so that binding remains current.

## Exact source, test, and generated-output change list

```text
scale/experiments/nonlatent_iclr/__init__.py
scale/experiments/nonlatent_iclr/cli.py
scale/experiments/nonlatent_iclr/claims.py
scale/experiments/nonlatent_iclr/inventory.py
scale/experiments/nonlatent_iclr/models.py
scale/experiments/nonlatent_iclr/publication.py
scale/experiments/nonlatent_iclr/schema.py
scale/experiments/nonlatent_iclr/service.py
scale/experiments/nonlatent_iclr/pyrightconfig.json
scale/tests/nonlatent_iclr/test_audit_cli.py
scale/tests/nonlatent_iclr/test_cli_contract.py
scale/tests/nonlatent_iclr/test_publication.py
scale/tests/nonlatent_iclr/test_schema_contract.py
scale/tests/nonlatent_iclr/test_service_inventory.py
scale/tests/nonlatent_iclr/pyrightconfig.json
DAN/nonlatent_iclr/evidence_ledger.json
DAN/nonlatent_iclr/evidence_audit.md
```

Reporting additionally appended `.omo/notepads/nonlatent-rnn-dlm-iclr-research/learnings.md` and `issues.md`, and created the receipts under this repair evidence root. The pre-repair ledger/audit remain archived under `archive-before-repair/`.

## Public CLI interface for downstream workers

Repository-root interface:

```text
python -m scale.experiments.nonlatent_iclr.cli prepare --task 1 [root options]
python -m scale.experiments.nonlatent_iclr.cli analyze --task 1 [root options]
python -m scale.experiments.nonlatent_iclr.cli verify --task 1 --case happy|failure [root options]
python -m scale.experiments.nonlatent_iclr.cli render --task 1 --out PATH [root options]
```

The equivalent plan-style invocation from `scale/` is `python -m experiments.nonlatent_iclr.cli ...`. Root options are `--repo-root`, `--evidence-root`, `--external-root`, `--live-root`, and `--staged-root`. Commands emit sorted, indented JSON with exactly `blocked_claims`, `detail`, and `status`; normal failures use exit 2. Downstream consumers must currently pass `--evidence-root <repair-final-20260911-a>` to verify the regenerated publication successfully.

## Cleanup and constraints

- Manual QA fixtures were removed.
- Reviewer-owned `/tmp/opencode/nlat-context-review`, `/tmp/opencode/niclr-review-probe`, and `/tmp/opencode/secprobe` were removed after all lanes completed; QA and goal reviewers had already removed their own fixtures.
- No plan-checkbox, Boulder, shared-ledger, git, GPU, qz, scheduler, task-2, corpus-download, credential, immutable v7 snapshot, or unrelated shell-script change was made.
