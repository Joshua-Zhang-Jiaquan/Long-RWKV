# Independent task-01 review

## Verdict

**AdversarialVerify: `needs-fix` (high confidence).**

The nominal six tests and normal CLI flow pass, and the 16 fixed in-repository R1-R6 entries in the real ledger match their current files. Task 1 is nevertheless not complete: ordinary digest-consistent malformed ledgers can pass as fresh, required inventory can disappear, raw endpoint discovery does not name most real panels, existing metadata is not inspected, and a receipt-write interruption leaves a permanently unverifiable active ledger.

This is an offline data-publication consistency review. It used no network, GPU, scheduler, credentials, git, package installation, old launcher execution, or model-binary reads.

## Commands and observed statuses

All runtime commands used `CUDA_VISIBLE_DEVICES=""` and isolated copies under `/tmp/opencode/nonlatent-av-gpt56sol-20260911-a/`. That temporary root was removed after the probes.

### Tests

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=101 python3 -m pytest -q -p no:cacheprovider tests/nonlatent_iclr/test_audit_cli.py
exit 0: 6 passed in 0.43s

PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=202 python3 -m pytest -q -p no:cacheprovider tests/nonlatent_iclr/test_audit_cli.py
exit 0: 6 passed in 0.42s
```

The two no-install `uv run --no-project --offline ... pytest` attempts each exited 1 because uv selected a managed Python without pytest (`No module named pytest`). The already-installed system Python had pytest 8.1.1 and produced the two passing runs above. See `pytest-*.log` and `pytest-*-runs.json`.

### Real-v7-copy CLI flow

```text
python3 -m experiments.nonlatent_iclr.cli --help
exit 0; listed prepare, analyze, verify, render

... prepare --task 1 --repo-root <copy> --evidence-root <temp-evidence>
exit 0; AUDIT_COMPLETE; blocked_claims=11

... verify --task 1 --case happy ...
exit 0; AUDIT_COMPLETE; "fresh audit"; blocked_claims=11

... verify --task 1 --case failure ...
exit 0; EXPECTED_FAILURE_CONFIRMED

... analyze --task 1 ...
exit 0; AUDIT_COMPLETE; blocked_claims=11

... render --out <copy>/DAN/nonlatent_iclr/evidence_audit.md ...
exit 0; AUDIT_COMPLETE; output 8,872 bytes; sha256 85930b74e730ceb32f91a05430808c552511333c96c202f0324b9fd6711d7a5e

... prepare --task 2 ...
exit 2; UNSUPPORTED; "task 2 is not implemented"
```

The happy result means ledger/receipt/source freshness only. On the real ledger it retains 11 unresolved claims and does **not** certify positive model science.

## Confirmed defects

### 1. Required schema and coverage are not verified

Each mutation was made only in an isolated copy, serialized as valid JSON, and paired with a recomputed receipt digest. This does not ask for protection against a privileged rewrite; it tests whether a digest-consistent ledger is structurally and semantically valid.

Observed `verify --case happy` outcomes:

| Mutation | Exit/status |
| --- | --- |
| `schema_version = 2` | `0 / AUDIT_COMPLETE` |
| `schema_version = "1"` | `0 / AUDIT_COMPLETE` |
| unknown artifact identity | `0 / AUDIT_COMPLETE` |
| `identity="sha256"`, digest `null` | `0 / AUDIT_COMPLETE` |
| non-string path on an unverified artifact | `0 / AUDIT_COMPLETE` |
| numeric accounting note | `0 / AUDIT_COMPLETE` |
| string artifact byte count | `0 / AUDIT_COMPLETE` |
| numeric claim detail | `0 / AUDIT_COMPLETE` |
| remove required `report.md` artifact and its generated claim | `0 / AUDIT_COMPLETE` |
| empty `artifacts=[]` and `claims=[]` | `0 / AUDIT_COMPLETE`, `blocked_claims=0` |
| artifact list item is a scalar | `2 / MALFORMED` |
| malformed digest text | `2 / STALE`, not schema-invalid |
| numeric attempt | exit `1`, uncaught `TypeError` |
| numeric path on a sha256 artifact | exit `1`, uncaught `TypeError` |

The empty-ledger result is confirmed incomplete-coverage acceptance and misleading success output. Evidence: `schema-probes.json`, `scalar-type-probes.json`, `empty-inventory-probe.json`.

### 2. Receipt-step interruption cannot recover

A local test double let the real ledger write succeed, then raised `OSError(5)` exactly when `_atomic_write` was called for `prepare_receipt.json`. This was repeated in two fresh fixtures.

Both repetitions produced the same state:

- initial call raised `OSError`;
- ledger exists, receipt does not;
- first retry: exit `2 / PUBLICATION_EXISTS`;
- happy verify: exit `2 / TAMPERED`;
- second retry: exit `2 / PUBLICATION_EXISTS`.

No repetition recovered. Both left an unverifiable active publication. Evidence: `interruption-probes.json`.

### 3. Raw-record discovery misses available endpoint panels

The actual R3 script uses f-strings. The generated ledger records these paths:

```text
cap_lm_m4loop_s
cap_lm_m4loop_s4750
cap_lm_n2_
cap_lm_n2_s
m2_baseline_triangle/ability_curve_m4loop.csv
sampler_gate_m4_loop_s
sampler_gate_m4_loop_s4750
sampler_gate_m4_n2_
```

Only three are exact expected locations. Ten exact panel names are absent, including both N2 brackets and every reps2/3/4 cap-LM and sampler panel. All 13 expected locations were observed present by bounded directory/file metadata checks; the cap-LM panels each held 40 matching shard files and the sampler panels each held 8. Raw contents were not bulk-read. Evidence: `real-ledger-audit.json`, `discovery-metadata-views.json`.

### 4. Generated metadata does not inspect availability

All three configured metadata JSON files were regular and small enough to hash, but the ledger unconditionally records `identity="unverified"`, `bytes=null`, and `sha256=null`:

| File | Observed bytes | Independently observed sha256 |
| --- | ---: | --- |
| loop step-4750 `meta.json` | 43 | `321ce0465b2cedb13c870c1ad4ec2c7c36d14cd37670e9caa018b9e63954e7ef` |
| N2 step-6000 `meta.json` | 43 | `ffd02c77be71b72b28413e3ab58cff63fbc158239e91b0a1c820b304192b24e2` |
| tokenizer `tokenizer_config.json` | 1,101 | `4e03aa0f5d6b1a4006a0d9e9f070f01418e734ab17c7c48f28333596d6bd5e29` |

A controlled existing path and missing path received identical generated metadata. This confirms availability is not inspected. No model binary was opened. Evidence: `discovery-metadata-views.json`.

### 5. Default claims do not satisfy task-01 result-level accounting

The real ledger is internally digest-consistent and contains 27 artifacts and 29 claims (`observed=16`, `unresolved=11`, `claimed=1`, `corrected=1`). Its 16 fixed local inventory entries all match file hashes and sizes.

However, claim details are artifact identity strings plus two generic statements. None names the plan's cited historical result/interpretation anchors: `LM1B`, `WikiText103`, `max_run_frac`, masked-token accuracy, `tau`, `r100`, or the 4.98B exposure observation. Task 2 owns metric recomputation and correction; task 1 still requires each cited result to have provenance and an interpretation status. That task-01 mapping is unsupported.

The rendered audit is only a heading plus the ledger JSON. It adds no missing claim mapping. Evidence: `real-ledger-audit.json`.

### 6. Source-root and publication invariants have ordinary consistency gaps

- After preparation, replacing the copied `DAN/v7_arch_round` directory with an ancestor symlink to the same bytes outside the copied repo still returned `0 / AUDIT_COMPLETE` without changing the receipt.
- Digest-consistent absolute and `../..` artifact paths outside the source root also returned `0 / AUDIT_COMPLETE`.
- A direct artifact symlink was correctly rejected (`2 / STALE`), showing the gap is specifically ancestor/path containment.
- A controlled destination created after `_atomic_write`'s existence check but immediately before its real `os.replace` was overwritten. `_atomic_write` returned true and final bytes were the publisher's, not the contender's.

Evidence: `containment-probes.json`, `atomic-write-race.json`.

### 7. View labels are not consistently derived from paths

The current real paths happened to align with states (`snapshot=observed`, absent local live mirror=`corrected`, present staged mirror=`observed`). Controlled ordinary path states expose hardcoded labels:

- absent snapshot path still yields `observed: DAN/v7_arch_round is bound by hashes`;
- present `scale/models` yields state `observed` but detail `local scale code mirror absent`.

Evidence: `discovery-metadata-views.json`.

### 8. Static diagnostic remains open

Pyright reports no diagnostics for `service.py` or `cli.py`. It reports `reportMissingImports` at test line 12 for `experiments.nonlatent_iclr.service`. `scale/` has no project configuration; runtime `sys.path` mutation lets pytest pass but does not clear the diagnostic. `basedpyright` and `ruff` are unavailable and were not installed. Evidence: `static-diagnostics.md`.

## Adversarial cases and scope

- `stale_state`: changed checksum and missing saved panel were rejected in the isolated failure CLI (`EXPECTED_FAILURE_CONFIRMED`).
- `dirty_worktree preservation`: no git command was used. Before/after filesystem manifests preserve original hashes. A concurrent change was observed in `DAN/v7_arch_round/scripts/run_prune_and_launch_ext.sh`; this review did not touch or revert it (details below).
- `misleading_success_output`: confirmed by the empty digest-consistent ledger returning `AUDIT_COMPLETE`, `blocked_claims=0`.
- `malformed_input`: exercised as listed above.
- `repeated_interruptions`: two identical receipt-step interruption runs; neither recovered.
- `flaky_tests`: six tests passed twice under distinct `PYTHONHASHSEED` values.
- inert source text: a controlled instruction-like string was read as text and not executed; the f-string endpoint was still truncated to `cap_lm_m4loop_s`.
- cancel/resume: the receipt-step interruption plus two retries exercised the applicable resume behavior. No command hung, so hung-command cancellation was not triggered.

## Source preservation and cleanup

`source_metadata.before.json` records 64 protected entries and 48 source-to-fixture hash comparisons; every copied file matched at fixture creation. Requested worker files, the production ledger/audit, plan, notepads, and all 16 ledger-bound R1-R6 files retained their baseline identities.

The final broad manifest observed one unrelated concurrent source change and its parent-directory mtime:

```text
DAN/v7_arch_round/scripts/run_prune_and_launch_ext.sh
before: size 3223, sha256 db73281f13fa84bb4bf86eead607f644f5fac099e1d26245a0555404cfda0092
after:  size 3742, sha256 2ec6cf2fa81f00fc9de438c23405da3db482cdcac9f776ef091b946114e11ab1
```

No review command targeted that path; filesystem metadata alone cannot attribute the concurrent writer, so the review neither claims global-tree immutability nor cleans up another session's change. See `source_metadata.after.json`, `no-product-write-receipt.json`, and `write-set-receipt.json`.

Cleanup command `rm -rf -- /tmp/opencode/nonlatent-av-gpt56sol-20260911-a` exited 0. Only review-owned temporary fixtures were removed. See `cleanup-receipt.json`.

## Exact repair priorities

1. **P0 — schema and coverage:** reject any schema version other than integer 1; parse every scalar/list/object field; restrict identity values; require a 64-hex digest for `sha256`; require the complete task-01 inventory and result-to-provenance/status mapping; return structured `MALFORMED` instead of tracebacks.
2. **P0 — recoverable publication:** do not expose the ledger as active before its receipt is durable. A receipt-step failure must leave a retryable draft or a stable explicit failure that can recover without an unverifiable immutable ledger.
3. **P0 — evidence discovery:** resolve the actual R3 f-string panel matrix to exact endpoint names and inspect each metadata JSON's observed existence/size/hash. Missing artifacts must remain explicit blockers.
4. **P1 — source/publication invariants:** reject absolute/traversal paths and ancestor-symlink source escapes; make no-overwrite publication atomic at the destination operation, not by a precheck followed by replacing rename.
5. **P1 — view/claim meanings:** derive view state and detail from the same observed path condition; distinguish audit freshness from task-01 completion and from positive science.
6. **P2 — tooling:** make the test import resolvable to the available type checker, then run the configured checker/linter when available without weakening runtime isolation.

No plan checkbox, Boulder state, product file, historical ledger, or v7 test source was intentionally modified by this review.
