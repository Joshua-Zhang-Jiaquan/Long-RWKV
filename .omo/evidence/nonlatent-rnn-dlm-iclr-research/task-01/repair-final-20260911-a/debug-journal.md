# Task-01 repair debug journal

## 2026-09-11 intake

### Symptom

The partial repair still lacks strict boundary validation, complete origin-aware freshness, recoverable publication, dynamic view semantics, complete provenance, and verified CLI containment.

### Hypothesis matrix

| ID | Hypothesis | Prediction | Disproof condition | Planned probe |
|---|---|---|---|---|
| H1 | Schema validation checks shapes but not exact identity/coverage semantics. | Duplicate required paths, wrong sources/kinds, empty claims/views, and malformed receipt fields can pass. | Every adversarial digest-consistent ledger and receipt is rejected with a precise structured status. | Add isolated schema/receipt mutations before source repair. |
| H2 | Freshness has no serialized origin model. | External metadata and raw panels are skipped or checked against the snapshot root. | Mutating every hashed artifact at its declared source yields `STALE`. | Add explicit external-root fixtures and source-drift tests. |
| H3 | Receipt-first publication can strand a terminal-looking receipt. | A failure between receipt and ledger causes an orphan that cannot be safely reconciled. | Retry deterministically recovers without overwrite or contradictory state. | Inject each publication-boundary failure and retry. |
| H4 | Filesystem checks are race-prone and insufficiently contained. | Symlink swaps or destinations outside approved roots can be accepted. | All path escape/symlink/contention probes fail closed without external writes. | Add output-root containment and concurrent write tests. |
| H5 | Inventory/claims/views are presentation-level approximations. | Panel details are not bounded shard identities, claims do not cite records/fields, and view state/detail can conflict. | Generated ledger contains exact bounded identities and dynamically derived claim/view evidence. | Assert generated semantic records, counts, hashes, and view branches. |

### First observations

- Baseline suite: 13/13 passed before repair.
- Schema RED: 8 failed and 6 passed. H1 is supported: the current validator rejects the desired closed contract but lacks tuple identity, duplicate coverage, required claims/views, duplicate-key parsing, and receipt validation.
- Service RED: 11/11 failed at the missing explicit-root constructor boundary. H2 and H5 are supported: source origins are global/hardcoded rather than ledger-bound and injectable, preventing complete external freshness and dynamic-view proofs.
- Publication RED: 9/9 failed. H3 and H4 are supported: publication has no root-relative containment contract, typed unsafe-path failure, or ledger-first recoverable transaction interface.
- CLI RED: 4/4 failed. The CLI exposes neither explicit source-root options nor structured parser failures, and therefore cannot prove command completeness or render containment.

### Root cause checkpoint

The failures share two architectural causes rather than isolated branches: the ledger has no serialized source-origin contract, and publication accepts a target path instead of an approved root plus relative destination. Repair will replace these boundaries first, then route service and CLI behavior through them.

### Repair outcome

- Closed ledger and receipt boundaries now reject duplicate JSON keys, wrong scalar types, wrong tuple identity, duplicate coverage, missing claims/views, and invalid source roots.
- Artifact inventory now binds explicit roots, hashes exact named panels through bounded manifests, records metadata identities, and verifies every serialized origin.
- Publication now writes the ledger first, reports an orphan as `PUBLICATION_PENDING`, and can safely complete its immutable receipt on retry.
- Root-relative descriptor traversal rejects escapes and symlink parents; concurrent writers cannot overwrite a winner.
- The focused task suite is GREEN: 53/53 passed.

### Constraints

- Do not modify `DAN/v7_arch_round/` or the unrelated concurrent shell-script change.
- Do not run GPU, scheduler, corpus-download, git commit, plan-checkbox, Boulder, or shared-start-ledger operations.
- Add regression proofs before production repair and preserve all outputs under this evidence directory.
