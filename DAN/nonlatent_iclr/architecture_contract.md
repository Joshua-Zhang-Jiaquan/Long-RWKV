# Non-latent architecture contract

`architecture_contract.json` and its receipt are a source-bound, machine-readable
partial task-3 artifact. They do not enable a model or training path.

## Source-backed facts and bounded observations

- `attn_fwd` and `attn_bwd` are separately constructed modules with copied
  initialization, not tied parameters.
- The recycled loop re-executes the same blocks. The trained inner call is
  32 base plus 16 recycled block passes (48), distinct from outer denoising
  function evaluations.
- The trainer's printed count is captured before optional attachments. The
  separately retained qualification-05 and lifecycle-v7 rank evidence binds
  the deployed checkpoint: step 4750, 1955 state tensors, 1953 parameter tensors,
  and 4,091,581,441 parameter elements. No nominal 2.9B label substitutes for this.
- Installed FLA source is version- and SHA-256-bound without importing CUDA. That
  inspection is not a cache-kernel execution or equivalence proof.

## CPU coverage and limits

The tiny CPU harness checks real `torch.nn.Module`/parameter identities,
copied initialization, re-executed shared loop modules, session reset, immutable
prefix carry, and canvas-edit invalidation. It is engineering evidence only;
it is not proof of real FLA kernel-cache equivalence.

The canonical `cpu_source_evidence` separately binds 72 passing CPU tests (23
trainer, 49 initialization) to historical source, extraction modules and tests.
This covers source-extracted optimizer grouping/gradient policy/mask loss,
fusion initialization, zero-loop-gate identity and 32+16 block calls versus
outer denoising NFE. It is not an active-model training qualification.

## Lifecycle-v7 reconciliation (2026-09-14)

`runtime_evidence` retains qualification-05 unchanged in scope. The separate
`lifecycle_v7_evidence` pins the accepted v7 review and ledger, request, job,
run, nonce, manifest, raw controller receipt and canonical controller binding.
Eight rank records and eight lifecycle sidecars are hash-checked and reparsed;
derived predicates require exact device cuda:N and 12 calls/12 synchronizations
per rank. The bounded no-cache/full-canvas/nonlatent lifecycle establishes
session/foreign-request isolation, edit invalidation, close/reset/reopen
retirement and synchronized caller-boundary-exception recovery. It does not
establish CUDA-fault recovery or universal determinism.

Observed parameter categories (tensors / elements), identical on all eight ranks:

| Category | Tensors | Elements |
|---|---:|---:|
| Forward attention | 829 | 934,049,280 |
| Backward attention | 829 | 934,049,280 |
| Fusion | 64 | 209,797,120 |
| Shared | 230 | 2,013,685,760 |
| Loop | 1 | 1 |

## Semantics-v8 reconciliation (2026-09-14)

The separate `semantics_v8_evidence` binds review
`4996f460a0ac6befe2f42e6a17b15a555d938f65d352faebe6534087b923b6f0`
and review ledger
`5c5bb4f6dd3384c968512b0d8de69576847d410a31f6deb4075c06e307965427`.
All 29 worker records are hash-checked, including eight core ranks, eight
lifecycle sidecars and eight model-semantics sidecars. Authoritative lifecycle
and semantics predicates are reparsed and evaluated, with exact cuda:N,
checkpoint/parameter/controller identities and the 126-entry manifest binding.
Neither qualification-05 nor lifecycle-v7 is replaced or reinterpreted.

## Task3 acceptance at the tested scope

- Active/model-bound optimizer membership and gradient policy: **satisfied for
  two nonlatent freeze configurations and stage A only**. Baseline groups have
  1310/643 tensors (4,089,937,920/1,643,521 elements); frozen configuration has two
  empty groups. Stage-A observations include 900 gated nonzero gradients and
  733 ungated nonzero gradients after gating; None gradients remain None.
- Active input/output mask-loss semantics: **satisfied for tested corruption and
  rank-local actual-logit loss/backward**. Eligibility is 46 tokens, selection 22;
  loss matches selected mean and excludes unselected loss inputs within
  rel_tol=1e-6 / abs_tol=2e-6. Empty-mask loss is zero and unselected logit gradients
  are zero. Visible context can still influence selected bidirectional logits.
- Full-model FFN prefix cache and tied-loop prefix cache: **unsupported and
  unqualified**; no implementation required while disabled/unclaimed. Full-canvas
  and streaming interfaces are distinct; standalone attention-cache equivalence
  is not full-model cache qualification.

Nonlatent path, bounded lifecycle/session isolation, observed category counts,
source-bound initialization and block/NFE clock separation are satisfied at
their stated scopes. The two previously OPEN empirical acceptance fields are now
satisfied at the above bounded scopes. Plan completion is left to the coordinator;
this reconciliation does not change its checkbox. Publication returns
`BLOCKED_EXTERNAL` with `whole_architecture_ready=false`; `ENGINEERING_ONLY`
still denotes CPU controls, not the separately identified GPU observations.
Four unqualified claim entries remain: the two caches, long-context/performance,
and image/binary identity. They are retained scope limitations, not additional
empirical Task3 work. Later gradient stages, optimizer updates/state serialization,
distributed/FSDP reduction and general training/convergence remain unclaimed.
Zero gradients do not guarantee AdamW weight immobility; empty frozen groups are
not useful nonlatent training.

Root integration may call `prepare(repo_root)`, `verify(repo_root)`, and
`analyze(repo_root)` from `scale.experiments.nonlatent_iclr.architecture_task`.
