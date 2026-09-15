# Independent bounded runtime review

## Verdict

**REJECT** promotion of the `BOUNDED_PARTIAL_QUALIFICATION_MILESTONE_ONLY` from this evidence package. Task3 remains incomplete.

The execution identity is coherent and 22/22 declared source hashes verify. Eight unique ranks (`0..7`) report world size 8 on eight H100 80GB GPUs. The checkpoint evidence is step 4750, 1,955 state tensors, 4,091,581,441 parameters, and checkpoint SHA-256 `ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07`. Every rank reports finite masked logits with shape `[1,48,65536]`; standalone FLA cache evidence reports prefix 32, sequence 48, and maximum absolute delta 0.0 under tolerance 0.05.

The blocking defect is an exact runtime-environment identity mismatch. `preflight.json` and `runtime_environment.json` record torch `2.8.0a0+5228986c39.nv25.6`, while every rank receipt records `2.8.0a0+5228986c39.nv25.06`. Therefore the asserted exact `MATCH` is not independently established. Under fail-fast review, the evidence-quality, security/integrity, and hands-on QA passes failed; goal/constraint and historical-context reviewers passed only the bounded interpretation.

## Bound identity

- Job: `job-724fd7b8-97f8-43b6-bac5-bab67eae314b`
- Run: `qualification-20260912-01-91f32832-f583-4638-aedc-be4ac9887ab3`
- Nonce: `a41af12290a62f67829670728be9b480`
- Request SHA-256: `55df1dfa714dd54c4b642af5b5cda7caee76946d57e09512d9f7d492fb3fa919`
- Manifest SHA-256: `9271a2036e84c3fa29442a9f7627c9889d024106b8286321d82da26df856e51d`
- Controller receipt SHA-256: `4b4e2655e6e833decb32411357c5bbc04d89fd68f0698e4bf4115604feeafd91`
- Monitoring ledger: `submission/monitoring-v6/evidence-sha256.txt` (9/9 verified)
- Worker ledger: `submission/monitoring-v6/worker-output-hashes.sha256` (13/13 verified)

## Preserved limitations

- Aggregate scientific status remains `PARTIAL_QUALIFICATION`; scheduler `job_succeeded` is operational evidence only.
- Full-model prefix cache remains `NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED`.
- Loop prefix cache remains `NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES`.
- The observed image digest is unknown.
- `GetJobLog` returned business `InternalError`, so scheduler-log corroboration is unavailable.
- Source hash ledgers are unsigned and do not themselves provide authenticated provenance.
- No full-model cache, long-context, performance/goodput, billing, or exact-duration claim follows from these short checks.

## Exact next prerequisite

Produce an immutable, identity-bound environment reconciliation receipt that either corrects the torch version evidence or defines and justifies a canonicalization proving `2.8.0a0+5228986c39.nv25.6` and `2.8.0a0+5228986c39.nv25.06` equivalent. Bind that receipt and any regenerated aggregate/monitoring outputs with new SHA-256 ledgers, then repeat this bounded independent review.
