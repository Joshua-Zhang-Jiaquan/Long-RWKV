# Qualification06 actual adapter-v6 GPU runtime review

## Verdict: PASS — bounded partial milestone only

Scheduler job `job-d10874d4-7a9e-4c36-9fe0-401b699f8698` reached `job_succeeded`. Scientific status remains separately and correctly recorded as `PARTIAL_QUALIFICATION`.

The request SHA-256 is `486a2602130b95e1559b720bd0b0e42c8d738cd9a2e03b4b0fb6fdba1f98c2e4`. Its fresh run id and nonce are bound through the exact job spec, controller receipt, launcher, aggregate, and all rank reports; each occurs in only one controller receipt. Submission records one CreateJob call, zero retries, and zero resubmissions.

All 14 worker-output ledger entries and 25 live-execution ledger entries verified. The eight distinct rank reports cover ranks 0–7 at world size 8. Every report binds the same job, request, run, nonce, 111-entry manifest, checkpoint step 4750, checkpoint SHA-256 `ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07`, 1,955 checkpoint tensors, and 4,091,581,441 model parameters. Every rank produced finite logits with shape `[1,48,65536]`.

The deployed 111-entry manifest binds `model_checks.py` at SHA-256 `6ddb94ddffc01d0fa416dcd7b7bfc0cc9dde201a864ccaf84e338201ab7ec023` and approved `full_canvas.py` at SHA-256 `1e5400e4283744d6c29ce533f0138b2b3e2b7bcd78b1a64b668a79f97c6b3e22`. The actual manifested `masked_forward` imports `FullCanvasAdapter`, calls `open` and `forward`, and calls `reset` in `finally`. Combined with all-rank runtime success, this establishes a successful actual adapter-backed GPU forward, not merely a CPU recording spy.

Torch `2.8.0a0+5228986c39.nv25.6` from distribution metadata and `2.8.0a0+5228986c39.nv25.06` from `torch.__version__` normalize to the same PEP 440 value under the previously established extraction reasoning. Raw-string equality is not required, and this review does not claim arbitrary Torch binary identity.

`GetJobLog` returned transport exit 0 with business `InternalError`; GPFS receipts and hash-bound outputs are therefore the runtime evidence. The requested image tag and scheduler image id are recorded, but the observed image digest remains unknown.

## Preserved limits

The successful path does not test injected GPU failures, runtime reset after such failures, GPU edit lifecycle, or cross-session behavior. Source placement of reset in `finally` is structurally verified, but those failure/lifecycle scenarios remain untested on GPU.

Full-model prefix caching remains `NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED`. Loop prefix caching remains `NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES`. There is no full Task3, long-context, performance, goodput, or image-binary claim.

## Next Task3 prerequisite

Wire and qualify FFN token-shift state for full-model prefix caching and prove tied-layer state-alias equivalence for loop caching under fresh identity-bound GPU evidence. Include bounded failure/reset, edit-lifecycle, and cross-session GPU scenarios before a full Task3 claim.
