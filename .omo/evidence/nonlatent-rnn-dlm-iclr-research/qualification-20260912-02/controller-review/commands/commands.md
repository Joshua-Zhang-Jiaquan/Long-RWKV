# Qualification 02 controller-review command record

Working directory unless noted:

`/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY`

No qz command, submit command, dry-run, ListJobs, login, network, CUDA, large-weight hash, git, or install operation was executed.

## Request/permit/authority bindings

```bash
sha256sum \
  .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-02/submission/exact_job_spec.json \
  .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-02/submission/permit.json \
  .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-02/submission/controller-state.json \
  .omo/authorizations/nonlatent-gpu-campaign-20260912.json
```

Observed:

```text
ece8b1bf041f31ee84bb9b400d21856c31c458770982b75568b7be31c6f0c42e  exact_job_spec.json
06e6f0c2afb7e1a2e84d2dbf4eaad7542b5604b7abc1dbb1ffe1fea493f546f9  permit.json
1bc1a0f9d6d0d837597c2fa29e6eb470c42569048d7ed98f4bbe5d8b5ea809dd  controller-state.json
067e063296ab8a738ec796afb3e92aa2732a0da261360285a591c84c05221b0a  nonlatent-gpu-campaign-20260912.json
```

A standard-library assertion command parsed these current bytes and checked the unlimited cumulative campaign, campaign-over-legacy authority mapping, request/permit/state hashes, fresh grant identity, exact project/workspace/LCG/spec/image, 1×8 H100 shape, 1,800,000 ms/four GPU-hours, one call, zero retry/reserve, launcher-only command, and exact eight-entry environment. Output:

```json
{"request_sha256":"ece8b1bf041f31ee84bb9b400d21856c31c458770982b75568b7be31c6f0c42e","request_bytes":1694,"run_id":"qualification-20260912-01-28575096-6ce0-4fd8-9e86-3e3804c86c14","grant_instance_id":"grant-qualification-20260912-02-2e9052ad-a45a-4e6d-868b-1dc2b3415039","campaign_unlimited_cumulative":true,"legacy_profile_authority":false,"nodes":1,"gpus":8,"runtime_ms":1800000,"gpu_hours":4,"one_call":true}
```

## Designated focused CPU suite

From the qualification-02 submission directory:

```bash
ls /tmp/opencode >/dev/null && \
tmp_dir="$(mktemp -d /tmp/opencode/controller02-tests.XXXXXX)" && \
trap 'rm -rf "$tmp_dir"' EXIT && \
PATH=/nonexistent TMPDIR="$tmp_dir" PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python -m pytest -q -p no:cacheprovider \
  test_response_capture.py test_controller_runtime.py \
  test_campaign_contract.py test_admission_parser.py
```

Result:

```text
Running 26 items in this shard
.......................... [100%]
26 passed in 0.35s
```

## Root review-gate transition probe

An EXIT-trapped `/tmp/opencode/controller02-root-flip.XXXXXX` copy changed only the four permitted review fields, rewired only temporary permit/state/request paths, loaded the real campaign/context, validated the exact request, and evaluated the real live gate with `ROOT_PASS_CONFIRMED`. A runtime-cap mutation was separately rejected by the strict permit model. No submit function or runner was called.

```json
{"allowed_root_fields":["controller_review_verdict","live_submission_verdict","prior_ambiguous_reconciliation_verdict","status"],"live_gate_passed":true,"request_bytes_unchanged":true,"grant_instance_unchanged":true,"immutable_per_job_change_rejected":true,"qz_calls":0}
```

## Attempt-01 history and fresh-artifact check

A standard-library read-only assertion checked the old marker/sanitized receipt, old controller source hashes, fresh permit/run/nonce/grant identity, all expected fresh submission/response/GetJob/receipt paths, and the private dry-run capture. Output:

```json
{"old_attempt_preserved":true,"old_create_calls":1,"old_admission":"UNKNOWN","old_source_unchanged":true,"fresh_run_id":"qualification-20260912-01-28575096-6ce0-4fd8-9e86-3e3804c86c14","fresh_nonce_unique":true,"fresh_grant_instance":"grant-qualification-20260912-02-2e9052ad-a45a-4e6d-868b-1dc2b3415039","fresh_submission_artifacts_present":false,"dry_run_capture_private_and_bound":true,"reconciliation_sha256":"d2185c73e85d72a375e5673dd60b21e45d8c122a586f65c56f1ecac907c35fdf"}
```

The observed 02:49 scheduler result came only from the already-persisted `reconciliation.md`; this review made no scheduler call.

## Prepared source receipts

From the qualification-02 submission directory:

```bash
sha256sum -c artifact-hashes.sha256
```

All 33 entries reported `OK`.

From the repository root:

```bash
sha256sum -c \
  .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-02/submission/input-hashes.sha256
```

All nine entries reported `OK`, including the immutable attempt-01 marker/receipt and unchanged payload contracts/source/manifest. The payload's prior 143-test suite was not rerun.

## Static diagnostics

`lsp_diagnostics` was requested separately for the 12 production controller modules and four focused test files listed in `cap-validation.json`; all 16 returned `No diagnostics found`.
