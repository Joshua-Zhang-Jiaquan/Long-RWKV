# Independent one-shot submission-preflight command record

Working directory unless noted:

`/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY`

This review made zero qz, login, network, CUDA, git, install, or submit-command calls. It did not repeat the already-recorded dry-run.

## Raw request and authorization hashes

```bash
sha256sum \
  .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-01/submission/exact_job_spec.json \
  .omo/authorizations/nonlatent-h100-qualification-20260912-01.json
```

Observed:

```text
437dd1af38b91713c464b0b8c8430765ddf0fdd3c4fb592f51afa817daeb076b  exact_job_spec.json
e7af34a994015297c7eff4e209fff6029cdc47de8c6f3456a7bc108539de0f82  nonlatent-h100-qualification-20260912-01.json
```

A standard-library Python assertion command parsed the exact 1,693 request bytes and current authorization, then checked project/workspace/LCG/spec/image, one instance, eight GPUs implied by the authorized resolved spec, four derived GPU-hours, `"1800000"`, disabled fault tolerance, zero retry/reservation, exact launcher-only argv, and the exact eight-entry environment. Output:

```json
{"request_sha256":"437dd1af38b91713c464b0b8c8430765ddf0fdd3c4fb592f51afa817daeb076b","bytes":1693,"nodes":1,"gpus":8,"gpu_hours":4,"runtime_ms":"1800000","launcher_only":true,"required_env_exact":true,"no_retry_or_reserve":true}
```

## Bounded controller tests with qz unavailable

From the submission directory:

```bash
ls /tmp/opencode >/dev/null && \
tmp_dir="$(mktemp -d /tmp/opencode/submission-preflight-tests.XXXXXX)" && \
trap 'rm -rf "$tmp_dir"' EXIT && \
PATH=/nonexistent TMPDIR="$tmp_dir" PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python -m pytest -q -p no:cacheprovider \
  test_controller.py test_submission_contract.py
```

Result:

```text
Running 18 items in this shard
.................. [100%]
18 passed in 1.07s
```

Because `PATH=/nonexistent`, no qz program could be resolved. Tests involving CreateJob behavior used `_FixtureRunner`; reserve subprocesses invoked only the absolute current Python executable.

## Supplemental fake-runner sequence probe

An all-temporary `/usr/bin/python -c` harness copied the reservation/request/dry-run receipt into an EXIT-trapped `/tmp/opencode/submission-sequence-probe.XXXXXX` directory, changed only a temporary authorization copy to PASS, replaced `controller_runtime.run_qz` with a fake runner, and replaced `publish_receipt` with a fake that asserted `admitted-job.json` already existed. It then tested a second call, a nonzero fake result, and a successful response containing two job IDs. `PATH=/nonexistent` and `PYTHONDONTWRITEBYTECODE=1` remained set. Output:

```json
{"fake_create_calls":1,"attempt_preceded_call":true,"actual_job_id_not_run_name":true,"admission_preceded_receipt":true,"repeat_refused":true,"generic_rejection_calls":1,"generic_rejection_retry_refused":true,"ambiguous_response_calls":1,"ambiguous_response_retry_refused":true,"gpfs_publish_replaced_by_fake":true}
```

No GPFS destination was written by this probe.

## Root PASS compatibility probe

An EXIT-trapped temporary copy changed only `preflight_verdict` to `PASS` and status to ready, loaded it through the real authorization model, loaded the existing reservation, and rebuilt the request. It asserted the rebuilt bytes equal `exact_job_spec.json`, the temporary updated authorization hash differs from the historical state hash, and changing `maximum_nodes` to two fails Pydantic literal validation. Corrected output:

```json
{"root_pass_update_accepted":true,"request_bytes_unchanged":true,"historical_state_auth_hash_not_current_gate":true,"immutable_tuple_change_rejected":true,"temporary_updated_auth_sha256":"388a924ef41278dc66b2d5664227e4a5a50e4ec671206bd0e32584d22c491780"}
```

The first draft of this reviewer-only probe exited 1 because it looked for non-reexported `controller.Authorization`; the corrected command imported `Authorization` from `controller_models` and passed. Both temporary directories were removed by their EXIT traps; this was a probe typo, not a controller defect.

## Frozen source/input receipts

From the submission directory:

```bash
sha256sum -c artifact-hashes.sha256
```

All 16 listed controller artifacts reported `OK`.

From the repository root:

```bash
sha256sum -c \
  .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-01/submission/input-hashes.sha256
```

All seven current pre-root-update inputs reported `OK`, including unchanged payload-final2 contracts/source and manifest. After the deliberate root PASS edit, only the authorization line becomes historical; the controller does not use this receipt as a live gate.

## Prior dry-run receipt read only

The existing receipt records one literal dry-run, qz exit 0, zero valid job IDs, request hash `437dd1af...`, `admission_verified: false`, and `capacity_verified: false`. This review did not invoke dry-run or any qz command.

## Submission-artifact absence

A standard-library `os.path.lexists` assertion checked the two local markers and exact GPFS receipt destination, then cross-checked authorization/reservation state. Output:

```json
{"paths_present":{"request_attempt":false,"admitted_job":false,"controller_receipt":false},"authorization_jobs_submitted":0,"authorization_gpu_hours_consumed":0,"controller_status":"NOT_SUBMITTED","reservation_count":1}
```

## Static diagnostics

`lsp_diagnostics` was requested independently for `controller.py`, `controller_models.py`, `controller_contract.py`, `controller_storage.py`, and `controller_runtime.py`; every file returned `No diagnostics found`.
