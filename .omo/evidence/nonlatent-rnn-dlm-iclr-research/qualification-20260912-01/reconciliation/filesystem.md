# Filesystem reconciliation — exact reserved qualification output

Observation time: `2026-09-12T08:43:42Z`.

## Exact paths

- Reserved shared output: `/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/qualification-20260912-01-7bef6cf8-fda4-46b9-84ff-d7ca4fe5b892`
  - Exact-path `Read` and `stat` both returned `No such file or directory`.
- Corresponding controller receipt: `/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/controller-receipts/qualification-20260912-01-7bef6cf8-fda4-46b9-84ff-d7ca4fe5b892.json`
  - This is the exact filename bound by `submission/controller-state.json:1`, `submission/exact_job_spec.json:1`, and `submission/sanitized-execution-receipt.json:48`.
  - Exact-path `Read` and `stat` both returned `No such file or directory`.

The shared-output directory was absent, so there were no launcher, preflight, rank, aggregate, boot, or terminal/timeout files at that exact path to inspect. The launcher would create the directory at `scale/experiments/nonlatent_iclr/qualification/run_qualification.sh:82-84`, after its initial argument/environment/source checks at lines 30-79. Once the directory exists, it reserves `runtime-manifest.json`, `preflight.json`, `aggregate.json`, and `launcher.json` at lines 87-93; failures after trap installation are intended to publish a terminal `launcher.json` at lines 129-143.

## What this proves

- Shared output does **not** prove that the payload started or stopped. It contains no start, preflight, rank, aggregate, boot, or terminal status observation.
- Absence means only that no output directory was present at the observation time. It does not prove non-admission: a job could have remained queued, failed before launcher execution, or exited during the launcher's checks before output creation.
- No timeout terminal receipt was available to inspect because no payload-start artifact was present.
- No actual scheduler job ID is present in this filesystem evidence. The reserved run ID/job name is not a scheduler job ID and is not substituted for one.

## Bound identities from retained local evidence

- Attempt time: `2026-09-12T08:30:43Z` (`submission/request-attempt.json:1`); file mtime `2026-09-12 08:30:43.642277518 +0000`.
- Run ID/job name: `qualification-20260912-01-7bef6cf8-fda4-46b9-84ff-d7ca4fe5b892`.
- Nonce: `d75f2419e63b1c052beda405b5c7d963`.
- Submitted request SHA-256: `437dd1af38b91713c464b0b8c8430765ddf0fdd3c4fb592f51afa817daeb076b`.
- Source manifest SHA-256: `92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c`.
- Project/workspace/LCG/spec: `project-160ccb20-98ab-4538-a847-01d1f83d5b0f` / `ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6` / `lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e` / `7166bd2e-6cbe-4bd9-be38-762d11003e7f` (`submission/exact_job_spec.json:1`).
- Sanitized outcome: `AMBIGUOUS_CREATE_OUTCOME_STOP_NO_RETRY`, actual job ID `null`, admission `UNKNOWN`, recorded `2026-09-12T08:31:57Z` (`submission/sanitized-execution-receipt.json:12-21,73-74`).

No post-hoc controller receipt was created and no runtime verdict was assigned.
