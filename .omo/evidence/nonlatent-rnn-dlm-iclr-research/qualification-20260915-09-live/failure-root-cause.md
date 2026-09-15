# Root cause: job-f8b209ca-50b1-4e88-a9f5-0d712f8c56a0 failed at launcher startup

## Observed
- Job `job-f8b209ca-50b1-4e88-a9f5-0d712f8c56a0` (run `qualification-20260912-01-229d6978-f899-4b7c-8d8a-5f316a0d2563`) reported `job_failed` after `running_time_ms=6000`.
- Timeline: created 1789454872000, `resource_prepared` 1789454955000, `run` 1789454955000, finished 1789454961000 — so a container was assigned and ran about four seconds.
- Instance record: node `qb-prod-gpu187`, `instance_failed`, `running_time_ms=4000`.
- **No output directory was created** under `/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/<run-id>`, so the launcher died before its first `mkdir`.
- `GetJobLog` returned business `InternalError` again, so no pod log was available; the diagnosis was made by reproducing the launcher's early path locally.

## Root cause (reproduced)
The v9 launcher computed its staged root from the **qualification directory** instead of the payload root:

    readonly DEFAULT_STAGED_ROOT="${PAYLOAD_DIR}/model_source/scale"

`PAYLOAD_DIR` is `dirname(run_qualification.sh)` = `.../payloads/<payload>/scale/experiments/nonlatent_iclr/qualification`, so the derived path was `.../qualification/model_source/scale`, which does not exist. The launcher's own bytecode guard rejected it before any output was written:

    BYTECODE_PRECHECK_REJECTED:invalid_owned_root:.../qualification/model_source/scale

Reproduced verbatim by running the v9 payload's launcher locally with the job's environment (dummy run identity, 45 s bound): exit 1, that exact message, and no output directory — the same signature as the failed job.

Earlier v8 payloads did not hit this because their launcher hard-coded an absolute staged-root path rather than deriving one. The derivation was introduced in this wave to avoid rewriting the launcher for each payload version; it was correct in shape (`${PAYLOAD_ROOT}/model_source/scale`) but used the wrong base variable.

## Fix
`readonly DEFAULT_STAGED_ROOT="${REPO_ROOT}/model_source/scale"`, where `REPO_ROOT` is already `cd "${PAYLOAD_DIR}/../../../.."` — the payload root that contains `model_source/scale`.

## Verification of the fix (before any further submission)
`stage_calibration_v9.py` now runs a **launcher startup probe** as part of staging: it executes the staged launcher locally with a dummy identity under a 20 s bound and asserts (a) the guard did not reject and (b) the output directory was created. For v10 this reports `guard_rejection=false`, `reached_runtime_stage=true`, transcript showing `BYTECODE_PRECHECK_CLEAN` followed by a fully `MATCH`-ing runtime environment record with all four roots present (`release`, `staged_source`, `checkpoint`, `model`) and all package versions matching. The probe's output directory is removed afterwards.

## Standing rule adopted
A payload is not submitted until its own launcher has cleared its startup guards locally. That check is cheap (seconds) and is the only thing that would have caught this class of defect before spending a job.

## What is NOT claimed
No measurement, model result, or capability claim came from the failed job. It produced zero evidence beyond the scheduler records preserved here. The v9 payload is left untouched as history; the corrected payload is v10 (`manifest_sha256 fcbd2a9195f1b4f8a97df16c1df3d564f294edd9b5b6354841169d03fd9891c1`).
