#!/usr/bin/env bash
# Direct qz submission for the calibration-v15 payload: warm-up inside the OOM guard with the
# ladder stopping there, a cache release between rungs, and no allocator override. (Task-6 throughput forecast input).
# Generates fresh run identity, submits exactly one CreateJob, and writes the binding receipt the
# worker launcher awaits. No retries. The governing authority for this job is the campaign
# authorization .omo/authorizations/nonlatent-gpu-campaign-20260912.json; the receipt's
# authorization_id field is a frozen literal of the job contract and is left unchanged, with the
# governing authority recorded separately in authority.json.
set -Eeuo pipefail

readonly ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification
readonly V15=${ROOT}/payloads/qualification-20260915-15-global-ffaa464d-calibration-v15
readonly EV=.omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260915-15-live
readonly MANIFEST_SHA=d22e139f4916dd06e0e19ea9e9d1f8d1c3318ce35e538c06f4520a76a61e4743
readonly IMAGE=docker.sii.shaipower.online/inspire-studio/relay2:v2

mkdir -p "${EV}"
RUN_ID="qualification-20260912-01-$(python3 -c 'import uuid;print(uuid.uuid4())')"
NONCE="$(python3 -c 'import secrets;print(secrets.token_hex(16))')"
RECEIPT="${ROOT}/controller-receipts/${RUN_ID}.json"
SPEC="${EV}/createjob-spec.json"

python3 - "$SPEC" "$RUN_ID" "$NONCE" "$MANIFEST_SHA" "$RECEIPT" "$V15" "$IMAGE" <<'PY'
import json, sys
spec, run_id, nonce, manifest_sha, receipt, v15, image = sys.argv[1:]
body = {
    "auto_fault_tolerance": False,
    "command": f"/usr/bin/bash {v15}/scale/experiments/nonlatent_iclr/qualification/run_qualification.sh",
    "description": "Direct-qz nonlatent calibration: 1x8 H100 SXM 80GB, 30 minutes, zero retries. Measures per-arm forward/backward/optimizer throughput for the Task-6 forecast ledger. Makes no long-context, goodput or quality claim.",
    "enable_notification": False,
    "enable_troubleshoot": False,
    "envs": [
        {"name": "QUALIFICATION_RUN_ID", "value": run_id},
        {"name": "QUALIFICATION_NONCE", "value": nonce},
        {"name": "QUALIFICATION_MANIFEST_SHA256", "value": manifest_sha},
        {"name": "QUALIFICATION_JOB_RECEIPT", "value": receipt},
        {"name": "QUALIFICATION_MODE", "value": "calibration"},
        {"name": "QUALIFICATION_CHECKPOINT_STEP", "value": "4750"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
        {"name": "HF_DATASETS_OFFLINE", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ],
    "fault_tolerance_max_retry": 0,
    "framework": "pytorch",
    "framework_config": [{
        "image": image,
        "image_type": "SOURCE_PRIVATE",
        "instance_count": 1,
        "shm_gi": 1800.0,
        "spec_id": "7166bd2e-6cbe-4bd9-be38-762d11003e7f",
    }],
    "logic_compute_group_id": "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
    "max_running_time_ms": "1800000",
    "name": run_id,
    "project_id": "project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
    "task_priority": 4,
    "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
}
with open(spec, "w", encoding="utf-8") as handle:
    json.dump(body, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

REQUEST_SHA="$(sha256sum "${SPEC}" | cut -d' ' -f1)"
printf '%s\n' "RUN_ID=${RUN_ID}" "NONCE=${NONCE}" "RECEIPT=${RECEIPT}" "REQUEST_SHA=${REQUEST_SHA}" > "${EV}/pending-identity.txt"

printf '%s\n' "=== DRY RUN ==="
qz train CreateJob --dry-run --data "$(cat "${SPEC}")" -o yaml 2>&1 | tail -5

printf '%s\n' "=== CREATEJOB ==="
if ! qz train CreateJob --data "$(cat "${SPEC}")" -o json > "${EV}/createjob.response.json" 2>"${EV}/createjob.stderr.txt"; then
    printf '%s\n' "CreateJob failed"; cat "${EV}/createjob.stderr.txt"; exit 1
fi
cat "${EV}/createjob.response.json"

JOB_ID="$(python3 - "${EV}/createjob.response.json" <<'PY'
import json, sys
doc = json.loads(open(sys.argv[1]).read())
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "job_id" and isinstance(v, str):
                return v
            r = walk(v)
            if r:
                return r
    elif isinstance(o, list):
        for x in o:
            r = walk(x)
            if r:
                return r
    return None
print(walk(doc) or "")
PY
)"
if [[ -z "${JOB_ID}" ]]; then printf '%s\n' "NO JOB_ID in response"; exit 1; fi
printf '%s\n' "JOB_ID=${JOB_ID}"

python3 - "$RECEIPT" "$JOB_ID" "$REQUEST_SHA" "$RUN_ID" "$NONCE" "$MANIFEST_SHA" <<'PY'
import json, sys
receipt, job_id, request_sha, run_id, nonce, manifest_sha = sys.argv[1:]
body = {
    "schema_version": 1,
    "job_id": job_id,
    "job_identity_origin": "controller_qz_createjob_receipt",
    "submitted_request_sha256": request_sha,
    "authorization_id": "nonlatent-h100-qualification-20260912-01",
    "project_id": "project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
    "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
    "logic_compute_group_id": "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
    "spec_id": "7166bd2e-6cbe-4bd9-be38-762d11003e7f",
    "run_id": run_id,
    "nonce": nonce,
    "source_manifest_sha256": manifest_sha,
    "requested_image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
    "scheduler_image_id": "image-7330d118-df9d-4e2e-82b1-c2543e831eb4",
    "observed_image_digest": None,
    "authorized_gpu_type": "NVIDIA_H100_SXM_80G",
    "requested_nodes": 1,
    "requested_gpus_per_node": 8,
    "requested_gpus": 8,
    "maximum_runtime_seconds": 1800,
    "maximum_gpu_hours": 4,
    "maximum_job_submissions": 1,
    "maximum_automatic_retries": 0,
}
with open(receipt, "w", encoding="utf-8") as handle:
    json.dump(body, handle)
print("receipt_written=" + receipt)
PY

printf '%s\n' "DONE"
