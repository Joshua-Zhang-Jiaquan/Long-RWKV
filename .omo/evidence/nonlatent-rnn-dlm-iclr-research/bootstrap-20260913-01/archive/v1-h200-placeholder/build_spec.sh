#!/usr/bin/env bash
# build_spec.sh — emit job_spec_candidate.json with the diagnostic payload INLINED
# as a single-line base64 `command` (zero quoting/expansion hazards; payload is
# never executed while preparing — base64 is pure string encoding).
#
# Usage:
#   bash build_spec.sh                 # candidate with PLACEHOLDER pool id (NOT submit-able)
#   POOL_ID=lcg-<full-uuid> bash build_spec.sh   # submit-ready candidate
#
# Known pool fragment from the H200 training pool: lcg-71b971a7-...  (confirm the
# full id from a prior successful job in this pool before finalizing).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
POOL_ID="${POOL_ID:-lcg-71b971a7-PENDING-FULL-UUID-CONFIRM-BEFORE-SUBMIT}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
NONCE8="$(printf '%08x' "$((RANDOM * 32768 + RANDOM))")"

python3 - "$HERE/bootstrap_diag.sh" "$HERE/job_spec_candidate.json" "$RUN_ID" "$NONCE8" "$POOL_ID" <<'PYSPEC'
import base64, json, sys

payload_path, out_path, run_id, nonce8, pool_id = sys.argv[1:6]
payload = open(payload_path).read()

# Single-line command: decode + run from /tmp. No dependency on any mounted path.
b64 = base64.b64encode(payload.encode()).decode()
command = (
    "echo " + b64 + " | base64 -d > /tmp/bootstrap_diag.$$.sh"
    " && bash /tmp/bootstrap_diag.$$.sh; rc=$?;"
    " echo BOOTSTRAP_DIAG_EXIT=$rc; exit $rc"
)
assert "\n" not in command and "'" not in command and '"' not in command

ref_root = "/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/logs"
qual_root = "/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification"

spec = {
    "name": f"bootstrap-diag-p1-h200-{run_id}-{nonce8}",
    "framework": "pytorch",
    "command": command,
    "framework_config": [{
        "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
        "image_type": "SOURCE_PRIVATE",
        "instance_count": 1,
        "shm_gi": 1800,
        "spec_id": "7166bd2e-6cbe-4bd9-be38-762d11003e7f",
    }],
    "logic_compute_group_id": pool_id,
    "project_id": "project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
    "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
    "task_priority": 4,
    "max_running_time_ms": "180000",
    "auto_fault_tolerance": False,
    "fault_tolerance_max_retry": 0,
    "tb_summary_path": f"{ref_root}/bootstrap-tb-{run_id}",
    "envs": [
        {"name": "BOOT_RUN_ID", "value": run_id},
        {"name": "BOOT_NONCE", "value": nonce8},
        {"name": "BOOT_PAYLOAD_PATH",
         "value": "/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY"},
        {"name": "BOOT_STAGED_MODEL_PATH",
         "value": "/inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale"},
        {"name": "BOOT_OUT_ROOTS", "value": f"{ref_root} {qual_root}"},
    ],
    "description": (
        "BOOTSTRAP ENVIRONMENT PROBE (CPU + driver query only; no model load, no "
        "training, no installs): diagnoses the <5s no-output failure lineage of "
        "qualification job e0c9f792. Inline payload; does NOT require the project "
        "source mount. Publishes diag.json to "
        "outputs_birwkv_diffusion/logs/bootstrap-<run>-<nonce> and "
        "nonlatent_iclr_qualification/bootstrap-<run>-<nonce>. This is a bootstrap "
        "probe, NOT a model qualification or paper result."
    ),
    # NOTE: BOOT_META_PATH / BOOT_RECEIPT_PATH intentionally omitted (unset ->
    # recorded as not_configured). Set BOOT_META_PATH once the checkpoint metadata
    # filename is confirmed; never point it at weight files.
}

with open(out_path, "w") as f:
    json.dump(spec, f, indent=2)
print("WROTE", out_path)
print("name:", spec["name"])
if "PENDING" in pool_id:
    print("WARNING: logic_compute_group_id is a PLACEHOLDER — set POOL_ID=lcg-<full-uuid> "
          "to regenerate a submit-ready spec. DO NOT dry-run/submit the placeholder spec.")
PYSPEC
