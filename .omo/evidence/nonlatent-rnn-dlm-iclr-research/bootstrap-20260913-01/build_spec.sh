#!/usr/bin/env bash
# build_spec.sh v3 — emit job_spec_candidate.json with ACTUAL authorized resources
# (from .omo/authorizations/nonlatent-gpu-campaign-20260912.json) and the diagnostic
# payload INLINED, wrapped EXPLICITLY as:
#     /usr/bin/bash -lc <shlex.quote(inner)>
# matching the proven reference-launch pattern, so pipe/assignments execute under a
# documented shell regardless of scheduler shell parsing. The inner wrapper decodes
# the payload to /tmp, runs it with ALL control values inline, trap-removes the temp
# script on exit, preserves the payload exit status, and echoes BOOTSTRAP_DIAG_EXIT.
# ONLY BOOT_API_SENTINEL travels via job envs (API env channel pod-proof).
#
# IDENTITY PINNED (no job has been submitted): run_id/nonce/sentinel are frozen
# constants so re-freezes keep the same job name. FORCE_REBUILD=1 required once
# request_frozen.json exists; iterate via a NEW archived evidence dir instead.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FROZEN="$HERE/request_frozen.json"
if [[ -e "$FROZEN" && "${FORCE_REBUILD:-0}" != "1" ]]; then
  echo "FROZEN: request already finalized in $FROZEN — do not regenerate (run id/nonce must stay stable)."
  echo "To iterate: archive this evidence dir and start a new one."
  exit 7
fi

# --- actual authorized resources (verified against campaign authorization) ---
LCG="lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
PROJECT_ID="project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
WORKSPACE_ID="ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
SPEC_ID="7166bd2e-6cbe-4bd9-be38-762d11003e7f"
IMAGE="docker.sii.shaipower.online/inspire-studio/relay2:v2"

# --- pinned identity (unchanged across the shell-wrap re-freeze; nothing submitted) ---
RUN_ID="20260913T073456Z"
NONCE8="1fe0b3fe"
SENTINEL="boot-sentinel-1fe0b3fe-39a38dd743e94dd"

python3 - "$HERE/bootstrap_diag.sh" "$HERE/job_spec_candidate.json" "$HERE/request_frozen.json" \
         "$RUN_ID" "$NONCE8" "$SENTINEL" \
         "$LCG" "$PROJECT_ID" "$WORKSPACE_ID" "$SPEC_ID" "$IMAGE" <<'PYSPEC'
import base64, hashlib, json, shlex, sys
from datetime import datetime, timezone

(payload_path, out_path, frozen_path, run_id, nonce8, sentinel,
 lcg, project_id, workspace_id, spec_id, image) = sys.argv[1:12]

payload = open(payload_path).read()

PROJ = "/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY"
GLOB = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
ref_root = f"{GLOB}/outputs_birwkv_diffusion/logs"
qual_root = f"{GLOB}/nonlatent_iclr_qualification"

# Inline controls — every value must be quote/meta-char free (asserted below).
controls = {
    "BOOT_RUN_ID": run_id,
    "BOOT_NONCE": nonce8,
    "BOOT_OUT_ROOTS": f"{ref_root}:{qual_root}",
    "BOOT_PAYLOAD_PATH": PROJ,
    "BOOT_LAUNCHER_PATH": f"{PROJ}/scale/experiments/nonlatent_iclr/qualification/run_qualification.sh",
    "BOOT_STAGED_MODEL_PATH": f"{GLOB}/qz_stage_traj4096_v7/scale",
    "BOOT_RECEIPT_PATH": (f"{GLOB}/nonlatent_iclr_qualification/controller-receipts/"
                          "qualification-20260912-01-0ff9241e-460a-4647-95b3-af4c0fba16ce.json"),
    "BOOT_META_PATH": "",  # 43-byte checkpoint meta NOT located -> stays not_configured
    "BOOT_API_SENTINEL_EXPECTED": sentinel,
}
for k, v in controls.items():
    assert v == "" or ("'" not in v and '"' not in v and "$" not in v and "`" not in v
                       and "\\" not in v and "\n" not in v), f"unsafe control value: {k}"

b64 = base64.b64encode(payload.encode()).decode()
inline = " ".join(f"{k}={v}" for k, v in controls.items() if v != "")

# Explicit shell wrapper (same shape as proven reference launches):
# trap-removed temp script, payload exit status preserved through EXIT trap.
inner = (
    f'd=/tmp/bootstrap_diag.$$.sh'
    f"; trap 'rm -f \"$d\"' EXIT"
    f"; echo {b64} | base64 -d > \"$d\""
    f" && {inline} bash \"$d\""
    "; rc=$?; echo BOOTSTRAP_DIAG_EXIT=$rc; exit $rc"
)
command = "/usr/bin/bash -lc " + shlex.quote(inner)   # standard quoting only
assert "\n" not in command

# Structural gate: parsed argv must be exactly [/usr/bin/bash, -lc, <inner>].
argv = shlex.split(command)
assert argv[0] == "/usr/bin/bash" and argv[1] == "-lc", argv[:2]
assert argv[2] == inner
assert " base64 -d " in inner and inline in inner and "trap 'rm -f" in inner
assert "BOOT_DIAG_EXIT_MARKER_UNUSED" not in inner

spec = {
    "name": f"bootstrap-diag-p1-h100-{run_id}-{nonce8}",
    "framework": "pytorch",
    "command": command,
    "framework_config": [{
        "image": image,
        "image_type": "SOURCE_PRIVATE",
        "instance_count": 1,
        "shm_gi": 1800,
        "spec_id": spec_id,
    }],
    "logic_compute_group_id": lcg,
    "project_id": project_id,
    "workspace_id": workspace_id,
    "task_priority": 4,
    "max_running_time_ms": "180000",
    "auto_fault_tolerance": False,
    "fault_tolerance_max_retry": 0,
    "tb_summary_path": f"{ref_root}/bootstrap-tb-{run_id}",
    # ONLY the sentinel travels via the API env channel (pod-proof of env delivery).
    "envs": [{"name": "BOOT_API_SENTINEL", "value": sentinel}],
    "description": (
        "BOOTSTRAP ENVIRONMENT PROBE (CPU + driver query only; no model load, no "
        "training, no installs, no torch import): diagnoses the 5s/no-output failure "
        "lineage of job-e0c9f792. NVIDIA H100 SXM 80G x8, 1 node, 3-min cap. "
        "/usr/bin/bash -lc wrapped inline payload + inline controls (trap-cleaned temp "
        "script); BOOT_API_SENTINEL in envs only. Publishes diag.json under "
        "outputs_birwkv_diffusion/logs and nonlatent_iclr_qualification (unique child "
        "bootstrap-<run>-<nonce>). Bootstrap probe, NOT a model qualification or "
        "paper result."
    ),
}

json.dump(spec, open(out_path, "w"), indent=2)
spec_bytes = open(out_path, "rb").read()

frozen = {
    "schema": "bootstrap-diag-request-frozen/v2",
    "frozen_at_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "job_name": spec["name"],
    "run_id": run_id,
    "nonce": nonce8,
    "wrapper": "/usr/bin/bash -lc <shlex.quote(inner)>; inner trap-removes temp script; exit status preserved",
    "api_sentinel": {
        "channel": "job.envs ONLY",
        "expected_inline": True,
        "value_sha256": hashlib.sha256(sentinel.encode()).hexdigest(),
    },
    "request_sha256": hashlib.sha256(spec_bytes).hexdigest(),
    "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    "caps": {
        "gpu_type": "NVIDIA_H100_SXM_80G", "nodes": 1, "gpus_per_node": 8,
        "max_running_time_ms": 180000, "task_priority": 4,
        "auto_fault_tolerance": False, "fault_tolerance_max_retry": 0,
        "internal_timeout_term_s": 120, "internal_kill_grace_s": 5,
        "per_probe_timeout_s": 15,
    },
    "resources": {"logic_compute_group_id": lcg, "project_id": project_id,
                  "workspace_id": workspace_id, "spec_id": spec_id, "image": image},
    "read_paths": [
        "scheduler GetJobLog (full stdout incl. DIAG_SUMMARY / PATH / SENTINEL / PUBLISH lines)",
        f"{ref_root}/bootstrap-{run_id}-{nonce8}/diag.json",
        f"{qual_root}/bootstrap-{run_id}-{nonce8}/diag.json",
    ],
    "submit_command": 'qz train CreateJob --data "$(cat job_spec_candidate.json)" -o yaml',
    "authorization": ".omo/authorizations/nonlatent-gpu-campaign-20260912.json",
    "history": [
        "v2b archived: command began 'echo <b64> |' relying on scheduler shell parsing",
        "v3: /usr/bin/bash -lc + shlex.quote(inner), trap cleanup, identity pinned (nothing submitted)",
    ],
}
json.dump(frozen, open(frozen_path, "w"), indent=2)
print("WROTE", out_path)
print("FROZEN", frozen_path)
print("job_name:", spec["name"])
print("request_sha256:", frozen["request_sha256"])
PYSPEC
