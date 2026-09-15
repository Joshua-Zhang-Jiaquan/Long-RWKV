#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
export PATH=/usr/local/bin:/usr/bin:/bin

readonly EXPECTED_WORLD_SIZE=8
readonly WORKFLOW_TIMEOUT_SECONDS=1650
readonly PREFLIGHT_TIMEOUT_SECONDS=300
readonly RUNTIME_TIMEOUT_SECONDS=1200
readonly AGGREGATION_TIMEOUT_SECONDS=90
readonly CONTROLLER_RECEIPT_TIMEOUT_SECONDS=60
readonly NCCL_TIMEOUT_SECONDS=120
readonly KILL_GRACE_SECONDS=30
readonly ENVIRONMENT_TIMEOUT_SECONDS=30
readonly DEFAULT_OUTPUT_ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification
readonly DEFAULT_CHECKPOINT_DIR=/inspire/hdd/global_user/zhangjiaquan-253108540222/m2_baseline_triangle/m4loop_endpoint_ckpt
readonly DEFAULT_MODEL_DIR=/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF

readonly PAYLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${PAYLOAD_DIR}/../../../.." && pwd)"
readonly DEFAULT_STAGED_ROOT="${REPO_ROOT}/model_source/scale"

# The same bounded contract serves two payloads; the mode selects the work and is validated
# against an allowlist so a typo cannot silently run neither.
readonly QUALIFICATION_MODE="${QUALIFICATION_MODE:-probe}"
case "${QUALIFICATION_MODE}" in
    probe)
        readonly RUNTIME_MODULE=scale.experiments.nonlatent_iclr.qualification.probe
        readonly AGGREGATE_COMMAND=aggregate
        readonly SUCCESS_DETAIL=corevalid/fullmodelcacheunqualified
        ;;
    calibration)
        readonly RUNTIME_MODULE=scale.experiments.nonlatent_iclr.qualification.calibration
        readonly AGGREGATE_COMMAND=aggregate-calibration
        readonly SUCCESS_DETAIL=calibrated/task6forecastinput
        ;;
    *)
        printf '%s\n' 'QUALIFICATION_MODE must be probe or calibration' >&2
        exit 2
        ;;
esac
readonly PYTHON_BIN=/usr/bin/python
readonly STAGED_ROOT="${DEFAULT_STAGED_ROOT}"
readonly SHARED_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
readonly CHECKPOINT_DIR="${DEFAULT_CHECKPOINT_DIR}"
readonly MODEL_DIR="${DEFAULT_MODEL_DIR}"
readonly SOURCE_MANIFEST="${PAYLOAD_DIR}/runtime_manifest.json"
readonly MANIFEST_SHA256="${QUALIFICATION_MANIFEST_SHA256:-}"
readonly CONTROLLER_RECEIPT="${QUALIFICATION_JOB_RECEIPT:-}"

cd -- "${REPO_ROOT}"
export PYTHONSAFEPATH=1
export PYTHONPATH="${REPO_ROOT}:${STAGED_ROOT}"

if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' 'usage: QUALIFICATION_RUN_ID=<reserved-id> QUALIFICATION_NONCE=<32-hex> QUALIFICATION_MANIFEST_SHA256=<64-hex> QUALIFICATION_JOB_RECEIPT=<absolute-path> run_qualification.sh'
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    printf '%s\n' 'launcher accepts no positional arguments' >&2
    exit 2
fi
if [[ "${QUALIFICATION_WORKFLOW_INNER:-0}" != 1 ]]; then
    export QUALIFICATION_WORKFLOW_INNER=1
    exec /usr/bin/timeout --signal=TERM \
        --kill-after="${KILL_GRACE_SECONDS}s" \
        "${WORKFLOW_TIMEOUT_SECONDS}s" "${BASH_SOURCE[0]}" "$@"
fi
source "${PAYLOAD_DIR}/bytecode_guard.sh"
qualification_python_cache_setup
trap 'qualification_python_cache_cleanup' EXIT
qualification_assert_source_clean "${REPO_ROOT}/scale" "${STAGED_ROOT}"
if [[ -z "${QUALIFICATION_RUN_ID:-}" ]]; then
    printf '%s\n' 'QUALIFICATION_RUN_ID is required and must be externally reserved' >&2
    exit 2
fi
if [[ ! "${QUALIFICATION_RUN_ID}" =~ ^qualification-20260912-01-[a-z0-9-]{8,80}$ ]]; then
    printf '%s\n' 'QUALIFICATION_RUN_ID does not match the authorized unique-run format' >&2
    exit 2
fi
if [[ "${SHARED_OUTPUT_ROOT}" != /* || -L "${SHARED_OUTPUT_ROOT}" ]]; then
    printf '%s\n' 'SHARED_OUTPUT_ROOT must be an absolute non-symlink path' >&2
    exit 2
fi

readonly OUTPUT_DIR="${SHARED_OUTPUT_ROOT}/${QUALIFICATION_RUN_ID}"
mkdir -p "${SHARED_OUTPUT_ROOT}"
mkdir "${OUTPUT_DIR}"
set -o noclobber
readonly EARLY_FAILURE_RESULT="${OUTPUT_DIR}/early-failure.json"
readonly EARLY_FAILURE_TEMP="${OUTPUT_DIR}/.early-failure.tmp"
early_failure_detail=launcher_startup_failed

publish_early_failure() {
    local exit_code=$1
    if [[ "${exit_code}" -eq 0 || -e "${EARLY_FAILURE_RESULT}" ]]; then
        return 0
    fi
    printf '{"schema_version":1,"status":"FAILED","phase":"startup","detail":"%s","run_id":"%s"}\n' \
        "${early_failure_detail}" "${QUALIFICATION_RUN_ID}" >| "${EARLY_FAILURE_TEMP}"
    chmod 600 "${EARLY_FAILURE_TEMP}"
    ln -- "${EARLY_FAILURE_TEMP}" "${EARLY_FAILURE_RESULT}"
    rm -- "${EARLY_FAILURE_TEMP}"
}

early_cleanup() {
    local exit_code=$?
    trap - EXIT
    publish_early_failure "${exit_code}" || true
    qualification_python_cache_cleanup || true
    exit "${exit_code}"
}
trap early_cleanup EXIT

early_failure_detail=qualification_nonce_invalid
if [[ ! "${QUALIFICATION_NONCE:-}" =~ ^[0-9a-f]{32}$ ]]; then
    printf '%s\n' 'QUALIFICATION_NONCE must be controller-generated 32-character lowercase hex' >&2
    exit 2
fi
early_failure_detail=manifest_sha256_invalid
if [[ ! "${MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'QUALIFICATION_MANIFEST_SHA256 must be externally pinned 64-character lowercase hex' >&2
    exit 2
fi
early_failure_detail=controller_receipt_path_invalid
if [[ -z "${CONTROLLER_RECEIPT}" || "${CONTROLLER_RECEIPT}" != /* ]]; then
    printf '%s\n' 'QUALIFICATION_JOB_RECEIPT must be an absolute shared-GPFS path' >&2
    exit 2
fi
if [[ ! "${CONTROLLER_RECEIPT}" =~ ^/[a-zA-Z0-9._/-]+$ ]]; then
    printf '%s\n' 'QUALIFICATION_JOB_RECEIPT contains unsupported path characters' >&2
    exit 2
fi
if [[ -L "${CONTROLLER_RECEIPT}" ]]; then
    printf '%s\n' 'QUALIFICATION_JOB_RECEIPT must not be a symlink' >&2
    exit 2
fi
early_failure_detail=source_manifest_unavailable
if [[ ! -f "${SOURCE_MANIFEST}" || -L "${SOURCE_MANIFEST}" ]]; then
    printf '%s\n' 'SOURCE_MANIFEST must be a regular non-symlink file' >&2
    exit 2
fi

readonly NONCE="${QUALIFICATION_NONCE}"
readonly MANIFEST_SNAPSHOT="${OUTPUT_DIR}/runtime-manifest.json"
readonly RUNTIME_ENVIRONMENT_RESULT="${OUTPUT_DIR}/runtime_environment.json"
readonly PREFLIGHT_RECEIPT="${OUTPUT_DIR}/preflight.json"
readonly PREFLIGHT_TEMP="${OUTPUT_DIR}/.preflight.${NONCE}.tmp"
readonly AGGREGATE_RESULT="${OUTPUT_DIR}/aggregate.json"
readonly AGGREGATE_TEMP="${OUTPUT_DIR}/.aggregate.${NONCE}.tmp"
readonly LAUNCHER_RESULT="${OUTPUT_DIR}/launcher.json"
readonly LAUNCHER_TEMP="${OUTPUT_DIR}/.launcher.${NONCE}.tmp"
early_failure_detail=manifest_snapshot_failed
cp --no-clobber "${SOURCE_MANIFEST}" "${MANIFEST_SNAPSHOT}"
chmod 600 "${MANIFEST_SNAPSHOT}"

publish_once() {
    local source_path=$1
    local destination_path=$2
    chmod 600 "${source_path}"
    ln -- "${source_path}" "${destination_path}"
    rm -- "${source_path}"
}

write_launcher_result() {
    local status=$1
    local detail=$2
    local preflight_status=$3
    local launcher_status=$4
    local aggregate_status=$5
    if [[ -e "${LAUNCHER_RESULT}" ]]; then
        return 0
    fi
    printf '{"schema_version":2,"status":"%s","detail":"%s","preflight_status":%s,"launcher_status":%s,"aggregate_status":%s,"nonce":"%s","manifest_sha256":"%s","run_id":"%s","controller_receipt":"%s"}\n' \
        "${status}" "${detail}" "${preflight_status}" "${launcher_status}" \
        "${aggregate_status}" "${NONCE}" "${MANIFEST_SHA256}" \
        "${QUALIFICATION_RUN_ID}" "${CONTROLLER_RECEIPT}" >| "${LAUNCHER_TEMP}"
    publish_once "${LAUNCHER_TEMP}" "${LAUNCHER_RESULT}"
}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export QUALIFICATION_RUN_ID
export QUALIFICATION_JOB_RECEIPT="${CONTROLLER_RECEIPT}"

early_failure_detail=runtime_environment_observation_failed
/usr/bin/timeout --signal=TERM --kill-after="${KILL_GRACE_SECONDS}s" \
    "${ENVIRONMENT_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" \
    "${PAYLOAD_DIR}/runtime_environment.py" \
    --manifest "${MANIFEST_SNAPSHOT}" \
    --output "${RUNTIME_ENVIRONMENT_RESULT}" \
    --release-root "${REPO_ROOT}" \
    --staged-root "${STAGED_ROOT}" \
    --checkpoint-root "${CHECKPOINT_DIR}" \
    --model-root "${MODEL_DIR}"

launcher_pgid=""
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "${launcher_pgid}" ]]; then
        kill -TERM -- "-${launcher_pgid}" 2>/dev/null || true
        kill -KILL -- "-${launcher_pgid}" 2>/dev/null || true
        wait "${launcher_pgid}" 2>/dev/null || true
    fi
    if [[ "${exit_code}" -ne 0 && ! -e "${LAUNCHER_RESULT}" ]]; then
        write_launcher_result FAILED workflow_interrupted_or_timed_out null null null || true
    fi
    qualification_python_cache_cleanup || true
    exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 143' INT TERM

set +e
/usr/bin/timeout --signal=TERM --kill-after="${KILL_GRACE_SECONDS}s" \
    "${PREFLIGHT_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" \
    -m scale.experiments.nonlatent_iclr.qualification.cli validate \
    --output-dir "${OUTPUT_DIR}" \
    --nonce "${NONCE}" \
    --run-id "${QUALIFICATION_RUN_ID}" \
    --controller-receipt "${CONTROLLER_RECEIPT}" \
    --controller-receipt-timeout-seconds "${CONTROLLER_RECEIPT_TIMEOUT_SECONDS}" \
    --expected-world-size "${EXPECTED_WORLD_SIZE}" \
    --timeout-seconds "${RUNTIME_TIMEOUT_SECONDS}" \
    --nccl-timeout-seconds "${NCCL_TIMEOUT_SECONDS}" \
    --memory-fraction 0.90 \
    --staged-root "${STAGED_ROOT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --manifest "${MANIFEST_SNAPSHOT}" \
    --manifest-sha256 "${MANIFEST_SHA256}" \
    --preflight-receipt "${PREFLIGHT_RECEIPT}" > "${PREFLIGHT_TEMP}"
preflight_status=$?
set -e
if [[ "${preflight_status}" -eq 124 || "${preflight_status}" -eq 137 ]]; then
    printf '{"status":"FAILED","detail":"preflight_timeout"}\n' >| "${PREFLIGHT_TEMP}"
fi
publish_once "${PREFLIGHT_TEMP}" "${PREFLIGHT_RECEIPT}"
if [[ "${preflight_status}" -ne 0 ]]; then
    preflight_detail=preflight_failed
    if [[ "${preflight_status}" -eq 124 || "${preflight_status}" -eq 137 ]]; then
        preflight_detail=preflight_timeout
    fi
    write_launcher_result FAILED "${preflight_detail}" "${preflight_status}" null null
    exit 1
fi

/usr/bin/setsid /usr/bin/timeout --signal=TERM \
    --kill-after="${KILL_GRACE_SECONDS}s" "${RUNTIME_TIMEOUT_SECONDS}s" \
    /usr/local/bin/torchrun --standalone --nnodes=1 \
    --nproc-per-node="${EXPECTED_WORLD_SIZE}" --max-restarts=0 \
    -m "${RUNTIME_MODULE}" \
    --output-dir "${OUTPUT_DIR}" \
    --nonce "${NONCE}" \
    --run-id "${QUALIFICATION_RUN_ID}" \
    --controller-receipt "${CONTROLLER_RECEIPT}" \
    --controller-receipt-timeout-seconds "${CONTROLLER_RECEIPT_TIMEOUT_SECONDS}" \
    --expected-world-size "${EXPECTED_WORLD_SIZE}" \
    --timeout-seconds "${RUNTIME_TIMEOUT_SECONDS}" \
    --nccl-timeout-seconds "${NCCL_TIMEOUT_SECONDS}" \
    --memory-fraction 0.90 \
    --staged-root "${STAGED_ROOT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --manifest "${MANIFEST_SNAPSHOT}" \
    --manifest-sha256 "${MANIFEST_SHA256}" \
    --preflight-receipt "${PREFLIGHT_RECEIPT}" &
launcher_pgid=$!
set +e
wait "${launcher_pgid}"
launcher_status=$?
set -e
if [[ "${launcher_status}" -ne 0 ]]; then
    kill -TERM -- "-${launcher_pgid}" 2>/dev/null || true
    kill -KILL -- "-${launcher_pgid}" 2>/dev/null || true
fi
launcher_pgid=""

set +e
/usr/bin/timeout --signal=TERM --kill-after="${KILL_GRACE_SECONDS}s" \
    "${AGGREGATION_TIMEOUT_SECONDS}s" "${PYTHON_BIN}" \
    -m scale.experiments.nonlatent_iclr.qualification.cli "${AGGREGATE_COMMAND}" \
    --output-dir "${OUTPUT_DIR}" \
    --nonce "${NONCE}" \
    --run-id "${QUALIFICATION_RUN_ID}" \
    --controller-receipt "${CONTROLLER_RECEIPT}" \
    --manifest "${MANIFEST_SNAPSHOT}" \
    --manifest-sha256 "${MANIFEST_SHA256}" \
    --preflight-receipt "${PREFLIGHT_RECEIPT}" > "${AGGREGATE_TEMP}"
aggregate_status=$?
set -e
if [[ "${aggregate_status}" -eq 124 || "${aggregate_status}" -eq 137 ]]; then
    printf '{"status":"FAILED","detail":"aggregation_timeout"}\n' >| "${AGGREGATE_TEMP}"
fi
publish_once "${AGGREGATE_TEMP}" "${AGGREGATE_RESULT}"

terminal_status=FAILED
terminal_detail=runtime_or_aggregation_failed
if [[ "${launcher_status}" -eq 0 && "${aggregate_status}" -eq 0 ]]; then
    terminal_status=PARTIAL_QUALIFICATION
    terminal_detail="${SUCCESS_DETAIL}"
elif [[ "${launcher_status}" -eq 124 || "${launcher_status}" -eq 137 ]]; then
    terminal_detail=runtime_timeout
elif [[ "${aggregate_status}" -eq 124 || "${aggregate_status}" -eq 137 ]]; then
    terminal_detail=aggregation_timeout
fi
write_launcher_result "${terminal_status}" "${terminal_detail}" \
    "${preflight_status}" "${launcher_status}" "${aggregate_status}"
if [[ "${terminal_status}" != "PARTIAL_QUALIFICATION" ]]; then
    exit 1
fi
