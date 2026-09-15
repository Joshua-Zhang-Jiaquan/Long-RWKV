from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import subprocess
from typing import Final

from pydantic import ValidationError

from controller_contract import ReceiptView, build_receipt, extract_job_id, qz_arguments
from controller_models import (
    AdmissionRecord,
    AdmittedJobError,
    AttemptContext,
    AttemptRecord,
    ControllerError,
    ControllerState,
    DryRunReceipt,
    HexDigest,
    ProcessResult,
    ProcessRunner,
)
from controller_storage import (
    load_authorization,
    load_state,
    now,
    publish_receipt,
    request_path,
    request_with_digest,
    write_exclusive,
)


ROOT_CONFIRMATION: Final = "ROOT_PASS_CONFIRMED"
JOB_ID_PATTERN: Final = re.compile(
    rb"job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def record_and_call_once(
    context: AttemptContext, raw: bytes, runner: ProcessRunner
) -> str:
    attempt = AttemptRecord(
        run_id=context.run_id,
        submitted_request_sha256=context.submitted_request_sha256,
        recorded_at_utc=now(),
    )
    write_exclusive(
        context.path,
        attempt.model_dump_json().encode(),
        "submission_already_attempted",
    )
    result = runner(qz_arguments(raw, dry_run=False))
    if result.returncode != 0:
        raise ControllerError(code="createjob_failed")
    return extract_job_id(result.stdout)


def run_qz(arguments: tuple[str, ...]) -> ProcessResult:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise ControllerError(code="qz_outcome_ambiguous_no_retry") from error
    except OSError as error:
        raise ControllerError(code="qz_invocation_failed") from error
    return ProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_dry_run(authorization_path: Path, state_path: Path) -> DryRunReceipt:
    load_authorization(authorization_path)
    state = load_state(state_path)
    raw, digest = request_with_digest(state)
    result = run_qz(qz_arguments(raw, dry_run=True))
    observed_ids = len(JOB_ID_PATTERN.findall(result.stdout + result.stderr))
    request_validated = result.returncode == 0 and observed_ids == 0
    receipt = DryRunReceipt(
        run_id=state.run_id,
        submitted_request_sha256=digest,
        request_validated=request_validated,
        qz_exit_code=result.returncode,
        stdout_sha256=sha256(result.stdout).hexdigest(),
        stdout_bytes=len(result.stdout),
        stderr_sha256=sha256(result.stderr).hexdigest(),
        stderr_bytes=len(result.stderr),
        valid_job_ids_observed=observed_ids,
        recorded_at_utc=now(),
    )
    write_exclusive(
        request_path(state).parent / "dry-run-result.json",
        receipt.model_dump_json().encode(),
        "dry_run_already_recorded",
    )
    if not request_validated:
        raise ControllerError(code="dry_run_blocked")
    return receipt


def _load_dry_run(state: ControllerState, digest: HexDigest) -> DryRunReceipt:
    try:
        receipt = DryRunReceipt.model_validate_json(
            (request_path(state).parent / "dry-run-result.json").read_bytes()
        )
    except (OSError, ValidationError) as error:
        raise ControllerError(code="dry_run_receipt_invalid") from error
    if (
        not receipt.request_validated
        or receipt.run_id != state.run_id
        or receipt.submitted_request_sha256 != digest
        or receipt.valid_job_ids_observed != 0
    ):
        raise ControllerError(code="dry_run_receipt_invalid")
    return receipt


def execute_submission(
    authorization_path: Path, state_path: Path, root_confirmation: str
) -> ReceiptView:
    authorization = load_authorization(authorization_path)
    if root_confirmation != ROOT_CONFIRMATION or authorization.preflight_verdict != "PASS":
        raise ControllerError(code="root_pass_required")
    state = load_state(state_path)
    raw, digest = request_with_digest(state)
    _load_dry_run(state, digest)
    context = AttemptContext(
        path=request_path(state).parent / "request-attempt.json",
        run_id=state.run_id,
        submitted_request_sha256=digest,
    )
    job_id = record_and_call_once(context, raw, run_qz)
    admission = AdmissionRecord(
        job_id=job_id,
        submitted_request_sha256=digest,
        run_id=state.run_id,
    )
    try:
        write_exclusive(
            request_path(state).parent / "admitted-job.json",
            admission.model_dump_json().encode(),
            "admitted_job_record_failed",
        )
        receipt = build_receipt(admission, state)
        publish_receipt(receipt, state.controller_receipt_path)
    except ControllerError as error:
        raise AdmittedJobError(code=error.code, job_id=job_id) from error
    return receipt
