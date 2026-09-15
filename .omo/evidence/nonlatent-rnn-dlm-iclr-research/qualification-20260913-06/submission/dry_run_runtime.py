from __future__ import annotations

from pathlib import Path
import re
from typing import Final

from pydantic import ValidationError

from admission_parser import classify_createjob
from authority_models import SubmissionContext
from controller_models import (
    Admitted,
    CaptureTarget,
    CompletedProcess,
    ControllerError,
    Invocation,
    InvocationFailedProcess,
    ProcessRunner,
    Rejected,
    TimedOutProcess,
    Unknown,
)
from controller_storage import write_private_exclusive
from request_contract import qz_arguments, validate_request
from response_capture import invoke_and_capture, now
from runtime_records import DryRunReceipt


JOB_ID_PATTERN: Final = re.compile(
    rb"job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def run_dry_run(
    context: SubmissionContext, runner: ProcessRunner
) -> DryRunReceipt:
    validate_request(context.request_bytes, context)
    directory = Path(__file__).parent
    if (directory / "request-attempt.json").exists():
        raise ControllerError(code="dry_run_after_submission_forbidden")
    captured = invoke_and_capture(
        Invocation(
            arguments=qz_arguments(context.request_bytes, dry_run=True),
            timeout_seconds=120,
            capture=CaptureTarget(directory=directory, label="dry-run"),
        ),
        runner,
    )
    observed_ids = len(
        JOB_ID_PATTERN.findall(captured.outcome.stdout + captured.outcome.stderr)
    )
    match captured.outcome:
        case CompletedProcess(returncode=returncode):
            qz_exit_code = returncode
            process_valid = returncode == 0
        case TimedOutProcess() | InvocationFailedProcess():
            qz_exit_code = None
            process_valid = False
    match classify_createjob(captured.outcome):
        case Rejected():
            business_valid = False
        case Admitted() | Unknown():
            business_valid = True
    request_validated = process_valid and business_valid and observed_ids == 0
    receipt = DryRunReceipt(
        run_id=context.state.run_id,
        submitted_request_sha256=context.state.submitted_request_sha256,
        request_validated=request_validated,
        qz_exit_code=qz_exit_code,
        response_capture_path=captured.record.metadata_path,
        stdout_sha256=captured.record.stdout_sha256,
        stdout_bytes=captured.record.stdout_bytes,
        stderr_sha256=captured.record.stderr_sha256,
        stderr_bytes=captured.record.stderr_bytes,
        valid_job_ids_observed=observed_ids,
        recorded_at_utc=now(),
    )
    write_private_exclusive(
        directory / "dry-run-result.json",
        receipt.model_dump_json().encode("utf-8"),
        "dry_run_already_recorded",
    )
    if not request_validated:
        raise ControllerError(code="dry_run_blocked")
    return receipt


def load_dry_run(context: SubmissionContext) -> DryRunReceipt:
    try:
        receipt = DryRunReceipt.model_validate_json(
            (Path(__file__).parent / "dry-run-result.json").read_bytes()
        )
    except (OSError, ValidationError) as error:
        raise ControllerError(code="dry_run_receipt_invalid") from error
    if (
        not receipt.request_validated
        or receipt.run_id != context.state.run_id
        or receipt.submitted_request_sha256 != context.state.submitted_request_sha256
        or receipt.valid_job_ids_observed != 0
        or receipt.live_createjob_calls != 0
    ):
        raise ControllerError(code="dry_run_receipt_invalid")
    return receipt
