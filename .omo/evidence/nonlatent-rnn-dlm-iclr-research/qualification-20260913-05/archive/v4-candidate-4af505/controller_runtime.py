from __future__ import annotations

from pathlib import Path

from authority_models import SubmissionContext
from controller_models import (
    Admitted,
    AdmissionRecord,
    AdmittedJobError,
    AttemptContext,
    AttemptRecord,
    AttemptResult,
    CaptureTarget,
    ControllerError,
    Invocation,
    ProcessRunner,
    Rejected,
    Unknown,
)
from controller_storage import (
    load_admission,
    require_live_submission,
    write_private_exclusive,
)
from dry_run_runtime import load_dry_run, run_dry_run as run_dry_run
from job_confirmation import confirm_and_publish
from receipt_contract import ReceiptView
from request_contract import qz_arguments, validate_request
from response_capture import invoke_and_capture, now, require_capture_absent
from runtime_records import OutcomeEvidenceContext, persist_submission_decision


def record_capture_and_classify_once(
    context: AttemptContext, raw: bytes, runner: ProcessRunner
) -> AttemptResult:
    require_capture_absent(context.capture)
    attempt = AttemptRecord(
        campaign_authorization_id=context.campaign_authorization_id,
        permit_id=context.permit_id,
        grant_instance_id=context.grant_instance_id,
        run_id=context.run_id,
        submitted_request_sha256=context.submitted_request_sha256,
        status="ATTEMPT_RECORDED_BEFORE_CREATEJOB",
        recorded_at_utc=now(),
    )
    write_private_exclusive(
        context.attempt_path,
        attempt.model_dump_json().encode("utf-8"),
        "submission_already_attempted",
    )
    captured = invoke_and_capture(
        Invocation(
            arguments=qz_arguments(raw, dry_run=False),
            timeout_seconds=120,
            capture=context.capture,
        ),
        runner,
    )
    from admission_parser import classify_createjob

    return AttemptResult(
        decision=classify_createjob(captured.outcome),
        capture=captured.record,
    )


def _admission_for_context(
    admission: AdmissionRecord, context: SubmissionContext
) -> AdmissionRecord:
    if (
        admission.campaign_authorization_id
        != context.state.campaign_authorization_id
        or admission.permit_id != context.state.permit_id
        or admission.grant_instance_id != context.state.grant_instance_id
        or admission.run_id != context.state.run_id
        or admission.submitted_request_sha256
        != context.state.submitted_request_sha256
    ):
        raise ControllerError(code="admission_record_invalid")
    return admission


def _confirm_with_known_id(
    context: SubmissionContext, admission: AdmissionRecord, runner: ProcessRunner
) -> ReceiptView:
    try:
        return confirm_and_publish(context, admission, runner)
    except ControllerError as error:
        raise AdmittedJobError(code=error.code, job_id=admission.job_id) from error


def execute_submission(
    context: SubmissionContext, root_confirmation: str, runner: ProcessRunner
) -> ReceiptView:
    validate_request(context.request_bytes, context)
    require_live_submission(context, root_confirmation)
    directory = Path(__file__).parent
    load_dry_run(context)
    attempt_path = directory / "request-attempt.json"
    admission_path = directory / "admitted-job.json"
    if attempt_path.exists():
        if not admission_path.exists():
            raise ControllerError(code="submission_attempt_requires_reconciliation")
        admission = _admission_for_context(load_admission(admission_path), context)
        return _confirm_with_known_id(context, admission, runner)
    attempt_context = AttemptContext(
        attempt_path=attempt_path,
        capture=CaptureTarget(directory=directory, label="createjob"),
        campaign_authorization_id=context.state.campaign_authorization_id,
        permit_id=context.state.permit_id,
        grant_instance_id=context.state.grant_instance_id,
        run_id=context.state.run_id,
        submitted_request_sha256=context.state.submitted_request_sha256,
    )
    attempt = record_capture_and_classify_once(
        attempt_context, context.request_bytes, runner
    )
    evidence = OutcomeEvidenceContext(
        directory=directory,
        run_id=context.state.run_id,
        capture=attempt.capture,
    )
    match attempt.decision:
        case Rejected() as rejected:
            persist_submission_decision(evidence, rejected)
            raise ControllerError(code="createjob_rejected")
        case Unknown() as unknown:
            persist_submission_decision(evidence, unknown)
            raise ControllerError(code="createjob_outcome_unknown_no_retry")
        case Admitted(job_id=job_id):
            admission = AdmissionRecord(
                campaign_authorization_id=context.state.campaign_authorization_id,
                permit_id=context.state.permit_id,
                grant_instance_id=context.state.grant_instance_id,
                job_id=job_id,
                submitted_request_sha256=context.state.submitted_request_sha256,
                run_id=context.state.run_id,
            )
            try:
                write_private_exclusive(
                    admission_path,
                    admission.model_dump_json().encode("utf-8"),
                    "admission_record_failed",
                )
            except ControllerError as error:
                raise AdmittedJobError(code=error.code, job_id=job_id) from error
            return _confirm_with_known_id(context, admission, runner)
