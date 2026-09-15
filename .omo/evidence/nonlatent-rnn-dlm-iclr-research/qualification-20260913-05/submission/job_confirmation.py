from __future__ import annotations

from pathlib import Path
from typing import Final

from admission_parser import confirm_getjob
from authority_models import SubmissionContext
from controller_models import (
    AdmissionRecord,
    AdmittedJobError,
    CaptureTarget,
    ConfirmedJob,
    Invocation,
    JobExpectation,
    ProcessRunner,
    Rejected,
    Unknown,
)
from controller_storage import publish_receipt
from receipt_contract import ReceiptView, build_legacy_runtime_receipt
from request_contract import getjob_arguments
from response_capture import invoke_and_capture, now
from runtime_records import InitialGetJobRecord, persist_initial_getjob


SHARED_OUTPUT_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification"
)


def confirm_and_publish(
    context: SubmissionContext, admission: AdmissionRecord, runner: ProcessRunner
) -> ReceiptView:
    directory = Path(__file__).parent
    captured = invoke_and_capture(
        Invocation(
            arguments=getjob_arguments(admission.job_id),
            timeout_seconds=120,
            capture=CaptureTarget(directory=directory, label="initial-getjob"),
        ),
        runner,
    )
    confirmation = confirm_getjob(
        captured.outcome,
        JobExpectation(job_id=admission.job_id, name=context.state.job_name),
    )
    match confirmation:
        case ConfirmedJob(status=status):
            result = InitialGetJobRecord(
                job_id=admission.job_id,
                confirmation="CONFIRMED",
                current_status=status,
                detail_code=None,
                response_capture_path=captured.record.metadata_path,
                shared_output_directory=SHARED_OUTPUT_ROOT / context.state.run_id,
                recorded_at_utc=now(),
            )
        case Rejected(error_code=error_code):
            result = InitialGetJobRecord(
                job_id=admission.job_id,
                confirmation="REJECTED",
                current_status=None,
                detail_code=error_code,
                response_capture_path=captured.record.metadata_path,
                shared_output_directory=SHARED_OUTPUT_ROOT / context.state.run_id,
                recorded_at_utc=now(),
            )
        case Unknown(reason=reason):
            result = InitialGetJobRecord(
                job_id=admission.job_id,
                confirmation="UNKNOWN",
                current_status=None,
                detail_code=reason,
                response_capture_path=captured.record.metadata_path,
                shared_output_directory=SHARED_OUTPUT_ROOT / context.state.run_id,
                recorded_at_utc=now(),
            )
    persist_initial_getjob(directory, result)
    match confirmation:
        case ConfirmedJob():
            receipt = build_legacy_runtime_receipt(admission, context.state)
            publish_receipt(receipt, admission, context.state.controller_receipt_path)
            return receipt
        case Rejected() | Unknown():
            raise AdmittedJobError(
                code="admitted_job_getjob_confirmation_failed",
                job_id=admission.job_id,
            )
