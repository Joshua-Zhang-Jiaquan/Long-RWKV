from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from controller_models import (
    HexDigest,
    JobId,
    Rejected,
    ResponseCapture,
    RunId,
    Unknown,
    UtcTimestamp,
)
from controller_storage import write_private_exclusive
from response_capture import now, redact_payload


class DryRunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    run_id: RunId
    submitted_request_sha256: HexDigest
    literal_dry_run: Literal[True] = True
    request_validated: bool
    qz_exit_code: int | None
    response_capture_path: Path
    stdout_sha256: HexDigest
    stdout_bytes: int
    stderr_sha256: HexDigest
    stderr_bytes: int
    valid_job_ids_observed: int
    live_createjob_calls: Literal[0] = 0
    recorded_at_utc: UtcTimestamp


class SubmissionOutcomeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    run_id: RunId
    decision: Literal["REJECTED", "UNKNOWN"]
    error_code: str | None
    message: str | None
    reason: str | None
    response_capture_path: Path
    automatic_retry: Literal[False] = False
    recorded_at_utc: UtcTimestamp


class InitialGetJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    job_id: JobId
    confirmation: Literal["CONFIRMED", "REJECTED", "UNKNOWN"]
    current_status: str | None
    detail_code: str | None
    response_capture_path: Path
    shared_output_directory: Path
    recorded_at_utc: UtcTimestamp


@dataclass(frozen=True, slots=True)
class OutcomeEvidenceContext:
    directory: Path
    run_id: RunId
    capture: ResponseCapture


def persist_submission_decision(
    context: OutcomeEvidenceContext, decision: Rejected | Unknown
) -> None:
    match decision:
        case Rejected(error_code=error_code, message=message):
            record = SubmissionOutcomeRecord(
                run_id=context.run_id,
                decision="REJECTED",
                error_code=error_code,
                message=redact_payload(message.encode("utf-8")).decode("utf-8"),
                reason=None,
                response_capture_path=context.capture.metadata_path,
                recorded_at_utc=now(),
            )
        case Unknown(reason=reason):
            record = SubmissionOutcomeRecord(
                run_id=context.run_id,
                decision="UNKNOWN",
                error_code=None,
                message=None,
                reason=reason,
                response_capture_path=context.capture.metadata_path,
                recorded_at_utc=now(),
            )
    write_private_exclusive(
        context.directory / "submission-outcome.json",
        record.model_dump_json().encode("utf-8"),
        "submission_outcome_record_failed",
    )


def persist_initial_getjob(directory: Path, record: InitialGetJobRecord) -> None:
    write_private_exclusive(
        directory / "initial-getjob.json",
        record.model_dump_json().encode("utf-8"),
        "initial_getjob_record_failed",
    )
