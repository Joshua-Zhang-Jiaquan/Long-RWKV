from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints


CAMPAIGN_AUTHORIZATION_ID: Final = "nonlatent-gpu-campaign-20260912"
LEGACY_RUNTIME_PROFILE_ID: Final = "nonlatent-h100-qualification-20260912-01"
MANIFEST_SHA256: Final = (
    "4777fd86769f9264af91241c82655a9b8f51874500e79fbd92f9b19030003f9f"
)
IMAGE: Final = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
IMAGE_ID: Final = "image-7330d118-df9d-4e2e-82b1-c2543e831eb4"
PROJECT_ID: Final = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
WORKSPACE_ID: Final = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
LCG_ID: Final = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_ID: Final = "7166bd2e-6cbe-4bd9-be38-762d11003e7f"
GPU_TYPE: Final = "NVIDIA_H100_SXM_80G"
type HexDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type JobId = Annotated[
    str,
    StringConstraints(
        pattern=r"^job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
]
type Nonce = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
type RunId = Annotated[
    str,
    StringConstraints(
        pattern=r"^qualification-20260912-01-[0-9a-f]{8}-[0-9a-f]{4}-"
        r"4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    ),
]
type PermitId = Annotated[
    str,
    StringConstraints(
        pattern=r"^permit-nonlatent-gpu-campaign-20260912-[0-9a-f-]{36}$"
    ),
]
type GrantInstanceId = Annotated[
    str,
    StringConstraints(
        pattern=r"^grant-qualification-20260912-02-[0-9a-f-]{36}$"
    ),
]
type UtcTimestamp = Annotated[
    str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
]


@dataclass(frozen=True, slots=True)
class CompletedProcess:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True, slots=True)
class TimedOutProcess:
    timeout_seconds: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True, slots=True)
class InvocationFailedProcess:
    error_code: str
    stdout: bytes = b""
    stderr: bytes = b""


type ProcessOutcome = CompletedProcess | TimedOutProcess | InvocationFailedProcess


@dataclass(frozen=True, slots=True)
class CaptureTarget:
    directory: Path
    label: str


@dataclass(frozen=True, slots=True)
class Invocation:
    arguments: tuple[str, ...]
    timeout_seconds: int
    capture: CaptureTarget


class ProcessRunner(Protocol):
    def __call__(
        self, arguments: tuple[str, ...], timeout_seconds: int
    ) -> CompletedProcess: ...


class ResponseCapture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    label: str
    outcome: Literal["completed", "timed_out", "invocation_failed"]
    returncode: int | None
    timed_out: bool
    timeout_seconds: int
    invocation_error_code: str | None
    stdout_sha256: HexDigest
    stdout_bytes: int
    stdout_redacted_sha256: HexDigest
    stdout_redacted_bytes: int
    stdout_path: Path
    stderr_sha256: HexDigest
    stderr_bytes: int
    stderr_redacted_sha256: HexDigest
    stderr_redacted_bytes: int
    stderr_path: Path
    metadata_path: Path
    recorded_at_utc: UtcTimestamp


@dataclass(frozen=True, slots=True)
class CapturedInvocation:
    outcome: ProcessOutcome
    record: ResponseCapture


@dataclass(frozen=True, slots=True)
class Admitted:
    job_id: JobId
    identity_path: Literal["job_id", "Result.job_id"]


@dataclass(frozen=True, slots=True)
class Rejected:
    error_code: str
    message: str


@dataclass(frozen=True, slots=True)
class Unknown:
    reason: str


type AdmissionDecision = Admitted | Rejected | Unknown


@dataclass(frozen=True, slots=True)
class JobExpectation:
    job_id: JobId
    name: RunId


@dataclass(frozen=True, slots=True)
class ConfirmedJob:
    job_id: JobId
    status: str


type GetJobDecision = ConfirmedJob | Rejected | Unknown


@dataclass(frozen=True, slots=True)
class AttemptContext:
    attempt_path: Path
    capture: CaptureTarget
    campaign_authorization_id: Literal["nonlatent-gpu-campaign-20260912"]
    permit_id: PermitId
    grant_instance_id: GrantInstanceId
    run_id: RunId
    submitted_request_sha256: HexDigest


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    campaign_authorization_id: Literal["nonlatent-gpu-campaign-20260912"]
    permit_id: PermitId
    grant_instance_id: GrantInstanceId
    run_id: RunId
    submitted_request_sha256: HexDigest
    maximum_real_createjob_calls: Literal[1] = 1
    automatic_retries: Literal[0] = 0
    status: Literal["ATTEMPT_RECORDED_BEFORE_CREATEJOB"]
    recorded_at_utc: UtcTimestamp


class AdmissionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    campaign_authorization_id: Literal["nonlatent-gpu-campaign-20260912"]
    permit_id: PermitId
    grant_instance_id: GrantInstanceId
    job_id: JobId
    submitted_request_sha256: HexDigest
    run_id: RunId
    status: Literal["ADMITTED_GETJOB_CONFIRMATION_PENDING"] = (
        "ADMITTED_GETJOB_CONFIRMATION_PENDING"
    )


@dataclass(frozen=True, slots=True)
class AttemptResult:
    decision: AdmissionDecision
    capture: ResponseCapture


class ControllerError(RuntimeError):
    __slots__ = ("code",)

    code: str

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class AdmittedJobError(RuntimeError):
    __slots__ = ("code", "job_id")

    code: str
    job_id: str

    def __init__(self, code: str, job_id: str) -> None:
        super().__init__(code, job_id)
        self.code = code
        self.job_id = job_id
