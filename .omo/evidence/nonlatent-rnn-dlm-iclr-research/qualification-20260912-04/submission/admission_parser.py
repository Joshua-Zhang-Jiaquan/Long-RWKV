from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError

from controller_models import (
    GPU_TYPE,
    IMAGE,
    LCG_ID,
    PROJECT_ID,
    SPEC_ID,
    WORKSPACE_ID,
    Admitted,
    AdmissionDecision,
    CompletedProcess,
    ConfirmedJob,
    GetJobDecision,
    InvocationFailedProcess,
    JobExpectation,
    JobId,
    ProcessOutcome,
    Rejected,
    TimedOutProcess,
    Unknown,
)


class JobIdValue(RootModel[JobId]):
    model_config = ConfigDict(frozen=True, strict=True)


class ApiError(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    code: str = Field(alias="Code")
    message: str = Field(alias="Message")


class ResponseMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    action: str | None = Field(default=None, alias="Action")
    error: ApiError | None = Field(default=None, alias="Error")


class CreateJobResult(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    job_id: str | None = None


class CreateJobEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    response_metadata: ResponseMetadata | None = Field(
        default=None, alias="ResponseMetadata"
    )
    result: CreateJobResult | None = Field(default=None, alias="Result")
    job_id: str | None = None


class GpuInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    gpu_type: str | None = None


class SpecPrice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    quota_id: str | None = None
    gpu_count: int | None = None
    gpu_type: str | None = None
    gpu_info: GpuInfo | None = None


class JobFramework(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    image: str | None = None
    image_type: str | None = None
    instance_count: int | None = None
    shm_gi: int | float | None = None
    resource_spec_price: SpecPrice | None = None
    instance_spec_price_info: SpecPrice | None = None


class GetJobResult(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    job_id: str | None = None
    name: str | None = None
    project_id: str | None = None
    workspace_id: str | None = None
    logic_compute_group_id: str | None = None
    max_running_time_ms: str | None = None
    auto_fault_tolerance: bool | None = None
    fault_tolerance_max_retry: int | None = None
    task_priority: int | None = None
    status: str | None = None
    framework_config: tuple[JobFramework, ...] = ()


class GetJobEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    response_metadata: ResponseMetadata | None = Field(
        default=None, alias="ResponseMetadata"
    )
    result: GetJobResult | None = Field(default=None, alias="Result")


def _metadata_error(metadata: ResponseMetadata | None) -> Rejected | None:
    match metadata:
        case ResponseMetadata(error=ApiError(code=code, message=message)):
            return Rejected(error_code=code, message=message)
        case ResponseMetadata() | None:
            return None


def _valid_job_id(candidate: str) -> JobId | None:
    try:
        return JobIdValue.model_validate(candidate, strict=True).root
    except ValidationError:
        return None


def classify_createjob(outcome: ProcessOutcome) -> AdmissionDecision:
    match outcome:
        case TimedOutProcess():
            return Unknown(reason="createjob_timeout_no_retry")
        case InvocationFailedProcess():
            return Unknown(reason="createjob_invocation_failed")
        case CompletedProcess(returncode=returncode, stdout=stdout):
            if not stdout:
                return Unknown(reason="createjob_empty_response")
            try:
                envelope = CreateJobEnvelope.model_validate_json(stdout)
            except ValidationError:
                return Unknown(reason="createjob_unrecognized_response")
            rejection = _metadata_error(envelope.response_metadata)
            if rejection is not None:
                return rejection
            if returncode != 0:
                return Unknown(reason="createjob_process_nonzero")
            metadata = envelope.response_metadata
            if metadata is not None and metadata.action not in (None, "CreateJob"):
                return Unknown(reason="createjob_unexpected_action")
            direct = envelope.job_id
            nested = envelope.result.job_id if envelope.result is not None else None
            candidates = tuple(value for value in (direct, nested) if value is not None)
            if not candidates:
                return Unknown(reason="createjob_success_missing_job_id")
            validated = tuple(_valid_job_id(candidate) for candidate in candidates)
            if any(candidate is None for candidate in validated):
                return Unknown(reason="createjob_invalid_job_id")
            unique = {candidate for candidate in validated if candidate is not None}
            if len(unique) != 1:
                return Unknown(reason="createjob_multiple_job_ids")
            job_id = next(iter(unique))
            identity_path: Literal["job_id", "Result.job_id"] = (
                "job_id" if direct is not None else "Result.job_id"
            )
            return Admitted(job_id=job_id, identity_path=identity_path)


def _spec_price(framework: JobFramework) -> SpecPrice | None:
    return framework.instance_spec_price_info or framework.resource_spec_price


def _framework_matches(framework: JobFramework) -> bool:
    price = _spec_price(framework)
    if price is None:
        return False
    gpu_type = price.gpu_info.gpu_type if price.gpu_info is not None else price.gpu_type
    return (
        framework.image == IMAGE
        and framework.image_type == "SOURCE_PRIVATE"
        and framework.instance_count == 1
        and framework.shm_gi == 1800
        and price.quota_id == SPEC_ID
        and price.gpu_count == 8
        and gpu_type == GPU_TYPE
    )


def confirm_getjob(
    outcome: ProcessOutcome, expectation: JobExpectation
) -> GetJobDecision:
    match outcome:
        case TimedOutProcess():
            return Unknown(reason="getjob_timeout")
        case InvocationFailedProcess():
            return Unknown(reason="getjob_invocation_failed")
        case CompletedProcess(returncode=returncode, stdout=stdout):
            if not stdout:
                return Unknown(reason="getjob_empty_response")
            try:
                envelope = GetJobEnvelope.model_validate_json(stdout)
            except ValidationError:
                return Unknown(reason="getjob_unrecognized_response")
            rejection = _metadata_error(envelope.response_metadata)
            if rejection is not None:
                return rejection
            result = envelope.result
            metadata = envelope.response_metadata
            if (
                returncode != 0
                or result is None
                or metadata is None
                or metadata.action != "GetJob"
                or len(result.framework_config) != 1
            ):
                return Unknown(reason="getjob_binding_mismatch")
            if (
                result.job_id != expectation.job_id
                or result.name != expectation.name
                or result.project_id != PROJECT_ID
                or result.workspace_id != WORKSPACE_ID
                or result.logic_compute_group_id != LCG_ID
                or result.max_running_time_ms != "1800000"
                or result.auto_fault_tolerance is not False
                or result.fault_tolerance_max_retry != 0
                or result.task_priority != 0
                or result.status is None
                or not _framework_matches(result.framework_config[0])
            ):
                return Unknown(reason="getjob_binding_mismatch")
            return ConfirmedJob(job_id=expectation.job_id, status=result.status)
