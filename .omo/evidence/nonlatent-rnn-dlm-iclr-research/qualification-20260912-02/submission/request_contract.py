from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from authority_models import ControllerState, SubmissionContext
from controller_models import (
    IMAGE,
    LCG_ID,
    PROJECT_ID,
    SPEC_ID,
    WORKSPACE_ID,
    ControllerError,
    JobId,
    RunId,
)
from receipt_contract import (
    build_legacy_runtime_receipt as build_legacy_runtime_receipt,
    validate_legacy_runtime_receipt as validate_legacy_runtime_receipt,
)


LAUNCH_COMMAND: Final = (
    "/usr/bin/bash /inspire/hdd/project/multimodal-diffusion-language-model/"
    "zhangjiaquan-253108540222/DiffRWKV-RELAY/scale/experiments/"
    "nonlatent_iclr/qualification/run_qualification.sh"
)
DESCRIPTION: Final = (
    "Single authorized nonlatent runtime qualification: one 8xH100 SXM 80GB "
    "node, 30 minutes, zero retries."
)


class EnvironmentVariable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    value: str


class FrameworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    image: Literal["docker.sii.shaipower.online/inspire-studio/relay2:v2"]
    image_type: Literal["SOURCE_PRIVATE"]
    instance_count: Literal[1]
    shm_gi: Annotated[float, Field(ge=1800.0, le=1800.0)]
    spec_id: Literal["7166bd2e-6cbe-4bd9-be38-762d11003e7f"]


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    auto_fault_tolerance: Literal[False]
    command: Literal[
        "/usr/bin/bash /inspire/hdd/project/multimodal-diffusion-language-model/"
        "zhangjiaquan-253108540222/DiffRWKV-RELAY/scale/experiments/"
        "nonlatent_iclr/qualification/run_qualification.sh"
    ]
    description: Literal[
        "Single authorized nonlatent runtime qualification: one 8xH100 SXM 80GB "
        "node, 30 minutes, zero retries."
    ]
    enable_notification: Literal[False]
    enable_troubleshoot: Literal[False]
    envs: tuple[EnvironmentVariable, ...]
    fault_tolerance_max_retry: Literal[0]
    framework: Literal["pytorch"]
    framework_config: tuple[FrameworkConfig]
    logic_compute_group_id: Literal["lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"]
    max_running_time_ms: Literal["1800000"]
    name: RunId
    project_id: Literal["project-160ccb20-98ab-4538-a847-01d1f83d5b0f"]
    reserve_on_fail_ms: Literal["0"]
    reserve_on_success_ms: Literal["0"]
    task_priority: Literal[0]
    workspace_id: Literal["ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"]


class GetJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job_id: JobId


def expected_request(state: ControllerState) -> CreateJobRequest:
    envs = (
        EnvironmentVariable(name="QUALIFICATION_RUN_ID", value=state.run_id),
        EnvironmentVariable(name="QUALIFICATION_NONCE", value=state.nonce),
        EnvironmentVariable(
            name="QUALIFICATION_MANIFEST_SHA256",
            value=state.source_manifest_sha256,
        ),
        EnvironmentVariable(
            name="QUALIFICATION_JOB_RECEIPT",
            value=str(state.controller_receipt_path),
        ),
        EnvironmentVariable(name="HF_HUB_OFFLINE", value="1"),
        EnvironmentVariable(name="TRANSFORMERS_OFFLINE", value="1"),
        EnvironmentVariable(name="HF_DATASETS_OFFLINE", value="1"),
        EnvironmentVariable(name="PYTHONUNBUFFERED", value="1"),
    )
    return CreateJobRequest(
        auto_fault_tolerance=False,
        command=LAUNCH_COMMAND,
        description=DESCRIPTION,
        enable_notification=False,
        enable_troubleshoot=False,
        envs=envs,
        fault_tolerance_max_retry=0,
        framework="pytorch",
        framework_config=(
            FrameworkConfig(
                image=IMAGE,
                image_type="SOURCE_PRIVATE",
                instance_count=1,
                shm_gi=1800.0,
                spec_id=SPEC_ID,
            ),
        ),
        logic_compute_group_id=LCG_ID,
        max_running_time_ms="1800000",
        name=state.run_id,
        project_id=PROJECT_ID,
        reserve_on_fail_ms="0",
        reserve_on_success_ms="0",
        task_priority=0,
        workspace_id=WORKSPACE_ID,
    )


def validate_request(raw: bytes, context: SubmissionContext) -> CreateJobRequest:
    try:
        request = CreateJobRequest.model_validate_json(raw)
    except ValidationError as error:
        raise ControllerError(code="request_invalid") from error
    digest = sha256(raw).hexdigest()
    if (
        request != expected_request(context.state)
        or digest != context.state.submitted_request_sha256
        or digest != context.permit.submitted_request_sha256
    ):
        raise ControllerError(code="request_invalid")
    return request


def qz_arguments(raw: bytes, *, dry_run: bool) -> tuple[str, ...]:
    try:
        exact_text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ControllerError(code="request_invalid") from error
    base = ("qz", "train", "CreateJob", "--data", exact_text, "-o", "json")
    return (*base, "--dry-run") if dry_run else base


def getjob_arguments(job_id: JobId) -> tuple[str, ...]:
    payload = GetJobRequest(job_id=job_id).model_dump_json()
    return ("qz", "train", "GetJob", "--data", payload, "-o", "json")

