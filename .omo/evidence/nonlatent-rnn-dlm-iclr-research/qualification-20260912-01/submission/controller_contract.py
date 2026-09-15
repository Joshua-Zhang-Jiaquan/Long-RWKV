from __future__ import annotations

import importlib
from pathlib import Path
import re
import sys
from typing import Final, Protocol, runtime_checkable

from pydantic import ConfigDict, JsonValue, RootModel, ValidationError

from controller_models import (
    AUTHORIZATION_ID,
    DESCRIPTION,
    GPU_TYPE,
    IMAGE,
    IMAGE_ID,
    LAUNCH_COMMAND,
    LCG_ID,
    PROJECT_ID,
    SPEC_ID,
    WORKSPACE_ID,
    AdmissionRecord,
    Authorization,
    ControllerError,
    ControllerReceiptDraft,
    ControllerState,
    CreateJobRequest,
    EnvironmentVariable,
    FrameworkConfig,
    HexDigest,
    JobId,
    Nonce,
    RunId,
)

JOB_FIELD_PATTERN: Final = re.compile(rb'"job_id"\s*:\s*"([^"]*)"')


class JsonDocument(RootModel[JsonValue]):
    model_config = ConfigDict(frozen=True, strict=True)


class JobIdDocument(RootModel[JobId]):
    model_config = ConfigDict(frozen=True, strict=True)


class ReceiptView(Protocol):
    job_id: JobId
    submitted_request_sha256: HexDigest
    run_id: RunId
    nonce: Nonce
    observed_image_digest: None

    def model_dump_json(self, *, exclude_none: bool = False) -> str: ...

    def model_dump(self) -> dict[str, JsonValue]: ...


class ReceiptValidator(Protocol):
    def model_validate_json(self, raw: bytes) -> ReceiptView: ...


@runtime_checkable
class ReceiptModule(Protocol):
    ControllerReceipt: ReceiptValidator


def _receipt_model() -> ReceiptValidator:
    repository_root = Path(__file__).resolve().parents[5]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    try:
        module = importlib.import_module(
            "scale.experiments.nonlatent_iclr.qualification.controller_receipt"
        )
    except ModuleNotFoundError as error:
        raise ControllerError(code="receipt_model_unavailable") from error
    if not isinstance(module, ReceiptModule):
        raise ControllerError(code="receipt_model_invalid")
    return module.ControllerReceipt


def _request_for_state(state: ControllerState) -> CreateJobRequest:
    envs = (
        EnvironmentVariable(name="QUALIFICATION_RUN_ID", value=state.run_id),
        EnvironmentVariable(name="QUALIFICATION_NONCE", value=state.nonce),
        EnvironmentVariable(
            name="QUALIFICATION_MANIFEST_SHA256", value=state.source_manifest_sha256
        ),
        EnvironmentVariable(
            name="QUALIFICATION_JOB_RECEIPT", value=str(state.controller_receipt_path)
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
        name=state.job_name,
        project_id=PROJECT_ID,
        reserve_on_fail_ms="0",
        reserve_on_success_ms="0",
        task_priority=0,
        workspace_id=WORKSPACE_ID,
    )


def build_request(
    authorization: Authorization, state: ControllerState
) -> CreateJobRequest:
    if authorization.authorization_id != state.authorization_id:
        raise ControllerError(code="authorization_state_mismatch")
    return _request_for_state(state)


def serialize_request(request: CreateJobRequest) -> bytes:
    return request.model_dump_json().encode("utf-8")


def validate_request(raw: bytes, state: ControllerState) -> CreateJobRequest:
    try:
        request = CreateJobRequest.model_validate_json(raw)
    except ValidationError as error:
        raise ControllerError(code="request_invalid") from error
    if request != _request_for_state(state):
        raise ControllerError(code="request_invalid")
    return request


def qz_arguments(raw: bytes, *, dry_run: bool) -> tuple[str, ...]:
    try:
        exact_text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ControllerError(code="request_invalid") from error
    base = ("qz", "train", "CreateJob", "--data", exact_text, "-o", "json")
    return (*base, "--dry-run") if dry_run else base


def extract_job_id(response: bytes) -> JobId:
    if not response:
        raise ControllerError(code="createjob_response_missing")
    try:
        JsonDocument.model_validate_json(response)
    except ValidationError as error:
        raise ControllerError(code="createjob_response_invalid") from error
    matches = JOB_FIELD_PATTERN.findall(response)
    if not matches:
        raise ControllerError(code="createjob_response_missing")
    if len(matches) != 1:
        raise ControllerError(code="createjob_response_multiple_ids")
    try:
        return JobIdDocument.model_validate(
            matches[0].decode("ascii"), strict=True
        ).root
    except (UnicodeDecodeError, ValidationError) as error:
        raise ControllerError(code="createjob_response_invalid") from error


def build_receipt(
    admission: AdmissionRecord, state: ControllerState
) -> ReceiptView:
    if admission.run_id != state.run_id:
        raise ControllerError(code="receipt_invalid")
    draft = ControllerReceiptDraft(
        job_id=admission.job_id,
        job_identity_origin="controller_qz_createjob_receipt",
        submitted_request_sha256=admission.submitted_request_sha256,
        authorization_id=AUTHORIZATION_ID,
        project_id=PROJECT_ID,
        workspace_id=WORKSPACE_ID,
        logic_compute_group_id=LCG_ID,
        spec_id=SPEC_ID,
        run_id=state.run_id,
        nonce=state.nonce,
        source_manifest_sha256=state.source_manifest_sha256,
        requested_image=IMAGE,
        scheduler_image_id=IMAGE_ID,
        observed_image_digest=None,
        authorized_gpu_type=GPU_TYPE,
        requested_nodes=1,
        requested_gpus_per_node=8,
        requested_gpus=8,
        maximum_runtime_seconds=1800,
        maximum_gpu_hours=4,
        maximum_job_submissions=1,
        maximum_automatic_retries=0,
    )
    return _receipt_model().model_validate_json(
        draft.model_dump_json(exclude_none=False).encode()
    )


def validate_receipt(raw: bytes, admission: AdmissionRecord) -> ReceiptView:
    try:
        receipt = _receipt_model().model_validate_json(raw)
    except ValidationError as error:
        raise ControllerError(code="receipt_invalid") from error
    if (
        receipt.job_id != admission.job_id
        or receipt.run_id != admission.run_id
        or receipt.submitted_request_sha256 != admission.submitted_request_sha256
    ):
        raise ControllerError(code="receipt_invalid")
    return receipt
