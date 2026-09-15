from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from authority_models import ControllerState
from controller_models import (
    CAMPAIGN_AUTHORIZATION_ID,
    GPU_TYPE,
    IMAGE,
    IMAGE_ID,
    LCG_ID,
    LEGACY_RUNTIME_PROFILE_ID,
    PROJECT_ID,
    SPEC_ID,
    WORKSPACE_ID,
    AdmissionRecord,
    ControllerError,
    HexDigest,
    JobId,
    Nonce,
    RunId,
)


class ControllerReceiptDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    job_id: JobId
    job_identity_origin: Literal["controller_qz_createjob_receipt"]
    submitted_request_sha256: HexDigest
    authorization_id: Literal["nonlatent-h100-qualification-20260912-01"]
    project_id: Literal["project-160ccb20-98ab-4538-a847-01d1f83d5b0f"]
    workspace_id: Literal["ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"]
    logic_compute_group_id: Literal["lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"]
    spec_id: Literal["7166bd2e-6cbe-4bd9-be38-762d11003e7f"]
    run_id: RunId
    nonce: Nonce
    source_manifest_sha256: HexDigest
    requested_image: Literal[
        "docker.sii.shaipower.online/inspire-studio/relay2:v2"
    ]
    scheduler_image_id: Literal["image-7330d118-df9d-4e2e-82b1-c2543e831eb4"]
    observed_image_digest: None = None
    authorized_gpu_type: Literal["NVIDIA_H100_SXM_80G"]
    requested_nodes: Literal[1]
    requested_gpus_per_node: Literal[8]
    requested_gpus: Literal[8]
    maximum_runtime_seconds: Literal[1_800]
    maximum_gpu_hours: Literal[4]
    maximum_job_submissions: Literal[1]
    maximum_automatic_retries: Literal[0]


class ReceiptView(Protocol):
    job_id: JobId
    submitted_request_sha256: HexDigest
    authorization_id: str
    run_id: RunId
    nonce: Nonce

    def model_dump_json(self, *, exclude_none: bool = False) -> str: ...


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


def build_legacy_runtime_receipt(
    admission: AdmissionRecord, state: ControllerState
) -> ReceiptView:
    if (
        admission.campaign_authorization_id != CAMPAIGN_AUTHORIZATION_ID
        or admission.run_id != state.run_id
        or admission.permit_id != state.permit_id
        or admission.grant_instance_id != state.grant_instance_id
    ):
        raise ControllerError(code="receipt_invalid")
    draft = ControllerReceiptDraft(
        job_id=admission.job_id,
        job_identity_origin="controller_qz_createjob_receipt",
        submitted_request_sha256=admission.submitted_request_sha256,
        authorization_id=LEGACY_RUNTIME_PROFILE_ID,
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
    return _receipt_model().model_validate_json(draft.model_dump_json().encode())


def validate_legacy_runtime_receipt(
    raw: bytes, admission: AdmissionRecord
) -> ReceiptView:
    try:
        receipt = _receipt_model().model_validate_json(raw)
    except ValidationError as error:
        raise ControllerError(code="receipt_invalid") from error
    if (
        receipt.job_id != admission.job_id
        or receipt.run_id != admission.run_id
        or receipt.submitted_request_sha256 != admission.submitted_request_sha256
        or receipt.authorization_id != LEGACY_RUNTIME_PROFILE_ID
    ):
        raise ControllerError(code="receipt_invalid")
    return receipt
