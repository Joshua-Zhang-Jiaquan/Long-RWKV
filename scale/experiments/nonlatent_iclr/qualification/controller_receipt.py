from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import os
from pathlib import Path
import time
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

type HexDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type Nonce = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
type JobId = Annotated[
    str,
    StringConstraints(
        pattern=r"^job-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
]
type RunId = Annotated[
    str,
    StringConstraints(pattern=r"^qualification-20260912-01-[a-z0-9-]{8,80}$"),
]


class ControllerReceiptError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ControllerReceipt(BaseModel):
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

    def canonical_sha256(self) -> str:
        serialized = self.model_dump_json(
            exclude={"controller_receipt_sha256"}, exclude_none=False
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    def to_binding_evidence(self) -> ControllerBindingEvidence:
        values = self.model_dump()
        values["controller_receipt_sha256"] = self.canonical_sha256()
        return ControllerBindingEvidence.model_validate(values)


class ControllerBindingEvidence(ControllerReceipt):
    controller_receipt_sha256: HexDigest

    @model_validator(mode="after")
    def enforce_receipt_digest(self) -> Self:
        if self.controller_receipt_sha256 != self.canonical_sha256():
            raise ValueError("controller_receipt_digest_mismatch")
        return self


class ControllerReceiptExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: Path
    run_id: RunId
    nonce: Nonce
    source_manifest_sha256: HexDigest
    timeout_seconds: float = Field(ge=0.0, le=60.0)
    poll_interval_seconds: float = Field(gt=0.0, le=1.0)


def await_controller_receipt(
    expectation: ControllerReceiptExpectation,
    *,
    environment: Mapping[str, str] | None = None,
) -> ControllerReceipt:
    deadline = time.monotonic() + expectation.timeout_seconds
    while True:
        if expectation.path.is_symlink():
            raise ControllerReceiptError("controller_receipt_not_regular")
        if expectation.path.exists():
            receipt = _parse_receipt(expectation.path)
            _validate_expected_binding(receipt, expectation)
            _validate_native_job_identity(receipt, environment)
            return receipt
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise ControllerReceiptError("controller_receipt_timeout")
        time.sleep(min(expectation.poll_interval_seconds, remaining))


def _parse_receipt(path: Path) -> ControllerReceipt:
    try:
        if not path.is_file() or path.stat().st_size > 65_536:
            raise ControllerReceiptError("controller_receipt_not_regular")
        return ControllerReceipt.model_validate_json(path.read_bytes())
    except ControllerReceiptError:
        raise
    except (OSError, ValidationError) as error:
        raise ControllerReceiptError("controller_receipt_malformed") from error


def _validate_expected_binding(
    receipt: ControllerReceipt,
    expectation: ControllerReceiptExpectation,
) -> None:
    if receipt.run_id != expectation.run_id:
        raise ControllerReceiptError("controller_receipt_run_id_mismatch")
    if receipt.nonce != expectation.nonce:
        raise ControllerReceiptError("controller_receipt_nonce_mismatch")
    if receipt.source_manifest_sha256 != expectation.source_manifest_sha256:
        raise ControllerReceiptError("controller_receipt_manifest_mismatch")


def _validate_native_job_identity(
    receipt: ControllerReceipt,
    environment: Mapping[str, str] | None,
) -> None:
    effective_environment = os.environ if environment is None else environment
    for variable in ("QZ_JOB_ID", "JOB_ID"):
        native_job_id = effective_environment.get(variable, "").strip()
        if native_job_id and native_job_id != receipt.job_id:
            raise ControllerReceiptError("native_job_identity_mismatch")
