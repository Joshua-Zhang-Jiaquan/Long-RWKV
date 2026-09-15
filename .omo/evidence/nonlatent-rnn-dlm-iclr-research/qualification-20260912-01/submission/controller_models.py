from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


AUTHORIZATION_ID: Final = "nonlatent-h100-qualification-20260912-01"
MANIFEST_SHA256: Final = (
    "92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c"
)
IMAGE: Final = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
IMAGE_ID: Final = "image-7330d118-df9d-4e2e-82b1-c2543e831eb4"
PROJECT_ID: Final = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
WORKSPACE_ID: Final = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
LCG_ID: Final = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
SPEC_ID: Final = "7166bd2e-6cbe-4bd9-be38-762d11003e7f"
GPU_TYPE: Final = "NVIDIA_H100_SXM_80G"
LAUNCH_COMMAND: Final = (
    "/usr/bin/bash /inspire/hdd/project/multimodal-diffusion-language-model/"
    "zhangjiaquan-253108540222/DiffRWKV-RELAY/scale/experiments/"
    "nonlatent_iclr/qualification/run_qualification.sh"
)
DESCRIPTION: Final = (
    "Single authorized nonlatent runtime qualification: one 8xH100 SXM 80GB "
    "node, 30 minutes, zero retries."
)
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
type UtcTimestamp = Annotated[
    str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
]


class Authorization(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    schema_version: Literal[1]
    authorization_id: Literal["nonlatent-h100-qualification-20260912-01"]
    project_id: Literal["project-160ccb20-98ab-4538-a847-01d1f83d5b0f"]
    workspace_id: Literal["ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"]
    logic_compute_group_id: Literal["lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"]
    image: Literal["docker.sii.shaipower.online/inspire-studio/relay2:v2"]
    required_gpu_type: Literal["NVIDIA_H100_SXM_80G"]
    maximum_nodes: Literal[1]
    gpus_per_node: Literal[8]
    maximum_gpus: Literal[8]
    maximum_runtime_ms: Literal[1_800_000]
    maximum_gpu_hours: Literal[4]
    maximum_jobs: Literal[1]
    automatic_fault_tolerance: Literal[False]
    maximum_automatic_retries: Literal[0]
    maximum_job_submissions: Literal[1]
    resolved_spec_id: Literal["7166bd2e-6cbe-4bd9-be38-762d11003e7f"]
    preflight_verdict: str
    jobs_submitted: Literal[0]
    status: str


class ControllerState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    authorization_id: Literal["nonlatent-h100-qualification-20260912-01"]
    authorization_sha256: HexDigest
    run_id: RunId
    nonce: Nonce
    job_name: RunId
    request_path: Path
    controller_receipt_path: Path
    source_manifest_sha256: Literal[
        "92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c"
    ]
    status: Literal["NOT_SUBMITTED"]
    created_at_utc: UtcTimestamp


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


class AdmissionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    authorization_id: Literal["nonlatent-h100-qualification-20260912-01"] = (
        AUTHORIZATION_ID
    )
    job_id: JobId
    submitted_request_sha256: HexDigest
    run_id: RunId
    status: Literal["ADMITTED_RECEIPT_PENDING"] = "ADMITTED_RECEIPT_PENDING"


class AttemptContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: Path
    run_id: RunId
    submitted_request_sha256: HexDigest


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    authorization_id: Literal["nonlatent-h100-qualification-20260912-01"] = (
        AUTHORIZATION_ID
    )
    run_id: RunId
    submitted_request_sha256: HexDigest
    status: Literal["ATTEMPT_RECORDED_BEFORE_CREATEJOB"] = (
        "ATTEMPT_RECORDED_BEFORE_CREATEJOB"
    )
    maximum_job_submissions: Literal[1] = 1
    automatic_retries: Literal[0] = 0
    recorded_at_utc: UtcTimestamp


class DryRunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    authorization_id: Literal["nonlatent-h100-qualification-20260912-01"] = (
        AUTHORIZATION_ID
    )
    run_id: RunId
    submitted_request_sha256: HexDigest
    literal_dry_run: Literal[True] = True
    request_validated: bool
    qz_exit_code: int
    stdout_sha256: HexDigest
    stdout_bytes: int
    stderr_sha256: HexDigest
    stderr_bytes: int
    valid_job_ids_observed: int
    admission_verified: Literal[False] = False
    capacity_verified: Literal[False] = False
    recorded_at_utc: UtcTimestamp


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


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class ProcessRunner(Protocol):
    def __call__(self, arguments: tuple[str, ...]) -> ProcessResult: ...
