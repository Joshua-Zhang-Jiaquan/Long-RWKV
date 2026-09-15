from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from controller_models import (
    GrantInstanceId,
    HexDigest,
    Nonce,
    PermitId,
    RunId,
    UtcTimestamp,
)


class ReplacementPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    nodes: Literal[1]
    gpus_per_node: Literal[8]
    gpu_type: Literal["NVIDIA_H100_SXM_80G"]
    max_running_time_ms: Literal[1_800_000]
    automatic_fault_tolerance: Literal[False]
    automatic_retries: Literal[0]


class CampaignAuthorization(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    schema_version: Literal[1]
    authorization_id: Literal["nonlatent-gpu-campaign-20260912"]
    cumulative_gpu_hours: None
    cumulative_gpu_hours_policy: Literal["no_user_imposed_ceiling"]
    project_id: Literal["project-160ccb20-98ab-4538-a847-01d1f83d5b0f"]
    workspace_id: Literal["ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"]
    logic_compute_group_id: Literal["lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"]
    image: Literal["docker.sii.shaipower.online/inspire-studio/relay2:v2"]
    replaces_cumulative_and_attempt_limit_of: Literal[
        "nonlatent-h100-qualification-20260912-01"
    ]
    historical_attempt_record_must_be_preserved: Literal[True]
    replacement_qualification_authorized: Literal[True]
    initial_replacement_policy: ReplacementPolicy
    status: Literal["authorized_for_staged_execution"]


class Permit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    permit_id: PermitId
    grant_instance_id: GrantInstanceId
    campaign_authorization_id: Literal["nonlatent-gpu-campaign-20260912"]
    campaign_authorization_path: Path
    campaign_authorization_sha256: HexDigest
    legacy_runtime_profile_id: Literal["nonlatent-h100-qualification-20260912-01"]
    legacy_runtime_profile_role: Literal[
        "PAYLOAD_RECEIPT_SCHEMA_COMPATIBILITY_ONLY_NOT_AUTHORITY"
    ]
    replaces_ambiguous_run_id: RunId
    preserves_prior_attempt_as_consumed_or_unknown: Literal[True]
    run_id: RunId
    nonce: Nonce
    submitted_request_sha256: HexDigest
    source_manifest_sha256: Literal[
        "4777fd86769f9264af91241c82655a9b8f51874500e79fbd92f9b19030003f9f"
    ]
    controller_receipt_path: Path
    project_id: Literal["project-160ccb20-98ab-4538-a847-01d1f83d5b0f"]
    workspace_id: Literal["ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"]
    logic_compute_group_id: Literal["lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"]
    spec_id: Literal["7166bd2e-6cbe-4bd9-be38-762d11003e7f"]
    image: Literal["docker.sii.shaipower.online/inspire-studio/relay2:v2"]
    scheduler_image_id: Literal["image-7330d118-df9d-4e2e-82b1-c2543e831eb4"]
    observed_image_digest: None
    gpu_type: Literal["NVIDIA_H100_SXM_80G"]
    nodes: Literal[1]
    gpus_per_node: Literal[8]
    maximum_gpus: Literal[8]
    maximum_runtime_ms: Literal[1_800_000]
    maximum_gpu_hours: Literal[4]
    automatic_fault_tolerance: Literal[False]
    maximum_automatic_retries: Literal[0]
    maximum_real_createjob_calls: Literal[1]
    reserve_on_fail_ms: None
    reserve_on_success_ms: None
    retention_configuration: Literal["SERVER_DEFAULT_UNDOCUMENTED"]
    controller_review_verdict: Literal["PENDING", "PASS"]
    prior_ambiguous_reconciliation_verdict: Literal["PENDING", "CLEARED_NO_MATCH"]
    live_submission_verdict: Literal["NO_GO", "PASS"]
    status: Literal["PREPARED_FOR_ROOT_REVIEW", "READY_FOR_SINGLE_SUBMISSION"]
    created_at_utc: UtcTimestamp


class ControllerState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    campaign_authorization_id: Literal["nonlatent-gpu-campaign-20260912"]
    campaign_authorization_sha256: HexDigest
    permit_id: PermitId
    grant_instance_id: GrantInstanceId
    legacy_runtime_profile_id: Literal["nonlatent-h100-qualification-20260912-01"]
    run_id: RunId
    nonce: Nonce
    job_name: RunId
    request_path: Path
    permit_path: Path
    submitted_request_sha256: HexDigest
    controller_receipt_path: Path
    source_manifest_sha256: HexDigest
    maximum_real_createjob_calls: Literal[1]
    status: Literal["NOT_SUBMITTED"]
    created_at_utc: UtcTimestamp


@dataclass(frozen=True, slots=True)
class SubmissionContext:
    campaign: CampaignAuthorization
    permit: Permit
    state: ControllerState
    request_bytes: bytes
