from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

import controller_models as models
import controller_storage as storage
import request_contract as subject


ROOT: Final = Path(
    "/inspire/hdd/project/multimodal-diffusion-language-model/"
    "zhangjiaquan-253108540222/DiffRWKV-RELAY"
)
SUBMISSION: Final = ROOT / (
    ".omo/evidence/nonlatent-rnn-dlm-iclr-research/"
    "qualification-20260912-04/submission"
)
CAMPAIGN: Final = ROOT / ".omo/authorizations/nonlatent-gpu-campaign-20260912.json"
OLD_AUTHORIZATION: Final = ROOT / (
    ".omo/authorizations/nonlatent-h100-qualification-20260912-01.json"
)
PERMIT: Final = SUBMISSION / "permit.json"
STATE: Final = SUBMISSION / "controller-state.json"


def test_context_binds_new_campaign_as_authority_and_legacy_profile_as_metadata() -> None:
    # Given / When
    context = storage.load_submission_context(CAMPAIGN, PERMIT, STATE)

    # Then
    assert context.campaign.authorization_id == models.CAMPAIGN_AUTHORIZATION_ID
    assert context.permit.campaign_authorization_id == models.CAMPAIGN_AUTHORIZATION_ID
    assert context.permit.legacy_runtime_profile_id == models.LEGACY_RUNTIME_PROFILE_ID
    assert context.permit.legacy_runtime_profile_role == (
        "PAYLOAD_RECEIPT_SCHEMA_COMPATIBILITY_ONLY_NOT_AUTHORITY"
    )
    assert context.state.permit_id == context.permit.permit_id


def test_old_consumed_authorization_cannot_act_as_campaign_authority() -> None:
    # Given / When / Then
    with pytest.raises(models.ControllerError, match="campaign_authority_invalid"):
        storage.load_submission_context(OLD_AUTHORIZATION, PERMIT, STATE)


def test_request_enforces_per_attempt_caps_without_fake_job_id() -> None:
    # Given
    context = storage.load_submission_context(CAMPAIGN, PERMIT, STATE)

    # When
    request = subject.validate_request(context.request_bytes, context)

    # Then
    serialized = request.model_dump_json()
    assert request.name == context.state.run_id
    assert request.max_running_time_ms == "1800000"
    assert request.auto_fault_tolerance is False
    assert request.fault_tolerance_max_retry == 0
    assert "reserve_on_fail_ms" not in serialized
    assert "reserve_on_success_ms" not in serialized
    assert context.permit.retention_configuration == "SERVER_DEFAULT_UNDOCUMENTED"
    assert context.permit.reserve_on_fail_ms is None
    assert context.permit.reserve_on_success_ms is None
    assert request.task_priority == 4
    assert request.framework_config[0].instance_count == 1
    assert request.framework_config[0].spec_id == models.SPEC_ID
    assert request.framework_config[0].shm_gi == 1800.0
    assert all(item.name not in {"QZ_JOB_ID", "JOB_ID"} for item in request.envs)
    assert 1 * 8 * 1800 / 3600 == 4


def test_dry_run_arguments_preserve_exact_request_and_literal_flag() -> None:
    # Given
    context = storage.load_submission_context(CAMPAIGN, PERMIT, STATE)

    # When
    arguments = subject.qz_arguments(context.request_bytes, dry_run=True)

    # Then
    assert arguments[0:3] == ("qz", "train", "CreateJob")
    assert arguments.count("--dry-run") == 1
    assert arguments[arguments.index("--data") + 1].encode() == context.request_bytes


def test_pending_permit_blocks_live_submission() -> None:
    # Given
    context = storage.load_submission_context(CAMPAIGN, PERMIT, STATE)

    # When / Then
    with pytest.raises(models.ControllerError, match="live_submission_not_authorized"):
        storage.require_live_submission(context, "ROOT_PASS_CONFIRMED")


def test_compatibility_receipt_does_not_relabel_actual_authority() -> None:
    # Given
    context = storage.load_submission_context(CAMPAIGN, PERMIT, STATE)
    admission = models.AdmissionRecord(
        campaign_authorization_id=models.CAMPAIGN_AUTHORIZATION_ID,
        permit_id=context.permit.permit_id,
        grant_instance_id=context.permit.grant_instance_id,
        job_id="job-01234567-89ab-4cde-8fab-0123456789ab",
        submitted_request_sha256=context.state.submitted_request_sha256,
        run_id=context.state.run_id,
    )

    # When
    receipt = subject.build_legacy_runtime_receipt(admission, context.state)

    # Then
    assert receipt.authorization_id == models.LEGACY_RUNTIME_PROFILE_ID
    assert admission.campaign_authorization_id == models.CAMPAIGN_AUTHORIZATION_ID
    assert admission.permit_id == context.permit.permit_id
