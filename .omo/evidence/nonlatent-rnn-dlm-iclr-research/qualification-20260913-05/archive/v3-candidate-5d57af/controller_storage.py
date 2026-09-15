from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from authority_models import (
    CampaignAuthorization,
    ControllerState,
    Permit,
    SubmissionContext,
)
from controller_models import AdmissionRecord, ControllerError
from receipt_contract import ReceiptView, validate_legacy_runtime_receipt


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
CAMPAIGN_PATH: Final = Path(
    ".omo/authorizations/nonlatent-gpu-campaign-20260912.json"
)
RECEIPT_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/controller-receipts"
)
ROOT_CONFIRMATION: Final = "ROOT_PASS_CONFIRMED"


def resolve_repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_regular(path: Path, error_code: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise ControllerError(code=error_code)
        return path.read_bytes()
    except OSError as error:
        raise ControllerError(code=error_code) from error


def _parse_context_files(
    campaign_path: Path, permit_path: Path, state_path: Path
) -> tuple[CampaignAuthorization, Permit, ControllerState, bytes, bytes]:
    campaign_raw = _read_regular(campaign_path, "campaign_authority_invalid")
    permit_raw = _read_regular(permit_path, "permit_invalid")
    state_raw = _read_regular(state_path, "state_invalid")
    try:
        campaign = CampaignAuthorization.model_validate_json(campaign_raw)
        permit = Permit.model_validate_json(permit_raw)
        state = ControllerState.model_validate_json(state_raw)
    except ValidationError as error:
        code = (
            "campaign_authority_invalid"
            if campaign_path.resolve(strict=False)
            != (REPOSITORY_ROOT / CAMPAIGN_PATH).resolve(strict=False)
            else "submission_context_invalid"
        )
        raise ControllerError(code=code) from error
    return campaign, permit, state, campaign_raw, permit_raw


def load_submission_context(
    campaign_path: Path, permit_path: Path, state_path: Path
) -> SubmissionContext:
    campaign, permit, state, campaign_raw, _ = _parse_context_files(
        campaign_path, permit_path, state_path
    )
    request_path = resolve_repository_path(state.request_path)
    expected_request = state_path.parent / "exact_job_spec.json"
    expected_permit = state_path.parent / "permit.json"
    expected_receipt = RECEIPT_ROOT / f"{state.run_id}.json"
    request_raw = _read_regular(request_path, "request_invalid")
    campaign_digest = sha256(campaign_raw).hexdigest()
    request_digest = sha256(request_raw).hexdigest()
    paths_match = (
        campaign_path.resolve(strict=False)
        == (REPOSITORY_ROOT / CAMPAIGN_PATH).resolve(strict=False)
        and resolve_repository_path(permit.campaign_authorization_path)
        == campaign_path.resolve(strict=False)
        and resolve_repository_path(state.permit_path)
        == expected_permit.resolve(strict=False)
        and permit_path.resolve(strict=False) == expected_permit.resolve(strict=False)
        and request_path.resolve(strict=False) == expected_request.resolve(strict=False)
        and state.controller_receipt_path == expected_receipt
        and permit.controller_receipt_path == expected_receipt
    )
    identities_match = (
        campaign.authorization_id == state.campaign_authorization_id
        and campaign.authorization_id == permit.campaign_authorization_id
        and permit.permit_id == state.permit_id
        and permit.grant_instance_id == state.grant_instance_id
        and permit.legacy_runtime_profile_id == state.legacy_runtime_profile_id
        and permit.run_id == state.run_id
        and permit.nonce == state.nonce
        and state.job_name == state.run_id
        and permit.source_manifest_sha256 == state.source_manifest_sha256
    )
    digests_match = (
        campaign_digest == permit.campaign_authorization_sha256
        and campaign_digest == state.campaign_authorization_sha256
        and request_digest == permit.submitted_request_sha256
        and request_digest == state.submitted_request_sha256
    )
    if not paths_match or not identities_match or not digests_match:
        raise ControllerError(code="submission_context_invalid")
    return SubmissionContext(
        campaign=campaign,
        permit=permit,
        state=state,
        request_bytes=request_raw,
    )


def require_live_submission(
    context: SubmissionContext, root_confirmation: str
) -> None:
    permit = context.permit
    ready = (
        root_confirmation == ROOT_CONFIRMATION
        and permit.controller_review_verdict == "PASS"
        and permit.prior_ambiguous_reconciliation_verdict == "CLEARED_NO_MATCH"
        and permit.live_submission_verdict == "PASS"
        and permit.status == "READY_FOR_SINGLE_SUBMISSION"
    )
    if not ready:
        raise ControllerError(code="live_submission_not_authorized")


def write_private_exclusive(path: Path, payload: bytes, error_code: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise ControllerError(code=error_code) from error


def load_admission(path: Path) -> AdmissionRecord:
    try:
        return AdmissionRecord.model_validate_json(
            _read_regular(path, "admission_record_invalid")
        )
    except ValidationError as error:
        raise ControllerError(code="admission_record_invalid") from error


def publish_receipt(
    receipt: ReceiptView, admission: AdmissionRecord, destination: Path
) -> None:
    payload = receipt.model_dump_json(exclude_none=False).encode("utf-8")
    validate_legacy_runtime_receipt(payload, admission)
    temporary = destination.parent / f".{destination.name}.{receipt.nonce}.tmp"
    write_private_exclusive(temporary, payload, "receipt_publication_failed")
    try:
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    except OSError as error:
        raise ControllerError(code="receipt_publication_failed") from error
