from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import secrets
from typing import Final
from uuid import uuid4

from pydantic import ValidationError

from controller_contract import (
    ReceiptView,
    build_request,
    serialize_request,
    validate_receipt,
    validate_request,
)
from controller_models import (
    AUTHORIZATION_ID,
    MANIFEST_SHA256,
    AdmissionRecord,
    Authorization,
    ControllerError,
    ControllerState,
    HexDigest,
    UtcTimestamp,
)


RUN_PREFIX: Final = "qualification-20260912-01-"
REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
RECEIPT_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/controller-receipts"
)


def now() -> UtcTimestamp:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _authorization_with_hash(path: Path) -> tuple[Authorization, HexDigest]:
    try:
        raw = path.read_bytes()
        authorization = Authorization.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise ControllerError(code="authorization_invalid") from error
    return authorization, sha256(raw).hexdigest()


def load_authorization(path: Path) -> Authorization:
    authorization, _ = _authorization_with_hash(path)
    return authorization


def expected_paths(state_path: Path, run_id: str) -> tuple[Path, Path]:
    return (
        state_path.parent / "exact_job_spec.json",
        RECEIPT_ROOT / f"{run_id}.json",
    )


def request_path(state: ControllerState) -> Path:
    if state.request_path.is_absolute():
        return state.request_path
    return REPOSITORY_ROOT / state.request_path


def load_state(state_path: Path) -> ControllerState:
    try:
        state = ControllerState.model_validate_json(state_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ControllerError(code="existing_state_ambiguous") from error
    expected_request, receipt_path = expected_paths(
        state_path.resolve(strict=False), state.run_id
    )
    if (
        request_path(state) != expected_request
        or state.controller_receipt_path != receipt_path
    ):
        raise ControllerError(code="existing_state_ambiguous")
    return state


def write_exclusive(path: Path, payload: bytes, error_code: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise ControllerError(code=error_code) from error


def reserve(authorization_path: Path, state_path: Path) -> ControllerState:
    _, authorization_sha256 = _authorization_with_hash(authorization_path)
    if state_path.is_symlink():
        raise ControllerError(code="existing_state_ambiguous")
    if state_path.exists():
        return load_state(state_path)
    sibling_names = ("exact_job_spec.json", "request-attempt.json", "admitted-job.json")
    if any((state_path.parent / name).exists() for name in sibling_names):
        raise ControllerError(code="existing_state_ambiguous")
    run_id = f"{RUN_PREFIX}{uuid4()}"
    request_path, receipt_path = expected_paths(state_path, run_id)
    state = ControllerState(
        authorization_id=AUTHORIZATION_ID,
        authorization_sha256=authorization_sha256,
        run_id=run_id,
        nonce=secrets.token_hex(16),
        job_name=run_id,
        request_path=request_path,
        controller_receipt_path=receipt_path,
        source_manifest_sha256=MANIFEST_SHA256,
        status="NOT_SUBMITTED",
        created_at_utc=now(),
    )
    write_exclusive(
        state_path, state.model_dump_json().encode(), "reservation_publication_failed"
    )
    return state


def request_with_digest(state: ControllerState) -> tuple[bytes, HexDigest]:
    path = request_path(state)
    try:
        if path.is_symlink():
            raise ControllerError(code="request_invalid")
        raw = path.read_bytes()
    except OSError as error:
        raise ControllerError(code="request_missing") from error
    validate_request(raw, state)
    return raw, sha256(raw).hexdigest()


def prepare_request(authorization_path: Path, state_path: Path) -> HexDigest:
    authorization = load_authorization(authorization_path)
    state = load_state(state_path)
    raw = serialize_request(build_request(authorization, state))
    validate_request(raw, state)
    digest = sha256(raw).hexdigest()
    path = request_path(state)
    if path.exists():
        existing, existing_digest = request_with_digest(state)
        if existing != raw or existing_digest != digest:
            raise ControllerError(code="request_existing_mismatch")
    else:
        write_exclusive(path, raw, "request_publication_failed")
    digest_path = path.with_suffix(".sha256")
    digest_bytes = f"{digest}  {path.name}\n".encode()
    if digest_path.exists():
        if digest_path.read_bytes() != digest_bytes:
            raise ControllerError(code="request_hash_existing_mismatch")
    else:
        write_exclusive(digest_path, digest_bytes, "request_hash_publication_failed")
    return digest


def publish_receipt(receipt: ReceiptView, destination: Path) -> None:
    payload = receipt.model_dump_json(exclude_none=False).encode()
    admission = AdmissionRecord(
        job_id=receipt.job_id,
        submitted_request_sha256=receipt.submitted_request_sha256,
        run_id=receipt.run_id,
    )
    validate_receipt(payload, admission)
    temporary = destination.parent / f".{destination.name}.{receipt.nonce}.tmp"
    write_exclusive(temporary, payload, "receipt_publication_failed")
    try:
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    except OSError as error:
        raise ControllerError(code="receipt_publication_failed") from error
