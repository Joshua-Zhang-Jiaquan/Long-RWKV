from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Final

import pytest

import controller as subject


ROOT: Final = Path(
    "/inspire/hdd/project/multimodal-diffusion-language-model/"
    "zhangjiaquan-253108540222/DiffRWKV-RELAY"
)
AUTHORIZATION: Final = (
    ROOT / ".omo/authorizations/nonlatent-h100-qualification-20260912-01.json"
)
RUN_ID: Final = "qualification-20260912-01-00000000-0000-4000-8000-000000000000"
JOB_ID: Final = "job-01234567-89ab-4cde-8fab-0123456789ab"
OTHER_JOB_ID: Final = "job-fedcba98-7654-4321-8abc-ba9876543210"
MANIFEST_SHA256: Final = (
    "92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c"
)
RECEIPT_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/controller-receipts"
)


def _state(tmp_path: Path) -> subject.ControllerState:
    return subject.ControllerState(
        authorization_id="nonlatent-h100-qualification-20260912-01",
        authorization_sha256=sha256(AUTHORIZATION.read_bytes()).hexdigest(),
        run_id=RUN_ID,
        nonce="0" * 32,
        job_name=RUN_ID,
        request_path=tmp_path / "exact_job_spec.json",
        controller_receipt_path=RECEIPT_ROOT / f"{RUN_ID}.json",
        source_manifest_sha256=MANIFEST_SHA256,
        status="NOT_SUBMITTED",
        created_at_utc="2026-09-12T00:00:00Z",
    )


def _request_bytes(tmp_path: Path) -> tuple[subject.ControllerState, bytes]:
    state = _state(tmp_path)
    authorization = subject.load_authorization(AUTHORIZATION)
    request = subject.build_request(authorization, state)
    return state, subject.serialize_request(request)


def test_request_body_enforces_authorized_caps(tmp_path: Path) -> None:
    # Given
    state = _state(tmp_path)
    authorization = subject.load_authorization(AUTHORIZATION)

    # When
    request = subject.build_request(authorization, state)

    # Then
    assert request.name == RUN_ID
    assert request.command == (
        "/usr/bin/bash /inspire/hdd/project/multimodal-diffusion-language-model/"
        "zhangjiaquan-253108540222/DiffRWKV-RELAY/scale/experiments/"
        "nonlatent_iclr/qualification/run_qualification.sh"
    )
    assert request.max_running_time_ms == "1800000"
    assert request.reserve_on_fail_ms == "0"
    assert request.reserve_on_success_ms == "0"
    assert request.auto_fault_tolerance is False
    assert request.fault_tolerance_max_retry == 0
    assert request.task_priority == 0
    assert len(request.framework_config) == 1
    framework = request.framework_config[0]
    assert framework.instance_count == 1
    assert framework.shm_gi == 1800.0
    assert framework.spec_id == "7166bd2e-6cbe-4bd9-be38-762d11003e7f"
    assert framework.image_type == "SOURCE_PRIVATE"
    assert framework.image == authorization.image
    environment = {item.name: item.value for item in request.envs}
    assert environment == {
        "QUALIFICATION_RUN_ID": RUN_ID,
        "QUALIFICATION_NONCE": "0" * 32,
        "QUALIFICATION_MANIFEST_SHA256": MANIFEST_SHA256,
        "QUALIFICATION_JOB_RECEIPT": str(state.controller_receipt_path),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    assert 1 * 8 * 1800 / 3600 == 4


def test_request_validation_refuses_mismatched_tuple(tmp_path: Path) -> None:
    # Given
    state, raw = _request_bytes(tmp_path)
    mismatched = raw.replace(
        b"project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
        b"project-00000000-0000-0000-0000-000000000000",
    )

    # When / Then
    with pytest.raises(subject.ControllerError, match="request_invalid"):
        subject.validate_request(mismatched, state)


def test_success_response_yields_exact_job_id() -> None:
    # Given
    response = json.dumps({"job_id": JOB_ID, "status": "QUEUING"}).encode()

    # When
    job_id = subject.extract_job_id(response)

    # Then
    assert job_id == JOB_ID


def test_dry_run_arguments_preserve_exact_request_bytes(tmp_path: Path) -> None:
    # Given
    _, raw = _request_bytes(tmp_path)

    # When
    arguments = subject.qz_arguments(raw, dry_run=True)

    # Then
    assert arguments.count("--dry-run") == 1
    assert arguments[arguments.index("--data") + 1].encode() == raw
    assert arguments[0:3] == ("qz", "train", "CreateJob")


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (b"", "createjob_response_missing"),
        (b'{"job_id":"invalid"}', "createjob_response_invalid"),
        (
            json.dumps({"job_id": JOB_ID, "other": {"job_id": OTHER_JOB_ID}}).encode(),
            "createjob_response_multiple_ids",
        ),
    ],
)
def test_response_refuses_missing_invalid_or_multiple_ids(
    response: bytes, code: str
) -> None:
    # Given / When / Then
    with pytest.raises(subject.ControllerError, match=code):
        subject.extract_job_id(response)


def test_receipt_validates_successful_fixture_binding(tmp_path: Path) -> None:
    # Given
    state, raw = _request_bytes(tmp_path)
    admission = subject.AdmissionRecord(
        job_id=JOB_ID,
        submitted_request_sha256=sha256(raw).hexdigest(),
        run_id=state.run_id,
    )

    # When
    receipt = subject.build_receipt(admission, state)
    validated = subject.validate_receipt(receipt.model_dump_json().encode(), admission)

    # Then
    assert validated.__class__.__module__ == (
        "scale.experiments.nonlatent_iclr.qualification.controller_receipt"
    )
    assert validated.__class__.__name__ == "ControllerReceipt"
    assert validated.job_id == JOB_ID
    assert validated.submitted_request_sha256 == sha256(raw).hexdigest()
    assert b'"observed_image_digest":null' in validated.model_dump_json(
        exclude_none=False
    ).encode()


def test_receipt_refuses_mismatched_authorization_tuple(tmp_path: Path) -> None:
    # Given
    state, raw = _request_bytes(tmp_path)
    admission = subject.AdmissionRecord(
        job_id=JOB_ID,
        submitted_request_sha256=sha256(raw).hexdigest(),
        run_id=state.run_id,
    )
    payload = subject.build_receipt(admission, state).model_dump_json(
        exclude_none=False
    ).encode()
    mismatched = payload.replace(
        b"project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
        b"project-00000000-0000-0000-0000-000000000000",
    )

    # When / Then
    with pytest.raises(subject.ControllerError, match="receipt_invalid"):
        subject.validate_receipt(mismatched, admission)


def test_receipt_publication_is_no_clobber(tmp_path: Path) -> None:
    # Given
    state, raw = _request_bytes(tmp_path)
    admission = subject.AdmissionRecord(
        job_id=JOB_ID,
        submitted_request_sha256=sha256(raw).hexdigest(),
        run_id=state.run_id,
    )
    receipt = subject.build_receipt(admission, state)
    destination = tmp_path / "fixture-controller-receipt.json"
    original = b"already-present"
    destination.write_bytes(original)

    # When / Then
    with pytest.raises(subject.ControllerError, match="receipt_publication_failed"):
        subject.publish_receipt(receipt, destination)
    assert destination.read_bytes() == original


class _FixtureRunner:
    def __init__(self, attempt_path: Path) -> None:
        self.attempt_path = attempt_path
        self.call_count = 0

    def __call__(self, arguments: tuple[str, ...]) -> subject.ProcessResult:
        assert self.attempt_path.is_file()
        self.call_count += 1
        assert arguments[0:3] == ("qz", "train", "CreateJob")
        return subject.ProcessResult(
            returncode=0,
            stdout=json.dumps({"job_id": JOB_ID}).encode(),
            stderr=b"",
        )


def test_attempt_is_recorded_before_single_fixture_call(tmp_path: Path) -> None:
    # Given
    state, raw = _request_bytes(tmp_path)
    context = subject.AttemptContext(
        path=tmp_path / "request-attempt.json",
        run_id=state.run_id,
        submitted_request_sha256=sha256(raw).hexdigest(),
    )
    runner = _FixtureRunner(context.path)

    # When
    job_id = subject.record_and_call_once(context, raw, runner)

    # Then
    assert job_id == JOB_ID
    assert runner.call_count == 1


def test_existing_attempt_refuses_invocation(tmp_path: Path) -> None:
    # Given
    state, raw = _request_bytes(tmp_path)
    attempt_path = tmp_path / "request-attempt.json"
    attempt_path.write_bytes(b"attempt-already-recorded")
    context = subject.AttemptContext(
        path=attempt_path,
        run_id=state.run_id,
        submitted_request_sha256=sha256(raw).hexdigest(),
    )
    runner = _FixtureRunner(attempt_path)

    # When / Then
    with pytest.raises(subject.ControllerError, match="submission_already_attempted"):
        subject.record_and_call_once(context, raw, runner)
    assert runner.call_count == 0


def test_controller_error_allows_runtime_traceback_assignment() -> None:
    # Given
    error = subject.ControllerError(code="fixture_error")

    # When
    error.__traceback__ = None

    # Then
    assert str(error) == "fixture_error"
