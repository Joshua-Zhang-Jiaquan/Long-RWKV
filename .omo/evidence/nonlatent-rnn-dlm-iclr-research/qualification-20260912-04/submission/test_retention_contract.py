from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

import admission_parser
import authority_models
import controller_models as models
import request_contract as subject


SUBMISSION: Final = Path(__file__).parent
FIXTURES: Final = SUBMISSION / "fixtures"


def _explicit_zero_retention_request() -> bytes:
    raw_request = (SUBMISSION / "exact_job_spec.json").read_bytes()
    task_priority = b',"task_priority":4'
    assert raw_request.count(task_priority) == 1
    return raw_request.replace(
        task_priority,
        b',"reserve_on_fail_ms":"0","reserve_on_success_ms":"0"'
        + task_priority,
    )


def test_createjob_request_serialization_omits_optional_retention_fields() -> None:
    # Given
    raw_request = (SUBMISSION / "exact_job_spec.json").read_bytes()

    # When
    request = subject.CreateJobRequest.model_validate_json(raw_request)
    serialized = request.model_dump_json()

    # Then
    assert "reserve_on_fail_ms" not in serialized
    assert "reserve_on_success_ms" not in serialized


def test_createjob_request_rejects_explicit_zero_retention_fields() -> None:
    # Given
    raw_request = _explicit_zero_retention_request()

    # When / Then
    with pytest.raises(ValidationError):
        subject.CreateJobRequest.model_validate_json(raw_request)


def test_permit_records_undocumented_server_default_as_unknown() -> None:
    # Given
    raw_permit = (SUBMISSION / "permit.json").read_bytes()

    # When
    permit = authority_models.Permit.model_validate_json(raw_permit)

    # Then
    assert permit.retention_configuration == "SERVER_DEFAULT_UNDOCUMENTED"
    assert permit.reserve_on_fail_ms is None
    assert permit.reserve_on_success_ms is None


def test_captured_retention_error_remains_an_explicit_rejection() -> None:
    # Given
    outcome = models.CompletedProcess(
        returncode=0,
        stdout=(FIXTURES / "createjob-invalid-retention.json").read_bytes(),
        stderr=b"",
    )

    # When
    decision = admission_parser.classify_createjob(outcome)

    # Then
    assert decision == models.Rejected(
        error_code="InvalidParameter",
        message="retention time must be positive",
    )


def test_getjob_contract_does_not_synthesize_unknown_retention_values() -> None:
    # Given / When
    response_fields = admission_parser.GetJobResult.model_fields

    # Then
    assert "reserve_on_fail_ms" not in response_fields
    assert "reserve_on_success_ms" not in response_fields
