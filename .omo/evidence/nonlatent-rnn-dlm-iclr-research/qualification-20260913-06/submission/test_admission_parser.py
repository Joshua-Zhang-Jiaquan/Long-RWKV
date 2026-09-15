from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import controller_models as models
import admission_parser as subject


FIXTURES: Final = Path(__file__).parent / "fixtures"
JOB_ID: Final = "job-01234567-89ab-4cde-8fab-0123456789ab"
OTHER_JOB_ID: Final = "job-fedcba98-7654-4321-8abc-ba9876543210"


def _completed(stdout: bytes, returncode: int = 0) -> models.CompletedProcess:
    return models.CompletedProcess(returncode=returncode, stdout=stdout, stderr=b"")


def test_result_envelope_is_admitted_when_job_id_is_unique() -> None:
    # Given
    outcome = _completed((FIXTURES / "createjob-success-result.json").read_bytes())

    # When
    decision = subject.classify_createjob(outcome)

    # Then
    assert isinstance(decision, models.Admitted)
    assert decision.job_id == JOB_ID
    assert decision.identity_path == "Result.job_id"


def test_direct_envelope_is_admitted_when_job_id_is_unique() -> None:
    # Given
    outcome = _completed((FIXTURES / "createjob-success-direct.json").read_bytes())

    # When
    decision = subject.classify_createjob(outcome)

    # Then
    assert isinstance(decision, models.Admitted)
    assert decision.job_id == JOB_ID
    assert decision.identity_path == "job_id"


def test_business_error_is_rejected_when_process_exit_is_zero() -> None:
    # Given
    outcome = _completed((FIXTURES / "createjob-business-error.json").read_bytes())

    # When
    decision = subject.classify_createjob(outcome)

    # Then
    assert decision == models.Rejected(
        error_code="QuotaExceeded",
        message="parent project quota insufficient",
    )


def test_empty_response_is_unknown_when_process_exit_is_zero() -> None:
    # Given
    outcome = _completed(b"")

    # When
    decision = subject.classify_createjob(outcome)

    # Then
    assert decision == models.Unknown(reason="createjob_empty_response")


def test_unrecognized_json_is_unknown_when_job_id_is_missing() -> None:
    # Given
    outcome = _completed((FIXTURES / "createjob-unknown.json").read_bytes())

    # When
    decision = subject.classify_createjob(outcome)

    # Then
    assert decision == models.Unknown(reason="createjob_success_missing_job_id")


def test_non_json_response_is_unknown() -> None:
    # Given
    outcome = _completed(b"upstream gateway returned no envelope")

    # When
    decision = subject.classify_createjob(outcome)

    # Then
    assert decision == models.Unknown(reason="createjob_unrecognized_response")


def test_distinct_explicit_job_ids_are_unknown() -> None:
    # Given
    response = json.dumps(
        {"job_id": JOB_ID, "Result": {"job_id": OTHER_JOB_ID}}
    ).encode()

    # When
    decision = subject.classify_createjob(_completed(response))

    # Then
    assert decision == models.Unknown(reason="createjob_multiple_job_ids")


def test_invalid_explicit_job_id_is_unknown() -> None:
    # Given
    outcome = _completed(b'{"Result":{"job_id":"invalid"}}')

    # When
    decision = subject.classify_createjob(outcome)

    # Then
    assert decision == models.Unknown(reason="createjob_invalid_job_id")


def test_job_id_inside_error_payload_never_counts_as_admission() -> None:
    # Given
    response = json.dumps(
        {
            "ResponseMetadata": {
                "Action": "CreateJob",
                "Error": {
                    "Code": "Rejected",
                    "Message": f"reference {OTHER_JOB_ID}",
                },
            },
            "Result": {"reference_job_id": OTHER_JOB_ID},
        }
    ).encode()

    # When
    decision = subject.classify_createjob(_completed(response))

    # Then
    assert isinstance(decision, models.Rejected)


def test_timeout_is_unknown_even_with_partial_job_like_output() -> None:
    # Given
    outcome = models.TimedOutProcess(
        timeout_seconds=120,
        stdout=f'{{"job_id":"{JOB_ID}"'.encode(),
        stderr=b"partial",
    )

    # When
    decision = subject.classify_createjob(outcome)

    # Then
    assert decision == models.Unknown(reason="createjob_timeout_no_retry")


def test_getjob_confirms_returned_identity_and_exact_caps() -> None:
    # Given
    expectation = models.JobExpectation(
        job_id=JOB_ID,
        name="qualification-20260912-01-00000000-0000-4000-8000-000000000000",
    )
    response = json.dumps(
        {
            "ResponseMetadata": {"Action": "GetJob"},
            "Result": {
                "job_id": JOB_ID,
                "name": expectation.name,
                "project_id": models.PROJECT_ID,
                "workspace_id": models.WORKSPACE_ID,
                "logic_compute_group_id": models.LCG_ID,
                "max_running_time_ms": "1800000",
                "auto_fault_tolerance": False,
                "fault_tolerance_max_retry": 0,
                "task_priority": 0,
                "status": "job_queuing",
                "framework_config": [
                    {
                        "image": models.IMAGE,
                        "image_type": "SOURCE_PRIVATE",
                        "instance_count": 1,
                        "shm_gi": 1800,
                        "instance_spec_price_info": {
                            "gpu_count": 8,
                            "gpu_info": {"gpu_type": models.GPU_TYPE},
                            "quota_id": models.SPEC_ID,
                        },
                    }
                ],
            },
        }
    ).encode()

    # When
    result = subject.confirm_getjob(_completed(response), expectation)

    # Then
    assert result == models.ConfirmedJob(job_id=JOB_ID, status="job_queuing")


def test_getjob_refuses_a_different_returned_job_id() -> None:
    # Given
    expectation = models.JobExpectation(
        job_id=JOB_ID,
        name="qualification-20260912-01-00000000-0000-4000-8000-000000000000",
    )
    response = json.dumps(
        {
            "ResponseMetadata": {"Action": "GetJob"},
            "Result": {"job_id": OTHER_JOB_ID, "name": expectation.name},
        }
    ).encode()

    # When
    result = subject.confirm_getjob(_completed(response), expectation)

    # Then
    assert result == models.Unknown(reason="getjob_binding_mismatch")
