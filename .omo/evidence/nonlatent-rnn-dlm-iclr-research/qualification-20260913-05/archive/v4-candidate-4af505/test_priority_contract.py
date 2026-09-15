from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

import admission_parser
import controller_models as models
import request_contract as subject


SUBMISSION: Final = Path(__file__).parent
FIXTURES: Final = SUBMISSION / "fixtures"
JOB_ID: Final = "job-01234567-89ab-4cde-8fab-0123456789ab"
RUN_ID: Final = "qualification-20260912-01-00000000-0000-4000-8000-000000000000"


def _priority_zero_request() -> bytes:
    raw_request = (SUBMISSION / "exact_job_spec.json").read_bytes()
    priority_four = b'"task_priority":4'
    assert raw_request.count(priority_four) == 1
    return raw_request.replace(priority_four, b'"task_priority":0')


def test_createjob_priority_zero_is_rejected_by_preflight_contract() -> None:
    # Given
    raw_request = _priority_zero_request()

    # When / Then
    with pytest.raises(ValidationError):
        subject.CreateJobRequest.model_validate_json(raw_request)


def test_createjob_profile_priority_four_is_accepted() -> None:
    # Given
    raw_request = (SUBMISSION / "exact_job_spec.json").read_bytes()

    # When
    request = subject.CreateJobRequest.model_validate_json(raw_request)

    # Then
    assert request.task_priority == 4


def test_captured_invalid_priority_envelope_is_explicitly_rejected() -> None:
    # Given
    outcome = models.CompletedProcess(
        returncode=0,
        stdout=(FIXTURES / "createjob-invalid-priority.json").read_bytes(),
        stderr=b"",
    )

    # When
    decision = admission_parser.classify_createjob(outcome)

    # Then
    assert decision == models.Rejected(
        error_code="InvalidParameter",
        message="invalid JobOpenAPI.TaskPriority: value must be inside range [1, 10]",
    )


def test_getjob_priority_zero_does_not_mismatch_createjob_profile_priority() -> None:
    # Given
    expectation = models.JobExpectation(job_id=JOB_ID, name=RUN_ID)
    response = json.dumps(
        {
            "ResponseMetadata": {"Action": "GetJob"},
            "Result": {
                "job_id": JOB_ID,
                "name": RUN_ID,
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
    result = admission_parser.confirm_getjob(
        models.CompletedProcess(returncode=0, stdout=response, stderr=b""),
        expectation,
    )

    # Then
    assert result == models.ConfirmedJob(job_id=JOB_ID, status="job_queuing")
