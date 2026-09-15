from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

import controller_models as models
import controller_runtime as subject


JOB_ID = "job-01234567-89ab-4cde-8fab-0123456789ab"


class _AssertingRunner:
    def __init__(self, attempt_path: Path) -> None:
        self.attempt_path = attempt_path
        self.call_count = 0

    def __call__(
        self, arguments: tuple[str, ...], timeout_seconds: int
    ) -> models.CompletedProcess:
        assert self.attempt_path.is_file()
        assert timeout_seconds == 120
        self.call_count += 1
        return models.CompletedProcess(
            returncode=0,
            stdout=f'{{"Result":{{"job_id":"{JOB_ID}"}}}}'.encode(),
            stderr=b"",
        )


def _attempt_context(tmp_path: Path, raw: bytes) -> models.AttemptContext:
    return models.AttemptContext(
        attempt_path=tmp_path / "request-attempt.json",
        capture=models.CaptureTarget(directory=tmp_path, label="createjob"),
        campaign_authorization_id=models.CAMPAIGN_AUTHORIZATION_ID,
        permit_id="permit-nonlatent-gpu-campaign-20260912-00000000-0000-4000-8000-000000000000",
        grant_instance_id="grant-qualification-20260912-02-00000000-0000-4000-8000-000000000000",
        run_id="qualification-20260912-01-00000000-0000-4000-8000-000000000000",
        submitted_request_sha256=sha256(raw).hexdigest(),
    )


def test_attempt_precedes_single_call_and_response_capture(tmp_path: Path) -> None:
    # Given
    raw = b'{"name":"fixture"}'
    context = _attempt_context(tmp_path, raw)
    runner = _AssertingRunner(context.attempt_path)

    # When
    result = subject.record_capture_and_classify_once(context, raw, runner)

    # Then
    assert isinstance(result.decision, models.Admitted)
    assert result.decision.job_id == JOB_ID
    assert runner.call_count == 1
    assert (tmp_path / "createjob.response.json").is_file()


def test_existing_attempt_blocks_a_second_real_call(tmp_path: Path) -> None:
    # Given
    raw = b'{"name":"fixture"}'
    context = _attempt_context(tmp_path, raw)
    context.attempt_path.write_text("already attempted")
    runner = _AssertingRunner(context.attempt_path)

    # When / Then
    with pytest.raises(models.ControllerError, match="submission_already_attempted"):
        subject.record_capture_and_classify_once(context, raw, runner)
    assert runner.call_count == 0
