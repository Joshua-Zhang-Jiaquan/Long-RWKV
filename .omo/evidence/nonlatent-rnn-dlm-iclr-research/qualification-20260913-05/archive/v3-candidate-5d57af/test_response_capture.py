from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import stat
import subprocess

import pytest

import controller_models as models
import response_capture as subject


JOB_ID = "job-01234567-89ab-4cde-8fab-0123456789ab"


class _TimeoutRunner:
    def __call__(
        self, arguments: tuple[str, ...], timeout_seconds: int
    ) -> models.CompletedProcess:
        raise subprocess.TimeoutExpired(
            cmd=arguments,
            timeout=timeout_seconds,
            output=b'{"Code":"Pending",',
            stderr=b"Authorization: Bearer timeout-secret",
        )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_capture_persists_hash_length_content_and_private_mode(tmp_path: Path) -> None:
    # Given
    target = models.CaptureTarget(directory=tmp_path, label="complete")
    stdout = f'{{"job_id":"{JOB_ID}","Code":"Accepted"}}'.encode()
    outcome = models.CompletedProcess(returncode=0, stdout=stdout, stderr=b"")

    # When
    record = subject.persist_outcome(target, outcome)

    # Then
    assert record.stdout_sha256 == sha256(stdout).hexdigest()
    assert record.stdout_bytes == len(stdout)
    assert Path(record.stdout_path).read_bytes() == stdout
    assert _mode(Path(record.stdout_path)) == 0o600
    assert _mode(Path(record.metadata_path)) == 0o600


def test_json_redaction_removes_secrets_but_preserves_business_fields(
    tmp_path: Path,
) -> None:
    # Given
    raw = (
        f'{{"token":"token-secret","password":"password-secret",'
        f'"Authorization":"Bearer bearer-secret","Code":"QuotaExceeded",'
        f'"Message":"quota full","job_id":"{JOB_ID}"}}'
    ).encode()

    # When
    redacted = subject.redact_payload(raw)

    # Then
    assert b"token-secret" not in redacted
    assert b"password-secret" not in redacted
    assert b"bearer-secret" not in redacted
    assert b"[REDACTED]" in redacted
    assert b"QuotaExceeded" in redacted
    assert JOB_ID.encode() in redacted


def test_text_redaction_removes_bearer_and_assignment_secrets() -> None:
    # Given
    raw = b"Code=Denied token=abc123 Authorization: Bearer xyz789 request=kept"

    # When
    redacted = subject.redact_payload(raw)

    # Then
    assert b"abc123" not in redacted
    assert b"xyz789" not in redacted
    assert b"Code=Denied" in redacted
    assert b"request=kept" in redacted


def test_empty_completed_response_is_still_captured(tmp_path: Path) -> None:
    # Given
    target = models.CaptureTarget(directory=tmp_path, label="empty")
    outcome = models.CompletedProcess(returncode=0, stdout=b"", stderr=b"")

    # When
    record = subject.persist_outcome(target, outcome)

    # Then
    assert record.stdout_bytes == 0
    assert Path(record.stdout_path).is_file()
    assert Path(record.stderr_path).is_file()


def test_timeout_partial_output_is_captured_before_return(tmp_path: Path) -> None:
    # Given
    invocation = models.Invocation(
        arguments=("qz", "train", "CreateJob"),
        timeout_seconds=120,
        capture=models.CaptureTarget(directory=tmp_path, label="timeout"),
    )

    # When
    captured = subject.invoke_and_capture(invocation, _TimeoutRunner())

    # Then
    assert isinstance(captured.outcome, models.TimedOutProcess)
    assert captured.record.timed_out is True
    assert captured.record.stdout_bytes > 0
    assert b"Pending" in Path(captured.record.stdout_path).read_bytes()
    assert b"timeout-secret" not in Path(captured.record.stderr_path).read_bytes()


def test_capture_refuses_to_overwrite_prior_outcome(tmp_path: Path) -> None:
    # Given
    target = models.CaptureTarget(directory=tmp_path, label="once")
    outcome = models.CompletedProcess(returncode=0, stdout=b"{}", stderr=b"")
    subject.persist_outcome(target, outcome)

    # When / Then
    with pytest.raises(models.ControllerError, match="response_capture_exists"):
        subject.persist_outcome(target, outcome)
