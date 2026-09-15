from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Final

from pydantic import ConfigDict, JsonValue, RootModel, ValidationError

from controller_models import (
    CaptureTarget,
    CapturedInvocation,
    CompletedProcess,
    ControllerError,
    Invocation,
    InvocationFailedProcess,
    ProcessOutcome,
    ProcessRunner,
    ResponseCapture,
    TimedOutProcess,
    UtcTimestamp,
)


SENSITIVE_KEY: Final = re.compile(
    r"(?i)(token|password|passwd|secret|credential|authorization|auth|"
    r"api[_-]?key|access[_-]?key|private[_-]?key|cookie)"
)
BEARER_VALUE: Final = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
TEXT_SECRET: Final = re.compile(
    r"(?i)\b(token|password|passwd|secret|credential|authorization|auth|"
    r"api[_-]?key|access[_-]?key|private[_-]?key|cookie)"
    r"(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


class JsonPayload(RootModel[JsonValue]):
    model_config = ConfigDict(frozen=True, strict=True)


def now() -> UtcTimestamp:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _redact_text(text: str) -> str:
    bearer_safe = BEARER_VALUE.sub("Bearer [REDACTED]", text)
    return TEXT_SECRET.sub(r"\1\2[REDACTED]", bearer_safe)


def _redact_json(value: JsonValue) -> JsonValue:
    match value:
        case dict() as mapping:
            return {
                key: (
                    "[REDACTED]"
                    if SENSITIVE_KEY.search(key)
                    else _redact_json(child)
                )
                for key, child in mapping.items()
            }
        case list() as items:
            return [_redact_json(item) for item in items]
        case str() as text:
            return _redact_text(text)
        case bool() | int() | float() | None:
            return value


def redact_payload(raw: bytes) -> bytes:
    if not raw:
        return raw
    try:
        parsed = JsonPayload.model_validate_json(raw).root
    except ValidationError:
        return _redact_text(raw.decode("utf-8", errors="replace")).encode("utf-8")
    return json.dumps(
        _redact_json(parsed), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _write_private(path: os.PathLike[str], payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise ControllerError(code="response_capture_exists") from error


def capture_paths(target: CaptureTarget) -> tuple[Path, Path, Path]:
    return (
        target.directory / f"{target.label}.stdout.redacted",
        target.directory / f"{target.label}.stderr.redacted",
        target.directory / f"{target.label}.response.json",
    )


def require_capture_absent(target: CaptureTarget) -> None:
    if any(Path(path).exists() for path in capture_paths(target)):
        raise ControllerError(code="response_capture_exists")


def persist_outcome(target: CaptureTarget, outcome: ProcessOutcome) -> ResponseCapture:
    directory = target.directory
    label = target.label
    stdout_path = directory / f"{label}.stdout.redacted"
    stderr_path = directory / f"{label}.stderr.redacted"
    metadata_path = directory / f"{label}.response.json"
    require_capture_absent(target)
    match outcome:
        case CompletedProcess(returncode=returncode, stdout=stdout, stderr=stderr):
            outcome_name = "completed"
            timed_out = False
            timeout_seconds = 0
            invocation_error_code = None
        case TimedOutProcess(
            timeout_seconds=timeout_seconds, stdout=stdout, stderr=stderr
        ):
            outcome_name = "timed_out"
            returncode = None
            timed_out = True
            invocation_error_code = None
        case InvocationFailedProcess(
            error_code=invocation_error_code, stdout=stdout, stderr=stderr
        ):
            outcome_name = "invocation_failed"
            returncode = None
            timed_out = False
            timeout_seconds = 0
    redacted_stdout = redact_payload(stdout)
    redacted_stderr = redact_payload(stderr)
    record = ResponseCapture(
        label=label,
        outcome=outcome_name,
        returncode=returncode,
        timed_out=timed_out,
        timeout_seconds=timeout_seconds,
        invocation_error_code=invocation_error_code,
        stdout_sha256=sha256(stdout).hexdigest(),
        stdout_bytes=len(stdout),
        stdout_redacted_sha256=sha256(redacted_stdout).hexdigest(),
        stdout_redacted_bytes=len(redacted_stdout),
        stdout_path=stdout_path,
        stderr_sha256=sha256(stderr).hexdigest(),
        stderr_bytes=len(stderr),
        stderr_redacted_sha256=sha256(redacted_stderr).hexdigest(),
        stderr_redacted_bytes=len(redacted_stderr),
        stderr_path=stderr_path,
        metadata_path=metadata_path,
        recorded_at_utc=now(),
    )
    _write_private(stdout_path, redacted_stdout)
    _write_private(stderr_path, redacted_stderr)
    _write_private(metadata_path, record.model_dump_json().encode("utf-8"))
    return record


def _timeout_bytes(value: str | bytes | None) -> bytes:
    match value:
        case bytes() as raw:
            return raw
        case str() as text:
            return text.encode("utf-8")
        case None:
            return b""


def invoke_and_capture(
    invocation: Invocation, runner: ProcessRunner
) -> CapturedInvocation:
    try:
        outcome: ProcessOutcome = runner(
            invocation.arguments, invocation.timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        outcome = TimedOutProcess(
            timeout_seconds=invocation.timeout_seconds,
            stdout=_timeout_bytes(error.stdout),
            stderr=_timeout_bytes(error.stderr),
        )
    except OSError as error:
        outcome = InvocationFailedProcess(
            error_code=type(error).__name__, stderr=str(error).encode("utf-8")
        )
    record = persist_outcome(invocation.capture, outcome)
    return CapturedInvocation(outcome=outcome, record=record)


def run_subprocess(
    arguments: tuple[str, ...], timeout_seconds: int
) -> CompletedProcess:
    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )
    return CompletedProcess(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
