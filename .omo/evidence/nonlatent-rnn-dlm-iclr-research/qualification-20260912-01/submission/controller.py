#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic==2.11.5"]
# ///

# ─── How to run ───
# Use the preinstalled environment; do not resolve or install packages:
#   /usr/bin/python controller.py reserve AUTHORIZATION STATE
# Real submission additionally requires the root confirmation argument.
# ──────────────────

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Final

from controller_contract import (
    build_receipt as build_receipt,
    build_request as build_request,
    extract_job_id as extract_job_id,
    qz_arguments as qz_arguments,
    serialize_request as serialize_request,
    validate_receipt as validate_receipt,
    validate_request as validate_request,
)
from controller_models import (
    AdmissionRecord as AdmissionRecord,
    AdmittedJobError as AdmittedJobError,
    AttemptContext as AttemptContext,
    ControllerError as ControllerError,
    ControllerState as ControllerState,
    ProcessResult as ProcessResult,
)
from controller_runtime import (
    execute_submission as execute_submission,
    record_and_call_once as record_and_call_once,
    run_dry_run as run_dry_run,
)
from controller_storage import (
    load_authorization as load_authorization,
    load_state as load_state,
    prepare_request as prepare_request,
    publish_receipt as publish_receipt,
    request_with_digest as request_with_digest,
    reserve as reserve,
)


class CliCommand(StrEnum):
    RESERVE = "reserve"
    BUILD_REQUEST = "build-request"
    VALIDATE_REQUEST = "validate-request"
    DRY_RUN = "dry-run"
    SUBMIT = "submit"


def _command(arguments: Sequence[str]) -> CliCommand:
    if not arguments:
        raise ControllerError(code="usage_error")
    try:
        return CliCommand(arguments[0])
    except ValueError as error:
        raise ControllerError(code="usage_error") from error


def _require_arity(arguments: Sequence[str], expected: int) -> None:
    if len(arguments) != expected:
        raise ControllerError(code="usage_error")


def _reserve(arguments: Sequence[str]) -> None:
    _require_arity(arguments, 3)
    print(reserve(Path(arguments[1]), Path(arguments[2])).model_dump_json())


def _build_request(arguments: Sequence[str]) -> None:
    _require_arity(arguments, 3)
    print(prepare_request(Path(arguments[1]), Path(arguments[2])))


def _validate_request(arguments: Sequence[str]) -> None:
    _require_arity(arguments, 3)
    load_authorization(Path(arguments[1]))
    state = load_state(Path(arguments[2]))
    _, digest = request_with_digest(state)
    print(digest)


def _dry_run(arguments: Sequence[str]) -> None:
    _require_arity(arguments, 3)
    print(
        run_dry_run(Path(arguments[1]), Path(arguments[2])).model_dump_json()
    )


def _submit(arguments: Sequence[str]) -> None:
    _require_arity(arguments, 4)
    receipt = execute_submission(
        Path(arguments[1]), Path(arguments[2]), arguments[3]
    )
    print(receipt.model_dump_json(exclude_none=False))


type CliHandler = Callable[[Sequence[str]], None]
HANDLERS: Final[Mapping[CliCommand, CliHandler]] = MappingProxyType(
    {
        CliCommand.RESERVE: _reserve,
        CliCommand.BUILD_REQUEST: _build_request,
        CliCommand.VALIDATE_REQUEST: _validate_request,
        CliCommand.DRY_RUN: _dry_run,
        CliCommand.SUBMIT: _submit,
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        HANDLERS[_command(arguments)](arguments)
    except AdmittedJobError as error:
        print(
            f'{{"error":"{error.code}","admitted_job_id":"{error.job_id}"}}',
            file=sys.stderr,
        )
        return 2
    except ControllerError as error:
        print(error.code, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
