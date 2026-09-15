#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic==2.11.5"]
# ///

# ─── How to run ───
# Use the preinstalled environment; do not resolve or install packages:
#   /usr/bin/python controller.py --help
# Live submission additionally requires root-approved permit gates and confirmation.
# ──────────────────

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Final

from controller_models import AdmittedJobError, ControllerError
from controller_runtime import execute_submission
from controller_storage import load_submission_context
from dry_run_runtime import run_dry_run
from request_contract import validate_request
from response_capture import run_subprocess


HELP: Final = """usage:
  controller.py --help
  controller.py validate CAMPAIGN PERMIT STATE
  controller.py dry-run CAMPAIGN PERMIT STATE
  controller.py submit CAMPAIGN PERMIT STATE ROOT_PASS_CONFIRMED

submit performs at most one real CreateJob for this reservation, captures the
response before parsing, confirms any returned ID with one GetJob, and never
automatically retries an ambiguous outcome.
"""


class CliCommand(StrEnum):
    VALIDATE = "validate"
    DRY_RUN = "dry-run"
    SUBMIT = "submit"


def _require_arity(arguments: Sequence[str], expected: int) -> None:
    if len(arguments) != expected:
        raise ControllerError(code="usage_error")


def _context(arguments: Sequence[str]):
    return load_submission_context(
        Path(arguments[1]), Path(arguments[2]), Path(arguments[3])
    )


def _validate(arguments: Sequence[str]) -> None:
    _require_arity(arguments, 4)
    context = _context(arguments)
    validate_request(context.request_bytes, context)
    print(sha256(context.request_bytes).hexdigest())


def _dry_run(arguments: Sequence[str]) -> None:
    _require_arity(arguments, 4)
    print(run_dry_run(_context(arguments), run_subprocess).model_dump_json())


def _submit(arguments: Sequence[str]) -> None:
    _require_arity(arguments, 5)
    receipt = execute_submission(
        _context(arguments), arguments[4], run_subprocess
    )
    print(receipt.model_dump_json(exclude_none=False))


type CliHandler = Callable[[Sequence[str]], None]
HANDLERS: Final[Mapping[CliCommand, CliHandler]] = MappingProxyType(
    {
        CliCommand.VALIDATE: _validate,
        CliCommand.DRY_RUN: _dry_run,
        CliCommand.SUBMIT: _submit,
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--help"]:
        print(HELP, end="")
        return 0
    try:
        if not arguments:
            raise ControllerError(code="usage_error")
        try:
            command = CliCommand(arguments[0])
        except ValueError as error:
            raise ControllerError(code="usage_error") from error
        HANDLERS[command](arguments)
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
