"""GPU-job entrypoint with CPU-safe parsing and terminal rank publication."""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Sequence, assert_never

from .contracts import QualificationInputError, RankResult
from .controls import ProbeConfig, parse_probe_config
from .model_checks import MODEL_LOOP_RANGE, MODEL_LOOP_REPS
from .runtime_checks import RuntimeCheckFailure, run_runtime_checks
from .runtime_identity import is_authorized_h100


def main(argv: Sequence[str] | None = None) -> int:
    """Run one rank and publish exactly one nonce-bound terminal record."""
    effective_argv = sys.argv[1:] if argv is None else argv
    if "--help" in effective_argv:
        print(
            "usage: probe.py --output-dir PATH --nonce TOKEN "
            "--run-id ID --controller-receipt PATH "
            "--manifest PATH --manifest-sha256 DIGEST"
        )
        return 0
    parsed = parse_probe_config(effective_argv)
    match parsed.status:
        case "INVALID_ARGUMENT":
            print(
                json.dumps(
                    {"detail": parsed.detail, "status": parsed.status}, sort_keys=True
                )
            )
            return 2
        case "READY":
            config = parsed.config
        case unreachable:
            assert_never(unreachable)
    if config is None:
        print(
            json.dumps(
                {"detail": "missing_ready_config", "status": "INVALID_ARGUMENT"},
                sort_keys=True,
            )
        )
        return 2
    rank = _rank_from_environment()
    try:
        result = _run(config, rank)
    except Exception as error:  # noqa: BROAD_EXCEPT_OK
        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr, end="")
        match error:
            case RuntimeCheckFailure():
                failed_check = error.failed_check
                completed_checks = error.completed_checks
                detail = error.detail
            case _:
                failed_check = "runtime_boundary"
                completed_checks = ()
                detail = f"runtime_boundary:{type(error).__name__}:{error}"
        result = RankResult.failed(
            rank=rank,
            nonce=config.nonce,
            detail=detail,
            failed_check=failed_check,
            completed_checks=completed_checks,
            exception_type=type(error).__name__,
            traceback_text=traceback_text,
            source_manifest_sha256=config.manifest_sha256,
        )
    result.write_once(config.output_dir)
    print(result.model_dump_json())
    return 0 if result.status == "PASSED" else 1


def _rank_from_environment() -> int:
    raw_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", ""))
    try:
        rank = int(raw_rank)
    except ValueError as error:
        raise QualificationInputError("rank_environment_not_integer") from error
    if rank not in range(8):
        raise QualificationInputError("rank_environment_out_of_range")
    return rank


def _run(config: ProbeConfig, rank: int) -> RankResult:
    return run_runtime_checks(config, rank)


if __name__ == "__main__":
    raise SystemExit(main())
