"""CPU-safe command boundary for payload validation and result aggregation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Final, Sequence, assert_never

from .calibration_arms import DERIVED_ARMS, RANK_COUNT
from .calibration_contracts import aggregate, load_sidecar
from .contracts import QualificationInputError
from .controls import aggregate_results, parse_probe_config, verify_runtime_manifest
from .manifest import PreflightReceipt


_REQUIRED_EXECUTABLES: Final = (Path("/usr/bin/python"), Path("/usr/local/bin/torchrun"))
CALIBRATION_AGGREGATE_NAME: Final = "aggregate-calibration.json"
CALIBRATION_SIDECAR_PREFIX: Final = "calibration-rank-"


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CPU-only payload boundary without importing torch or FLA."""
    parser = argparse.ArgumentParser(prog="nonlatent-qualification")
    parser.add_argument("command", choices=("validate", "aggregate", "aggregate-calibration"))
    command, remaining = parser.parse_known_args(argv)
    parsed = parse_probe_config(remaining)
    match parsed.status:
        case "INVALID_ARGUMENT":
            print(json.dumps({"detail": parsed.detail, "status": parsed.status}, sort_keys=True))
            return 2
        case "READY":
            config = parsed.config
            if config is None:
                print(json.dumps({"detail": "missing_ready_config", "status": "INVALID_ARGUMENT"}, sort_keys=True))
                return 2
            match command.command:
                case "validate":
                    try:
                        _verify_required_executables()
                        verification = verify_runtime_manifest(
                            config.manifest_path,
                            expected_sha256=config.manifest_sha256,
                        )
                    except QualificationInputError as error:
                        print(
                            json.dumps(
                                {
                                    "detail": f"source_identity:{error}",
                                    "status": "FAILED",
                                },
                                sort_keys=True,
                            )
                        )
                        return 1
                    receipt = PreflightReceipt.from_verification(
                        nonce=config.nonce, verification=verification
                    )
                    print(receipt.model_dump_json())
                    return 0
                case "aggregate":
                    result = aggregate_results(
                        config.output_dir,
                        nonce=config.nonce,
                        manifest_sha256=config.manifest_sha256,
                    )
                    print(result.model_dump_json())
                    return 0 if result.status == "PARTIAL_QUALIFICATION" else 1
                case "aggregate-calibration":
                    return _aggregate_calibration(config.output_dir, nonce=config.nonce)
                case unreachable:
                    assert_never(unreachable)
        case unreachable:
            assert_never(unreachable)


def _aggregate_calibration(output_dir: Path, *, nonce: str) -> int:
    """Summarise the eight per-arm calibration records, grouped by arm.

    A partial job is published as PARTIAL rather than failed: the failures are the result, and
    hiding them would misrepresent how much of the matrix was measured. A job with no measurable
    arm at all is a failure.

    Every refusal prints a reason. An uncaught exception here would leave stdout empty, and the
    launcher publishes stdout verbatim, so a reviewer would receive an empty aggregate.json with
    nothing to explain it.
    """
    try:
        return _aggregate_calibration_body(output_dir, nonce=nonce)
    except Exception as error:  # noqa: BROAD_EXCEPT_OK - a refusal must still carry a reason
        print(
            json.dumps(
                {"detail": f"calibration_aggregation_failed:{type(error).__name__}:{error}"[:500], "status": "FAILED"},
                sort_keys=True,
            )
        )
        return 1


def _aggregate_calibration_body(output_dir: Path, *, nonce: str) -> int:
    expected_names = {f"{CALIBRATION_SIDECAR_PREFIX}{rank}.json" for rank in range(RANK_COUNT)}
    sidecars = []
    for rank in range(RANK_COUNT):
        path = output_dir / f"{CALIBRATION_SIDECAR_PREFIX}{rank}.json"
        if not path.is_file():
            print(json.dumps({"detail": f"missing_calibration_rank:{rank}", "status": "FAILED"}, sort_keys=True))
            return 1
        sidecar = load_sidecar(path)
        if sidecar.nonce != nonce:
            print(json.dumps({"detail": f"calibration_nonce_mismatch:rank-{rank}", "status": "FAILED"}, sort_keys=True))
            return 1
        sidecars.append(sidecar)
    unexpected = sorted(
        path.name for path in output_dir.glob(f"{CALIBRATION_SIDECAR_PREFIX}*.json")
        if path.name not in expected_names
    )
    if unexpected:
        print(json.dumps({"detail": f"unexpected_calibration_records:{unexpected}", "status": "FAILED"}, sort_keys=True))
        return 1
    result = aggregate(tuple(sidecars), derived_arms=dict(DERIVED_ARMS))
    target = output_dir / CALIBRATION_AGGREGATE_NAME
    payload = (result.model_dump_json(indent=2) + "\n").encode("utf-8")
    if target.is_file():
        if target.read_bytes() != payload:
            print(json.dumps({"detail": "calibration_aggregate_already_exists", "status": "FAILED"}, sort_keys=True))
            return 1
    else:
        _ = target.write_bytes(payload)
    print(result.model_dump_json())
    return 0 if any(arm.contributing_ranks > 0 for arm in result.arms) else 1


def _verify_required_executables() -> None:
    for executable in _REQUIRED_EXECUTABLES:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise QualificationInputError(
                f"required_executable_unavailable:{executable}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
