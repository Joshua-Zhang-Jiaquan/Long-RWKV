from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Final, Literal, NoReturn, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from .controller_receipt import RunId
from .contracts import HexDigest, Nonce, QualificationInputError


STAGED_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale"
)
#: The neutral step-4750 loop endpoint. The raw ``outputs_birwkv_diffusion/m4-loop-2p9b``
#: directory this default used to name was pruned, so the default pointed at nothing; this is the
#: surviving artifact and the identity the accepted qualifications pinned (sha256 ffaa464d…).
CHECKPOINT_DIR: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/m2_baseline_triangle/m4loop_endpoint_ckpt"
)
MODEL_DIR: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF"
)
DEFAULT_MANIFEST: Final = Path(__file__).with_name("runtime_manifest.json")


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise QualificationInputError(message)


class QualificationLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    expected_world_size: Literal[8] = 8
    workflow_timeout_seconds: Literal[1_650] = 1_650
    preflight_timeout_seconds: Literal[300] = 300
    runtime_timeout_seconds: Literal[1_200] = 1_200
    aggregation_timeout_seconds: Literal[90] = 90
    controller_receipt_timeout_seconds: Literal[60] = 60
    nccl_timeout_seconds: Literal[120] = 120
    kill_grace_seconds: Literal[30] = 30
    memory_fraction: Literal[0.90] = 0.90

    def validate_world_size(self, value: int) -> None:
        if value != self.expected_world_size:
            raise QualificationInputError("expected_world_size_must_be_8")

    def validate_timeout(self, value: int) -> None:
        if value > self.runtime_timeout_seconds:
            raise QualificationInputError("timeout_seconds_exceeds_1200")
        if value < 1:
            raise QualificationInputError("timeout_seconds_must_be_positive")

    def validate_nccl_timeout(self, value: int) -> None:
        if value > self.nccl_timeout_seconds:
            raise QualificationInputError("nccl_timeout_seconds_exceeds_120")
        if value < 1:
            raise QualificationInputError("nccl_timeout_seconds_must_be_positive")

    def validate_receipt_timeout(self, value: float) -> None:
        if not math.isfinite(value):
            raise QualificationInputError(
                "controller_receipt_timeout_seconds_must_be_finite"
            )
        if value > self.controller_receipt_timeout_seconds:
            raise QualificationInputError(
                "controller_receipt_timeout_seconds_exceeds_60"
            )
        if value < 0.0:
            raise QualificationInputError(
                "controller_receipt_timeout_seconds_must_be_nonnegative"
            )

    def validate_memory_fraction(self, value: float) -> None:
        if not math.isfinite(value):
            raise QualificationInputError("memory_fraction_must_be_finite")
        if value > self.memory_fraction:
            raise QualificationInputError("memory_fraction_exceeds_0_90")
        if value <= 0:
            raise QualificationInputError("memory_fraction_must_be_positive")


class ProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    output_dir: Path
    nonce: Nonce
    expected_world_size: Literal[8]
    timeout_seconds: int
    memory_fraction: float
    staged_root: Path
    checkpoint_dir: Path
    hf_model_root: Path
    manifest_path: Path
    manifest_sha256: HexDigest
    preflight_receipt: Path
    controller_receipt: Path
    run_id: RunId
    controller_receipt_timeout_seconds: float
    nccl_timeout_seconds: int


class ParseOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["READY", "INVALID_ARGUMENT"]
    detail: str
    config: ProbeConfig | None = None


def parse_probe_config(argv: Sequence[str]) -> ParseOutcome:
    parser = _ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1_200)
    parser.add_argument("--nccl-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--controller-receipt-timeout-seconds", type=float, default=60.0
    )
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--staged-root", type=Path, default=STAGED_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--controller-receipt", type=Path)
    parser.add_argument("--run-id")
    try:
        arguments = parser.parse_args(argv)
        limits = QualificationLimits()
        limits.validate_world_size(arguments.expected_world_size)
        limits.validate_timeout(arguments.timeout_seconds)
        limits.validate_nccl_timeout(arguments.nccl_timeout_seconds)
        limits.validate_receipt_timeout(arguments.controller_receipt_timeout_seconds)
        limits.validate_memory_fraction(arguments.memory_fraction)
        _require_lower_hex(arguments.nonce, length=32, name="nonce")
        if arguments.manifest_sha256 is None:
            raise QualificationInputError("manifest_sha256_is_required")
        _require_lower_hex(
            arguments.manifest_sha256, length=64, name="manifest_sha256"
        )
        preflight_receipt = arguments.preflight_receipt or (
            arguments.output_dir / "preflight.json"
        )
        controller_receipt = arguments.controller_receipt
        if controller_receipt is None:
            environment_path = os.environ.get("QUALIFICATION_JOB_RECEIPT", "").strip()
            if not environment_path:
                raise QualificationInputError("controller_receipt_is_required")
            controller_receipt = Path(environment_path)
        if not controller_receipt.is_absolute():
            raise QualificationInputError("controller_receipt_must_be_absolute")
        run_id = arguments.run_id or os.environ.get("QUALIFICATION_RUN_ID", "").strip()
        if not run_id:
            raise QualificationInputError("run_id_is_required")
        config = ProbeConfig(
            output_dir=arguments.output_dir,
            nonce=arguments.nonce,
            expected_world_size=arguments.expected_world_size,
            timeout_seconds=arguments.timeout_seconds,
            memory_fraction=arguments.memory_fraction,
            staged_root=arguments.staged_root,
            checkpoint_dir=arguments.checkpoint_dir,
            hf_model_root=arguments.model_dir,
            manifest_path=arguments.manifest,
            manifest_sha256=arguments.manifest_sha256,
            preflight_receipt=preflight_receipt,
            controller_receipt=controller_receipt,
            run_id=run_id,
            controller_receipt_timeout_seconds=(
                arguments.controller_receipt_timeout_seconds
            ),
            nccl_timeout_seconds=arguments.nccl_timeout_seconds,
        )
    except (QualificationInputError, ValidationError) as error:
        return ParseOutcome(status="INVALID_ARGUMENT", detail=str(error))
    return ParseOutcome(
        status="READY", detail="cpu_argument_validation_passed", config=config
    )


def _require_lower_hex(value: str, *, length: int, name: str) -> None:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise QualificationInputError(
            f"{name}_must_be_{length}_lower_hex_characters"
        )
