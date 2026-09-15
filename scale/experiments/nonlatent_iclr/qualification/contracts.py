"""Strict terminal evidence contracts for the bounded qualification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_contracts import (
    CHECKPOINT_SHA256,
    REQUIRED_CHECKS,
    CausalCacheEvidence,
    CheckName,
    FailureCheck,
    FailureEvidence,
    GPURecord,
    HexDigest,
    MemoryEvidence,
    ModelGeometry,
    Nonce,
    ParameterEvidence,
    QualificationContractError,
    QualificationInputError,
    QualificationRuntimeError,
    SuccessEvidence,
)


class RankResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    rank: int = Field(ge=0, le=7)
    nonce: Nonce
    status: Literal["PASSED", "FAILED"]
    detail: str = Field(min_length=1)
    checks: tuple[CheckName, ...]
    evidence: SuccessEvidence | FailureEvidence = Field(discriminator="kind")

    @model_validator(mode="after")
    def enforce_status_variant(self) -> Self:
        match self.evidence:
            case SuccessEvidence():
                if self.status != "PASSED":
                    raise QualificationContractError("status_evidence_variant_mismatch")
                if self.checks != REQUIRED_CHECKS:
                    raise QualificationContractError("passed_checks_not_exact")
                if self.detail != "all_bounded_checks_passed":
                    raise QualificationContractError("passed_detail_not_exact")
                if self.nonce != self.evidence.controller_binding.nonce:
                    raise QualificationContractError("controller_nonce_identity_mismatch")
            case FailureEvidence():
                if self.status != "FAILED":
                    raise QualificationContractError("status_evidence_variant_mismatch")
                if self.checks != self.evidence.completed_checks:
                    raise QualificationContractError("failed_checks_not_completed_checks")
            case unreachable:
                assert_never(unreachable)
        return self

    @classmethod
    def passing(
        cls, *, rank: int, nonce: str, evidence: SuccessEvidence
    ) -> RankResult:
        return cls(
            rank=rank,
            nonce=nonce,
            status="PASSED",
            detail="all_bounded_checks_passed",
            checks=REQUIRED_CHECKS,
            evidence=evidence,
        )

    @classmethod
    def failed(
        cls,
        *,
        rank: int,
        nonce: str,
        detail: str,
        failed_check: FailureCheck = "runtime_boundary",
        completed_checks: tuple[CheckName, ...] = (),
        exception_type: str = "QualificationFailure",
        traceback_text: str | None = None,
        source_manifest_sha256: str = "0" * 64,
    ) -> RankResult:
        return cls(
            rank=rank,
            nonce=nonce,
            status="FAILED",
            detail=detail,
            checks=completed_checks,
            evidence=FailureEvidence(
                kind="failure",
                failed_check=failed_check,
                completed_checks=completed_checks,
                exception_type=exception_type,
                traceback=traceback_text or detail,
                source_manifest_sha256=source_manifest_sha256,
            ),
        )

    def write_once(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"rank-{self.rank}.json"
        temporary = output_dir / f".rank-{self.rank}.{self.nonce}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(self.model_dump_json(indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


class AggregateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["PARTIAL_QUALIFICATION", "FAILED"]
    detail: str
    rank_results: tuple[RankResult, ...]
