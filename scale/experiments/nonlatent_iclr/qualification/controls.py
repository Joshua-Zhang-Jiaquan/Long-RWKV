"""CPU-safe aggregation boundary for the H100 payload."""

from __future__ import annotations

from pathlib import Path
from typing import assert_never

from pydantic import ValidationError

from .contracts import (
    AggregateResult,
    FailureEvidence,
    QualificationInputError,
    RankResult,
    REQUIRED_CHECKS,
    SuccessEvidence,
)
from .manifest import verify_runtime_manifest
from .runtime_config import (
    ParseOutcome,
    ProbeConfig,
    QualificationLimits,
    parse_probe_config,
)


def aggregate_results(
    output_dir: Path, *, nonce: str, manifest_sha256: str | None = None
) -> AggregateResult:
    """Require exactly eight fresh typed rank records before a partial verdict."""
    expected_ranks = tuple(range(QualificationLimits().expected_world_size))
    allowed_names = {f"rank-{rank}.json" for rank in expected_ranks}
    extra_names = _extra_rank_names(output_dir, allowed_names)
    if extra_names:
        return AggregateResult(
            status="FAILED",
            detail="extra_rank_results:" + ",".join(extra_names),
            rank_results=(),
        )
    missing = tuple(
        rank for rank in expected_ranks if not _result_path(output_dir, rank).is_file()
    )
    if missing:
        return AggregateResult(
            status="FAILED",
            detail="missing_rank_results:" + ",".join(str(rank) for rank in missing),
            rank_results=(),
        )
    parsed: list[RankResult] = []
    stable_identity: tuple[str, ...] | None = None
    stable_controller_binding: str | None = None
    for rank in expected_ranks:
        parsed_result = _parse_rank_result(_result_path(output_dir, rank), rank)
        if isinstance(parsed_result, str):
            return AggregateResult(
                status="FAILED", detail=parsed_result, rank_results=tuple(parsed)
            )
        if parsed_result.rank != rank:
            return AggregateResult(
                status="FAILED",
                detail=f"rank_identity_mismatch:{rank}",
                rank_results=tuple(parsed),
            )
        if parsed_result.nonce != nonce:
            return AggregateResult(
                status="FAILED",
                detail=f"stale_or_foreign_rank_result:{rank}",
                rank_results=tuple(parsed),
            )
        match parsed_result.evidence:
            case SuccessEvidence() as evidence:
                if evidence.rank != rank:
                    return AggregateResult(
                        status="FAILED",
                        detail=f"evidence_rank_identity_mismatch:{rank}",
                        rank_results=tuple(parsed),
                    )
                if (
                    manifest_sha256 is not None
                    and evidence.source_manifest_sha256 != manifest_sha256
                ):
                    return AggregateResult(
                        status="FAILED",
                        detail=f"source_manifest_identity_mismatch:{rank}",
                        rank_results=tuple(parsed),
                    )
                controller_binding = evidence.controller_binding.model_dump_json()
                if stable_controller_binding is None:
                    stable_controller_binding = controller_binding
                elif controller_binding != stable_controller_binding:
                    return AggregateResult(
                        status="FAILED",
                        detail=f"cross_rank_controller_binding_mismatch:{rank}",
                        rank_results=tuple(parsed),
                    )
                identity = _stable_success_identity(evidence)
                if stable_identity is None:
                    stable_identity = identity
                elif identity != stable_identity:
                    return AggregateResult(
                        status="FAILED",
                        detail=f"cross_rank_identity_mismatch:{rank}",
                        rank_results=tuple(parsed),
                    )
            case FailureEvidence():
                pass
            case unreachable:
                assert_never(unreachable)
        parsed.append(parsed_result)
    for result in parsed:
        if result.status == "FAILED":
            return AggregateResult(
                status="FAILED",
                detail=f"rank_runtime_failure:{result.rank}:{result.detail}",
                rank_results=tuple(parsed),
            )
    return AggregateResult(
        status="PARTIAL_QUALIFICATION",
        detail="corevalid/fullmodelcacheunqualified",
        rank_results=tuple(parsed),
    )


def _result_path(output_dir: Path, rank: int) -> Path:
    return output_dir / f"rank-{rank}.json"


def _extra_rank_names(output_dir: Path, allowed_names: set[str]) -> tuple[str, ...]:
    if not output_dir.is_dir():
        return ()
    candidates = (
        path.name
        for path in output_dir.iterdir()
        if path.name.startswith("rank-") or path.name.startswith(".rank-")
    )
    return tuple(sorted(name for name in candidates if name not in allowed_names))


def _parse_rank_result(path: Path, rank: int) -> RankResult | str:
    if path.is_symlink():
        return f"malformed_rank_result:{rank}"
    try:
        return RankResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError):
        return f"malformed_rank_result:{rank}"


def _stable_success_identity(evidence: SuccessEvidence) -> tuple[str, ...]:
    inventory = tuple(gpu.model_dump_json() for gpu in evidence.gpu_inventory)
    return (
        evidence.python_version,
        evidence.torch_version,
        evidence.fla_version,
        evidence.fla_core_version,
        evidence.transformers_version,
        evidence.safetensors_version,
        evidence.triton_version,
        evidence.source_manifest_sha256,
        str(evidence.verified_source_files),
        evidence.checkpoint_sha256,
        evidence.geometry.model_dump_json(),
        evidence.parameters.model_dump_json(),
        *inventory,
    )
