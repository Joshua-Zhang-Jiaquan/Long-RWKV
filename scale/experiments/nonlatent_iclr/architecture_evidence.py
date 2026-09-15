"""Immutable qualification-05 ingestion, without importing a model or opening weights."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from .architecture_evidence_io import EvidenceError as EvidenceError, bound_bytes as bound_bytes, read_ledger

from .architecture_evidence_models import (
    AcceptedRank, Aggregate, BoundedCounts, Launcher, Manifest, Review,
    RuntimeQualification, LifecycleQualification,
)
from .qualification.controller_receipt import ControllerReceipt
from .qualification.evidence_contracts import REQUIRED_CHECKS
from .architecture_semantics_models import SemanticsQualification

CAMPAIGN: Final = Path(".omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260913-05")
REVIEW_SHA256: Final = "7e63ee75c7c9a9f5ebd8b18a52da18b7768bc46c0f8d53c74bb64c92012f05f0"
REVIEW_LEDGER_SHA256: Final = "57abd4677a3f36c07cf61eb73798b05af6f69115fa3041268dd3388374fc0325"


def load_runtime_evidence(root: Path) -> RuntimeQualification | None:
    """Only isolated source-only roots may omit the entire campaign evidence tree."""
    campaign = root / CAMPAIGN
    canonical_root = Path(__file__).parents[3].resolve()
    evidence_tree = root / ".omo/evidence/nonlatent-rnn-dlm-iclr-research"
    if root.resolve() != canonical_root and not evidence_tree.exists():
        return None
    try:
        return _load(campaign)
    except (ValidationError, KeyError, UnicodeDecodeError) as error:
        raise EvidenceError("evidence_schema_mismatch") from error


def load_lifecycle_evidence(root: Path) -> LifecycleQualification | None:
    """Require v7 alongside historical evidence, never silently fall back on damage."""
    from .architecture_lifecycle import load_v7

    evidence_tree = root / ".omo/evidence/nonlatent-rnn-dlm-iclr-research"
    if root.resolve() != Path(__file__).parents[3].resolve() and not evidence_tree.exists():
        return None
    try:
        return load_v7(root)
    except (ValidationError, KeyError, UnicodeDecodeError) as error:
        raise EvidenceError("lifecycle_evidence_schema_mismatch") from error


def _load(campaign: Path) -> RuntimeQualification:
    review_dir = campaign / "runtime-review-superseding-01"
    reviewed = read_ledger(review_dir / "evidence-sha256.txt", REVIEW_LEDGER_SHA256)
    review = Review.model_validate_json(bound_bytes(review_dir / "verdict.json", REVIEW_SHA256))
    if set(reviewed) != {"verdict.json", "reconciliation.json", "review.md", "commands-and-outputs.md"}:
        raise EvidenceError("review_inventory_mismatch")
    binding = review.source_binding
    monitoring = campaign / "submission/monitoring-v6"
    _ = read_ledger(monitoring / "evidence-sha256.txt", binding.monitoring_ledger_sha256)
    worker = read_ledger(monitoring / "worker-output-hashes.sha256", binding.worker_output_ledger_sha256)
    expected = {f"rank-{rank}.json" for rank in range(8)} | {
        "aggregate.json", "launcher.json", "preflight.json", "runtime-manifest.json", "runtime_environment.json",
    }
    if set(worker) != expected:
        raise EvidenceError("worker_inventory_mismatch")
    _ = bound_bytes(campaign / "submission/exact_job_spec.json", binding.request_sha256)
    launcher = Launcher.model_validate_json(worker["launcher.json"])
    receipt = ControllerReceipt.model_validate_json(
        bound_bytes(Path(launcher.controller_receipt), binding.controller_receipt_sha256)
    )
    if (
        receipt.job_id != binding.job_id or receipt.run_id != binding.run_id
        or receipt.nonce != binding.nonce or receipt.submitted_request_sha256 != binding.request_sha256
        or receipt.source_manifest_sha256 != binding.manifest_sha256
        or launcher.run_id != binding.run_id or launcher.nonce != binding.nonce
        or launcher.manifest_sha256 != binding.manifest_sha256
        or sha256(worker["runtime-manifest.json"]).hexdigest() != binding.manifest_sha256
    ):
        raise EvidenceError("runtime_identity_mismatch")
    manifest = Manifest.model_validate_json(worker["runtime-manifest.json"])
    checkpoints = tuple(item for item in manifest.files if item.role == "checkpoint_model")
    if len(checkpoints) != 1:
        raise EvidenceError("checkpoint_manifest_cardinality")
    checkpoint = checkpoints[0]
    ranks = tuple(AcceptedRank.model_validate_json(worker[f"rank-{rank}.json"]) for rank in range(8))
    aggregate = Aggregate.model_validate_json(worker["aggregate.json"])
    if aggregate.rank_results != ranks or tuple(rank.rank for rank in ranks) != tuple(range(8)):
        raise EvidenceError("rank_aggregate_identity_mismatch")
    expected_controller = receipt.to_binding_evidence()
    for rank in ranks:
        evidence = rank.evidence
        if (
            rank.rank != evidence.rank or rank.nonce != binding.nonce
            or rank.checks != REQUIRED_CHECKS or evidence.controller_binding != expected_controller
            or evidence.checkpoint_sha256 != checkpoint.sha256
            or evidence.checkpoint_size_bytes != checkpoint.size_bytes
        ):
            raise EvidenceError("rank_run_checkpoint_mismatch")
        counts = BoundedCounts(
            ranks=tuple(item.rank for item in ranks), world_size=evidence.world_size,
            checkpoint_step=evidence.checkpoint_step, checkpoint_state_tensors=evidence.checkpoint_state_tensors,
            parameter_numel=evidence.parameters.total_numel, masked_logits_shape=evidence.masked_logits_shape,
            masked_logits_all_finite=evidence.masked_logits_all_finite,
            standalone_fla_cache_max_abs_delta=evidence.fla_causal_cache.max_abs_delta,
        )
        if counts != review.bounded_evidence_retained:
            raise EvidenceError("review_rank_counts_mismatch")
    first = ranks[0].evidence
    return RuntimeQualification(
        ranks=tuple(rank.rank for rank in ranks), world_size=first.world_size,
        checkpoint_step=first.checkpoint_step, checkpoint_state_tensors=first.checkpoint_state_tensors,
        parameter_numel=first.parameters.total_numel, masked_logits_shape=first.masked_logits_shape,
        masked_logits_all_finite=first.masked_logits_all_finite,
        standalone_fla_cache_max_abs_delta=first.fla_causal_cache.max_abs_delta,
        review_sha256=REVIEW_SHA256, source_binding=binding, checkpoint=checkpoint,
        controller_binding=expected_controller, full_model_prefix_cache=first.fullmodel_cachedprefix,
        loop_prefix_cache=first.loop_cachedprefix,
    )


def load_semantics_evidence(root: Path) -> SemanticsQualification | None:
    from .architecture_semantics import load_v8

    evidence_tree = root / '.omo/evidence/nonlatent-rnn-dlm-iclr-research'
    if root.resolve() != Path(__file__).parents[3].resolve() and not evidence_tree.exists():
        return None
    try:
        return load_v8(root)
    except (ValidationError, KeyError, UnicodeDecodeError) as error:
        raise EvidenceError('semantics_evidence_schema_mismatch') from error
