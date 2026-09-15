"""Hash-bound v7 reconciliation; reads records, never weights or runtime models."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

from .architecture_evidence_io import EvidenceError, bound_bytes, read_ledger
from .architecture_evidence_models import (
    AcceptedRank, Aggregate, Launcher, LifecyclePreflight, LifecycleQualification, LifecycleReview, Manifest, Record,
)
from .qualification.controller_receipt import ControllerReceipt
from .qualification.evidence_contracts import REQUIRED_CHECKS
from .qualification.lifecycle_sidecar import LifecycleSidecar

CAMPAIGN: Final = Path('.omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260914-07-live')
REVIEW_SHA256: Final = '64a1a457191561e0576afa80aa84caff1b80210585cb74e24533dcc427325bd2'
LEDGER_SHA256: Final = '19460d5864fc49dea22295dbfb4d0ffe5895394e10beeaf1d572de2fc7fad8ab'
WORKER_ROOT: Final = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification')


class LifecycleRecords(Record):
    review: LifecycleReview
    receipt: ControllerReceipt
    launcher: Launcher
    manifest: Manifest
    preflight: LifecyclePreflight
    aggregate: Aggregate
    ranks: tuple[AcceptedRank, ...]
    sidecars: tuple[LifecycleSidecar, ...]


def read_lifecycle_records(root: Path) -> LifecycleRecords:
    """Parse only exact-hash review, worker, request and receipt bytes."""
    campaign = root / CAMPAIGN
    directory = campaign / 'runtime-review'
    reviewed = read_ledger(directory / 'evidence-sha256.txt', LEDGER_SHA256)
    if set(reviewed) != {
        'commands-and-results.md', 'input-sha256.txt', 'review.md', 'scheduler-getjob.json',
        'scheduler-getjob.stderr.txt', 'verdict.json', 'verify_lifecycle_v7.py',
    }:
        raise EvidenceError('lifecycle_review_inventory_mismatch')
    review = LifecycleReview.model_validate_json(bound_bytes(directory / 'verdict.json', REVIEW_SHA256))
    binding = review.source_binding
    _ = bound_bytes(campaign / 'createjob-spec.json', binding.request_sha256)
    worker: dict[str, bytes] = {}
    for line in bound_bytes(campaign / 'worker-artifacts.sha256', binding.worker_ledger_sha256).decode().splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise EvidenceError('lifecycle_worker_ledger_schema')
        digest, name = fields
        if name in worker or Path(name).name != name:
            raise EvidenceError('lifecycle_worker_ledger_path')
        worker[name] = bound_bytes(WORKER_ROOT / binding.run_id / name, digest)
    expected = {f'{prefix}-{rank}.json' for prefix in ('rank', 'lifecycle') for rank in range(8)}
    expected |= {'launcher.json', 'aggregate.json', 'preflight.json', 'runtime_environment.json', 'runtime-manifest.json'}
    if set(worker) != expected or sha256(worker['runtime-manifest.json']).hexdigest() != binding.manifest_sha256:
        raise EvidenceError('lifecycle_worker_inventory_or_manifest_mismatch')
    launcher = Launcher.model_validate_json(worker['launcher.json'])
    receipt_path = WORKER_ROOT / 'controller-receipts' / (binding.run_id + '.json')
    if Path(launcher.controller_receipt) != receipt_path:
        raise EvidenceError('lifecycle_controller_path_mismatch')
    receipt = ControllerReceipt.model_validate_json(bound_bytes(receipt_path, binding.controller_receipt_file_sha256))
    return LifecycleRecords(
        review=review, receipt=receipt, launcher=launcher,
        manifest=Manifest.model_validate_json(worker['runtime-manifest.json']),
        preflight=LifecyclePreflight.model_validate_json(worker['preflight.json']),
        aggregate=Aggregate.model_validate_json(worker['aggregate.json']),
        ranks=tuple(AcceptedRank.model_validate_json(worker[f'rank-{i}.json']) for i in range(8)),
        sidecars=tuple(LifecycleSidecar.model_validate_json(worker[f'lifecycle-{i}.json']) for i in range(8)),
    )


def validate_lifecycle_records(records: LifecycleRecords) -> None:
    """Recompute derived predicates and cross-record identities, not review booleans."""
    binding = records.review.source_binding
    receipt = records.receipt
    launcher = records.launcher
    controller = receipt.to_binding_evidence()
    if (
        (receipt.job_id, receipt.run_id, receipt.nonce, receipt.submitted_request_sha256, receipt.source_manifest_sha256)
        != (binding.job_id, binding.run_id, binding.nonce, binding.request_sha256, binding.manifest_sha256)
        or controller.controller_receipt_sha256 != binding.controller_binding_canonical_sha256
        or (launcher.run_id, launcher.nonce, launcher.manifest_sha256)
        != (binding.run_id, binding.nonce, binding.manifest_sha256)
        or (records.preflight.nonce, records.preflight.manifest_sha256) != (binding.nonce, binding.manifest_sha256)
    ):
        raise EvidenceError('lifecycle_runtime_identity_mismatch')
    if (
        tuple(rank.rank for rank in records.ranks) != tuple(range(8))
        or tuple(sidecar.binding.rank for sidecar in records.sidecars) != tuple(range(8))
        or tuple(rank.rank for rank in records.review.rank_results) != tuple(range(8))
        or records.aggregate.rank_results != records.ranks
        or len(records.manifest.files) != binding.manifest_entries
    ):
        raise EvidenceError('lifecycle_rank_inventory_or_order_mismatch')
    checkpoints = tuple(entry for entry in records.manifest.files if entry.role == 'checkpoint_model')
    if len(checkpoints) != 1:
        raise EvidenceError('lifecycle_checkpoint_cardinality')
    checkpoint = checkpoints[0]
    for rank, sidecar, summary in zip(records.ranks, records.sidecars, records.review.rank_results, strict=True):
        core = rank.evidence
        observed = sidecar.observations
        side = sidecar.binding
        if (
            rank.rank != core.rank or rank.nonce != binding.nonce or rank.checks != REQUIRED_CHECKS
            or core.controller_binding != controller or side.controller != controller
            or core.source_manifest_sha256 != binding.manifest_sha256 or core.verified_source_files != 118
            or (side.run_id, side.nonce, side.source_manifest_sha256) != (binding.run_id, binding.nonce, binding.manifest_sha256)
            or core.checkpoint_sha256 != checkpoint.sha256 or core.checkpoint_size_bytes != checkpoint.size_bytes
            or side.checkpoint.model_dump(mode='json') != checkpoint.model_dump(mode='json')
            or core.checkpoint_step != side.checkpoint_step or core.checkpoint_state_tensors != side.checkpoint_state_tensors
            or core.parameters != records.ranks[0].evidence.parameters or core.geometry != side.geometry
            or (core.rank, core.local_rank, core.world_size) != (side.rank, side.local_rank, side.world_size)
            or summary.device != side.device or side.device != f'cuda:{rank.rank}'
        ):
            raise EvidenceError('lifecycle_rank_binding_mismatch')
        if not sidecar.passed or observed is None or not observed.passed:
            raise EvidenceError('lifecycle_derived_predicate_failed')
        if (
            observed.device != f'cuda:{rank.rank}' or observed.model_calls != 12 or observed.synchronizations != 12
            or sidecar.before is None or sidecar.before.parameter_dtypes != ('torch.bfloat16',)
            or sidecar.before.torch_version != core.torch_version
        ):
            raise EvidenceError('lifecycle_observation_settings_mismatch')


def load_v7(root: Path) -> LifecycleQualification:
    records = read_lifecycle_records(root)
    validate_lifecycle_records(records)
    first = records.ranks[0].evidence
    return LifecycleQualification(
        review_sha256=REVIEW_SHA256, review_ledger_sha256=LEDGER_SHA256,
        source_binding=records.review.source_binding, ranks=tuple(rank.rank for rank in records.ranks),
        devices=tuple(sidecar.binding.device for sidecar in records.sidecars), parameter_categories=first.parameters,
        checkpoint=next(entry for entry in records.manifest.files if entry.role == 'checkpoint_model'),
        controller_binding=records.receipt.to_binding_evidence(),
        full_model_prefix_cache=first.fullmodel_cachedprefix, loop_prefix_cache=first.loop_cachedprefix,
    )
