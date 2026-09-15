"""Reconcile immutable v8 observations without executing the runtime probe."""

from hashlib import sha256
from pathlib import Path
from typing import Final

from .architecture_evidence_io import EvidenceError, bound_bytes, read_ledger
from .architecture_evidence_models import AcceptedRank, Aggregate, Launcher, Manifest
from .architecture_semantics_models import (
    GroupCount, SemanticsPreflight, SemanticsQualification, SemanticsRankSummary,
    SemanticsRecords, SemanticsReview,
)
from .qualification.controller_receipt import ControllerReceipt
from .qualification.evidence_contracts import REQUIRED_CHECKS
from .qualification.lifecycle_sidecar import LifecycleSidecar
from .qualification.semantics_sidecar import SemanticsSidecar

CAMPAIGN: Final = Path('.omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260914-08-live')
REVIEW_SHA256: Final = '4996f460a0ac6befe2f42e6a17b15a555d938f65d352faebe6534087b923b6f0'
LEDGER_SHA256: Final = '5c5bb4f6dd3384c968512b0d8de69576847d410a31f6deb4075c06e307965427'
WORKER_ROOT: Final = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification')


def read_semantics_records(root: Path) -> SemanticsRecords:
    campaign = root / CAMPAIGN
    directory = campaign / 'runtime-review'
    reviewed = read_ledger(directory / 'evidence-sha256.txt', LEDGER_SHA256)
    if set(reviewed) != {
        'commands-and-results.md', 'input-sha256.txt', 'review.md', 'scheduler-getjob.json',
        'scheduler-getjob.stderr.txt', 'verdict.json', 'verify_semantics_v8.py',
    }:
        raise EvidenceError('semantics_review_inventory_mismatch')
    review = SemanticsReview.model_validate_json(bound_bytes(directory / 'verdict.json', REVIEW_SHA256))
    binding = review.source_binding
    _ = bound_bytes(campaign / 'createjob-spec.json', binding.request_sha256)
    worker: dict[str, bytes] = {}
    for line in bound_bytes(campaign / 'worker-artifacts.sha256', binding.worker_ledger_sha256).decode().splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise EvidenceError('semantics_worker_ledger_schema')
        digest, name = fields
        if name in worker or Path(name).name != name:
            raise EvidenceError('semantics_worker_ledger_path')
        worker[name] = bound_bytes(WORKER_ROOT / binding.run_id / name, digest)
    expected = {f'{prefix}-{i}.json' for prefix in ('rank', 'lifecycle', 'model-semantics') for i in range(8)}
    expected |= {'launcher.json', 'aggregate.json', 'preflight.json', 'runtime_environment.json', 'runtime-manifest.json'}
    if set(worker) != expected or sha256(worker['runtime-manifest.json']).hexdigest() != binding.manifest_sha256:
        raise EvidenceError('semantics_worker_inventory_or_manifest_mismatch')
    manifest = Manifest.model_validate_json(worker['runtime-manifest.json'])
    for name in ('semantics_evidence.py', 'semantics_sidecar.py', 'lifecycle_evidence.py',
                 'lifecycle_sidecar.py', 'lifecycle_binding.py', 'lifecycle_settings.py'):
        relative = 'scale/experiments/nonlatent_iclr/qualification/' + name
        entries = tuple(entry for entry in manifest.files if entry.path.endswith('/' + relative))
        if len(entries) != 1:
            raise EvidenceError('semantics_validator_manifest_cardinality')
        _ = bound_bytes(root / relative, entries[0].sha256)
    launcher = Launcher.model_validate_json(worker['launcher.json'])
    receipt_path = WORKER_ROOT / 'controller-receipts' / (binding.run_id + '.json')
    if Path(launcher.controller_receipt) != receipt_path:
        raise EvidenceError('semantics_controller_path_mismatch')
    receipt = ControllerReceipt.model_validate_json(bound_bytes(receipt_path, binding.controller_receipt_file_sha256))
    return SemanticsRecords(
        review=review, receipt=receipt, launcher=launcher, manifest=manifest,
        preflight=SemanticsPreflight.model_validate_json(worker['preflight.json']),
        aggregate=Aggregate.model_validate_json(worker['aggregate.json']),
        ranks=tuple(AcceptedRank.model_validate_json(worker[f'rank-{i}.json']) for i in range(8)),
        lifecycle=tuple(LifecycleSidecar.model_validate_json(worker[f'lifecycle-{i}.json']) for i in range(8)),
        semantics=tuple(SemanticsSidecar.model_validate_json(worker[f'model-semantics-{i}.json']) for i in range(8)),
    )


def summarize_semantics(sidecar: SemanticsSidecar) -> SemanticsRankSummary:
    """Require authoritative predicates before deriving review-comparable counts."""
    if not sidecar.passed or sidecar.mask is None or sidecar.gradients is None:
        raise EvidenceError('semantics_derived_predicate_failed')
    baseline, frozen = sidecar.membership
    rows = sidecar.gradients.parameters
    by_name = {row.name: row for row in rows}
    if (
        any(group.numel != sum(by_name[name].numel for name in group.names) for group in baseline.groups)
        or any(row.gated != ('attn_bwd' not in row.name and 'fuse_' not in row.name) for row in rows)
        or any(not row.requires_grad or 'latent_cond' in row.name for row in rows)
        or frozen.trainable_tensors != 0 or frozen.trainable_numel != 0
        or baseline.trainable_tensors != 1953 or baseline.trainable_numel != 4091581441
    ):
        raise EvidenceError('semantics_parameter_policy_mismatch')
    return SemanticsRankSummary(
        rank=sidecar.binding.rank, device=sidecar.binding.device,
        baseline_groups=tuple(GroupCount(tensors=group.tensors, numel=group.numel) for group in baseline.groups),
        baseline_trainable_tensors=baseline.trainable_tensors, baseline_trainable_numel=baseline.trainable_numel,
        frozen_trainable_tensors=frozen.trainable_tensors, frozen_trainable_numel=frozen.trainable_numel,
        gated_nonzero_before=sum(row.gated and row.before == 'nonzero' for row in rows),
        ungated_nonzero_after=sum(not row.gated and row.after == 'nonzero' for row in rows),
        mask=sidecar.mask, membership_passed=(baseline.passed, frozen.passed),
        optimizer_state_entries=baseline.optimizer_state_entries + frozen.optimizer_state_entries,
        settings_unchanged=sidecar.before == sidecar.after, sidecar_passed=sidecar.passed,
    )


def validate_semantics_records(records: SemanticsRecords) -> tuple[SemanticsRankSummary, ...]:
    binding = records.review.source_binding
    receipt = records.receipt
    controller = receipt.to_binding_evidence()
    launcher = records.launcher
    if (
        (receipt.job_id, receipt.run_id, receipt.nonce, receipt.submitted_request_sha256, receipt.source_manifest_sha256)
        != (binding.job_id, binding.run_id, binding.nonce, binding.request_sha256, binding.manifest_sha256)
        or controller.controller_receipt_sha256 != binding.controller_binding_canonical_sha256
        or (launcher.run_id, launcher.nonce, launcher.manifest_sha256) != (binding.run_id, binding.nonce, binding.manifest_sha256)
        or (records.preflight.nonce, records.preflight.manifest_sha256) != (binding.nonce, binding.manifest_sha256)
    ):
        raise EvidenceError('semantics_runtime_identity_mismatch')
    inventories = (
        tuple(row.rank for row in records.ranks), tuple(row.binding.rank for row in records.lifecycle),
        tuple(row.binding.rank for row in records.semantics),
        tuple(row.rank for row in records.review.lifecycle_rank_results),
        tuple(row.rank for row in records.review.semantics_rank_results),
    )
    if (any(inventory != tuple(range(8)) for inventory in inventories)
            or records.aggregate.rank_results != records.ranks or len(records.manifest.files) != 126
            or len({entry.path for entry in records.manifest.files}) != 126):
        raise EvidenceError('semantics_rank_inventory_or_order_mismatch')
    checkpoints = tuple(entry for entry in records.manifest.files if entry.role == 'checkpoint_model')
    if len(checkpoints) != 1:
        raise EvidenceError('semantics_checkpoint_cardinality')
    checkpoint = checkpoints[0]
    observations: list[SemanticsRankSummary] = []
    for rank, lifecycle, semantics, summary in zip(
        records.ranks, records.lifecycle, records.semantics, records.review.lifecycle_rank_results, strict=True,
    ):
        core = rank.evidence
        side = lifecycle.binding
        if (
            rank.rank != core.rank or rank.nonce != binding.nonce or rank.checks != REQUIRED_CHECKS
            or core.controller_binding != controller or side.controller != controller or semantics.binding != side
            or core.source_manifest_sha256 != binding.manifest_sha256 or core.verified_source_files != 126
            or (side.run_id, side.nonce, side.source_manifest_sha256) != (binding.run_id, binding.nonce, binding.manifest_sha256)
            or (core.rank, core.local_rank, core.world_size) != (side.rank, side.local_rank, side.world_size)
            or side.device != f'cuda:{rank.rank}' or summary.device != side.device
            or core.checkpoint_sha256 != checkpoint.sha256 or core.checkpoint_size_bytes != checkpoint.size_bytes
            or side.checkpoint.model_dump(mode='json') != checkpoint.model_dump(mode='json')
            or core.checkpoint_step != side.checkpoint_step or core.checkpoint_state_tensors != side.checkpoint_state_tensors
            or core.geometry != side.geometry or core.parameters != records.ranks[0].evidence.parameters
            or semantics.trainer_source_sha256 != binding.trainer_source_sha256
        ):
            raise EvidenceError('semantics_rank_binding_mismatch')
        observed = lifecycle.observations
        if not lifecycle.passed or observed is None or not observed.passed:
            raise EvidenceError('semantics_lifecycle_predicate_failed')
        if (observed.model_calls != 12 or observed.synchronizations != 12 or observed.device != side.device
                or lifecycle.before is None or semantics.before is None
                or lifecycle.before.parameter_dtypes != ('torch.bfloat16',)
                or semantics.before.parameter_dtypes != ('torch.bfloat16',)
                or lifecycle.before.torch_version != core.torch_version or semantics.before.torch_version != core.torch_version):
            raise EvidenceError('semantics_settings_mismatch')
        observations.append(summarize_semantics(semantics))
        gradients = semantics.gradients
        first_gradients = records.semantics[0].gradients
        if gradients is None or first_gradients is None:
            raise EvidenceError('semantics_gradients_missing')
        if tuple((row.name, row.numel, row.requires_grad) for row in gradients.parameters) != tuple(
            (row.name, row.numel, row.requires_grad) for row in first_gradients.parameters
        ):
            raise EvidenceError('semantics_parameter_inventory_mismatch')
    if tuple(observations) != records.review.semantics_rank_results:
        raise EvidenceError('semantics_review_observation_mismatch')
    return tuple(observations)


def load_v8(root: Path) -> SemanticsQualification:
    records = read_semantics_records(root)
    observations = validate_semantics_records(records)
    first = records.ranks[0].evidence
    return SemanticsQualification(
        review_sha256=REVIEW_SHA256, review_ledger_sha256=LEDGER_SHA256,
        source_binding=records.review.source_binding, ranks=tuple(row.rank for row in records.ranks),
        checkpoint=next(entry for entry in records.manifest.files if entry.role == 'checkpoint_model'),
        controller_binding=records.receipt.to_binding_evidence(), parameter_categories=first.parameters,
        rank_observations=observations, full_model_prefix_cache=first.fullmodel_cachedprefix,
        loop_prefix_cache=first.loop_cachedprefix,
    )
