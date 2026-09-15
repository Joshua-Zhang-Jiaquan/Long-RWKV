from typing import Literal

from .architecture_evidence_models import (
    AcceptedRank, Aggregate, Launcher, LifecycleReviewRank, Manifest, ManifestFile, Record,
)
from .qualification.controller_receipt import ControllerBindingEvidence, ControllerReceipt
from .qualification.evidence_contracts import HexDigest, Nonce, ParameterEvidence
from .qualification.lifecycle_sidecar import LifecycleSidecar
from .qualification.semantics_evidence import MaskEvidence
from .qualification.semantics_sidecar import SemanticsSidecar


class SemanticsReviewBinding(Record):
    job_id: Literal['job-cd3c6850-91cc-4a1e-8f10-a3ce6c5baec9']
    run_id: Literal['qualification-20260912-01-3257f249-eace-4a74-841f-0d7d8e0e32e2']
    nonce: Literal['a9750fd783cc98d70435b185135ebd38']
    request_sha256: Literal['ffcb99d8b34fdf29a6c8d52aa342d589abc063080143ca34a6860d58ad703ffb']
    manifest_sha256: Literal['bae0fa5a71aca185d1718757d56de2d8f01a2606575d2d907b7a0d88e82ab68a']
    manifest_entries: Literal[126]
    controller_receipt_file_sha256: HexDigest
    controller_binding_canonical_sha256: HexDigest
    worker_ledger_sha256: HexDigest
    trainer_source_sha256: Literal['0bee15b5af05a785b70ddfeffa3064161f04beccea36b44e9bfd01029e286b60']


class GroupCount(Record):
    tensors: int
    numel: int


class SemanticsRankSummary(Record):
    rank: int
    device: str
    baseline_groups: tuple[GroupCount, ...]
    baseline_trainable_tensors: int
    baseline_trainable_numel: int
    frozen_trainable_tensors: int
    frozen_trainable_numel: int
    gated_nonzero_before: int
    ungated_nonzero_after: int
    mask: MaskEvidence
    membership_passed: tuple[bool, bool]
    optimizer_state_entries: int
    settings_unchanged: bool
    sidecar_passed: bool


class SemanticsReview(Record):
    verdict: Literal['PASS']
    promotion_target: Literal['BOUNDED_MODEL_SEMANTICS_AND_LIFECYCLE_RUNTIME_MILESTONE']
    scheduler_status: Literal['job_succeeded']
    scientific_status: Literal['PARTIAL_QUALIFICATION']
    source_binding: SemanticsReviewBinding
    lifecycle_rank_results: tuple[LifecycleReviewRank, ...]
    semantics_rank_results: tuple[SemanticsRankSummary, ...]


class SemanticsPreflight(Record):
    status: Literal['PREFLIGHT_VERIFIED']
    nonce: Nonce
    manifest_sha256: HexDigest
    verified_source_files: Literal[126]


class SemanticsRecords(Record):
    review: SemanticsReview
    receipt: ControllerReceipt
    launcher: Launcher
    manifest: Manifest
    preflight: SemanticsPreflight
    aggregate: Aggregate
    ranks: tuple[AcceptedRank, ...]
    lifecycle: tuple[LifecycleSidecar, ...]
    semantics: tuple[SemanticsSidecar, ...]


class SemanticsQualification(Record):
    review_sha256: HexDigest
    review_ledger_sha256: HexDigest
    source_binding: SemanticsReviewBinding
    ranks: tuple[int, ...]
    checkpoint: ManifestFile
    controller_binding: ControllerBindingEvidence
    parameter_categories: ParameterEvidence
    rank_observations: tuple[SemanticsRankSummary, ...]
    lifecycle_calls_per_rank: Literal[12] = 12
    lifecycle_synchronizations_per_rank: Literal[12] = 12
    optimizer_membership_scope: Literal['loaded_model_baseline_and_frozen_nonlatent_two_groups'] = 'loaded_model_baseline_and_frozen_nonlatent_two_groups'
    gradient_policy_scope: Literal['stage_a_step_0_of_100_zero_preserves_none'] = 'stage_a_step_0_of_100_zero_preserves_none'
    mask_loss_scope: Literal['tested_corruption_actual_logits_rank_local_loss_backward'] = 'tested_corruption_actual_logits_rank_local_loss_backward'
    later_gradient_stages_gpu_qualified: Literal[False] = False
    optimizer_updates_or_state_serialization_qualified: Literal[False] = False
    distributed_loss_reduction_or_fsdp_qualified: Literal[False] = False
    general_training_or_convergence_claim: Literal[False] = False
    full_model_prefix_cache: Literal['NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED']
    loop_prefix_cache: Literal['NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES']
    whole_architecture_ready: Literal[False] = False
