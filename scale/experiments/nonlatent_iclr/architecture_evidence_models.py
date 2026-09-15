"""Typed projections of hash-pinned historical records; omitted fields remain hash-bound."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from .qualification.controller_receipt import ControllerBindingEvidence
from .qualification.evidence_contracts import CheckName, HexDigest, Nonce, ParameterEvidence, SuccessEvidence


class Record(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True, strict=True)


class ReviewBinding(Record):
    job_id: str
    run_id: str
    nonce: Nonce
    request_sha256: HexDigest
    manifest_sha256: HexDigest
    controller_receipt_sha256: HexDigest
    worker_output_ledger_sha256: HexDigest
    monitoring_ledger_sha256: HexDigest


class BoundedCounts(Record):
    ranks: tuple[int, ...]
    world_size: int
    checkpoint_step: int
    checkpoint_state_tensors: int
    parameter_numel: int
    masked_logits_shape: tuple[int, ...]
    masked_logits_all_finite: bool
    standalone_fla_cache_max_abs_delta: float


class Review(Record):
    verdict: Literal["PASS"]
    promotion_target: Literal["BOUNDED_PARTIAL_QUALIFICATION_MILESTONE_ONLY"]
    task_3_complete: Literal[False]
    source_binding: ReviewBinding
    bounded_evidence_retained: BoundedCounts


class CheckpointSuccess(SuccessEvidence):
    checkpoint_sha256: Literal["ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07"]


class AcceptedRank(Record):
    rank: int
    nonce: Nonce
    status: Literal["PASSED"]
    checks: tuple[CheckName, ...]
    evidence: CheckpointSuccess


class Aggregate(Record):
    status: Literal["PARTIAL_QUALIFICATION"]
    rank_results: tuple[AcceptedRank, ...]


class Launcher(Record):
    status: Literal["PARTIAL_QUALIFICATION"]
    nonce: Nonce
    manifest_sha256: HexDigest
    run_id: str
    controller_receipt: str


class ManifestFile(Record):
    path: str
    role: str
    sha256: HexDigest
    size_bytes: int


class Manifest(Record):
    files: tuple[ManifestFile, ...]


class RuntimeQualification(BoundedCounts):
    review_sha256: HexDigest
    source_binding: ReviewBinding
    checkpoint: ManifestFile
    controller_binding: ControllerBindingEvidence
    full_model_prefix_cache: str
    loop_prefix_cache: str
    observed_image_digest: None = None
    long_context_claim: Literal[False] = False
    performance_or_goodput_claim: Literal[False] = False
    arbitrary_torch_binary_identity_claim: Literal[False] = False
    torch_version_interpretation: Literal["nv25.6_nv25.06_metadata_normalization_only"] = "nv25.6_nv25.06_metadata_normalization_only"


class LifecycleReviewBinding(Record):
    job_id: Literal["job-5b99b0c6-bb92-44e2-924b-761de4ad276b"]
    run_id: Literal["qualification-20260912-01-462d8c9d-eb9b-41fd-9992-d1d44528bc3a"]
    nonce: Literal["859969188464fd48e9c81beddca875b9"]
    request_sha256: Literal["a744ff2fdd62dcfb1aec8d0b9614757d74f7a7fa91155e2578ed9202623b58e1"]
    manifest_sha256: Literal["a38e0558a721ab9a1e6c37711c7e1d5599b5cdc8778fa28f4a878500ddd499ed"]
    manifest_entries: Literal[118]
    controller_receipt_file_sha256: HexDigest
    controller_binding_canonical_sha256: HexDigest
    worker_ledger_sha256: HexDigest


class LifecycleReviewRank(Record):
    rank: int
    device: str
    rank_status: Literal["PASSED"]
    lifecycle_passed: Literal[True]
    sidecar_passed: Literal[True]
    calls: Literal[12]
    synchronizations: Literal[12]
    comparisons: Literal[9]
    zero_call_rejections: Literal[8]
    caller_boundary_injections: Literal[1]
    max_abs_comparison_error: float = Field(ge=0.0, le=0.0)
    settings_unchanged: Literal[True]


class LifecycleReview(Record):
    verdict: Literal["PASS"]
    promotion_target: Literal["BOUNDED_FULL_CANVAS_GPU_LIFECYCLE_MILESTONE_ONLY"]
    task_3_complete: Literal[False]
    scheduler_status: Literal["job_succeeded"]
    scientific_status: Literal["PARTIAL_QUALIFICATION"]
    source_binding: LifecycleReviewBinding
    rank_results: tuple[LifecycleReviewRank, ...]


class LifecycleQualification(Record):
    review_sha256: HexDigest
    review_ledger_sha256: HexDigest
    source_binding: LifecycleReviewBinding
    ranks: tuple[int, ...]
    devices: tuple[str, ...]
    checkpoint: ManifestFile
    controller_binding: ControllerBindingEvidence
    parameter_categories: ParameterEvidence
    model_calls_per_rank: Literal[12] = 12
    synchronizations_per_rank: Literal[12] = 12
    scope: Literal["bounded_gpu_full_canvas_no_cache_nonlatent_lifecycle"] = "bounded_gpu_full_canvas_no_cache_nonlatent_lifecycle"
    derived_lifecycle_passed: Literal[True] = True
    full_model_prefix_cache: Literal["NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED"]
    loop_prefix_cache: Literal["NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES"]
    optimizer_or_training_qualification: Literal[False] = False
    mask_loss_semantics_qualification: Literal[False] = False
    whole_architecture_ready: Literal[False] = False


class LifecyclePreflight(Record):
    status: Literal['PREFLIGHT_VERIFIED']
    nonce: Nonce
    manifest_sha256: HexDigest
    verified_source_files: Literal[118]
