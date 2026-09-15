from __future__ import annotations

from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .controller_receipt import ControllerBindingEvidence


type CheckName = Literal[
    "hardware",
    "source_identity",
    "checkpointload",
    "parameteridentity",
    "realFLAcausalcache",
    "actualmaskedforward",
    "resource_limits",
]
type FailureCheck = CheckName | Literal["runtime_boundary", "distributed_teardown"]
type HexDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type Nonce = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]

REQUIRED_CHECKS: Final[tuple[CheckName, ...]] = (
    "hardware",
    "source_identity",
    "checkpointload",
    "parameteridentity",
    "realFLAcausalcache",
    "actualmaskedforward",
    "resource_limits",
)
CHECKPOINT_SHA256: Final = "ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07"


class QualificationContractError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class QualificationInputError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class QualificationRuntimeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class GPURecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    uuid: str = Field(min_length=1)
    memory_mib: int = Field(ge=79_000)


class ModelGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    num_hidden_layers: Literal[32]
    hidden_size: Literal[2_560]
    vocab_size: Literal[65_536]
    num_heads: Literal[40]
    head_dim: Literal[64]
    intermediate_size: Literal[10_240]


class ParameterEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    total_tensors: Literal[1_953]
    total_numel: Literal[4_091_581_441]
    forward_attention_tensors: Literal[829]
    forward_attention_numel: Literal[934_049_280]
    backward_attention_tensors: Literal[829]
    backward_attention_numel: Literal[934_049_280]
    fusion_tensors: Literal[64]
    fusion_numel: Literal[209_797_120]
    shared_tensors: Literal[230]
    shared_numel: Literal[2_013_685_760]
    loop_tensors: Literal[1]
    loop_numel: Literal[1]


class CausalCacheEvidence(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False, extra="forbid", frozen=True, strict=True
    )

    scope: Literal["standalone_forward_rwkv7_attention"]
    sequence_length: Literal[48]
    prefix_length: Literal[32]
    max_abs_delta: float = Field(ge=0.0, le=0.05)
    atol: Literal[0.05]
    rtol: Literal[0.0]


class MemoryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    allocated_after_load_bytes: int = Field(ge=0)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    physical_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def enforce_ceiling(self) -> Self:
        ceiling = int(self.physical_bytes * 0.90)
        if max(
            self.allocated_after_load_bytes,
            self.peak_allocated_bytes,
            self.peak_reserved_bytes,
        ) > ceiling:
            raise QualificationContractError("memory_evidence_exceeds_0_90")
        return self


class SuccessEvidence(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False, extra="forbid", frozen=True, strict=True
    )

    kind: Literal["success"]
    start_utc: float = Field(gt=0.0)
    end_utc: float = Field(gt=0.0)
    elapsed_monotonic_seconds: float = Field(gt=0.0, le=1_200.0)
    rank: int = Field(ge=0, le=7)
    local_rank: int = Field(ge=0, le=7)
    world_size: Literal[8]
    gpu_inventory: tuple[GPURecord, ...]
    controller_binding: ControllerBindingEvidence
    python_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    fla_version: str = Field(min_length=1)
    fla_core_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    safetensors_version: str = Field(min_length=1)
    triton_version: str = Field(min_length=1)
    source_manifest_sha256: HexDigest
    verified_source_files: int = Field(ge=1)
    checkpoint_step: Literal[4_750]
    checkpoint_sha256: Literal[CHECKPOINT_SHA256]
    checkpoint_size_bytes: Literal[16_367_167_378]
    checkpoint_state_tensors: Literal[1_955]
    geometry: ModelGeometry
    parameters: ParameterEvidence
    fla_causal_cache: CausalCacheEvidence
    masked_logits_all_finite: Literal[True]
    masked_logits_shape: tuple[Literal[1], Literal[48], Literal[65_536]]
    memory: MemoryEvidence
    fullmodel_cachedprefix: Literal[
        "NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED"
    ]
    loop_cachedprefix: Literal[
        "NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES"
    ]

    @model_validator(mode="after")
    def enforce_cross_field_identity(self) -> Self:
        if self.end_utc < self.start_utc:
            raise QualificationContractError("runtime_end_precedes_start")
        if self.rank != self.local_rank:
            raise QualificationContractError("global_local_rank_mismatch")
        if len(self.gpu_inventory) != 8:
            raise QualificationContractError("gpu_inventory_count_not_8")
        if len({gpu.uuid for gpu in self.gpu_inventory}) != 8:
            raise QualificationContractError("gpu_inventory_uuids_not_unique")
        if (
            self.source_manifest_sha256
            != self.controller_binding.source_manifest_sha256
        ):
            raise QualificationContractError("controller_manifest_identity_mismatch")
        return self


class FailureEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["failure"]
    failed_check: FailureCheck
    completed_checks: tuple[CheckName, ...]
    exception_type: str = Field(min_length=1)
    traceback: str = Field(min_length=1)
    source_manifest_sha256: HexDigest

    @model_validator(mode="after")
    def enforce_completed_prefix(self) -> Self:
        if self.completed_checks != REQUIRED_CHECKS[: len(self.completed_checks)]:
            raise QualificationContractError("completed_checks_not_ordered_prefix")
        return self
