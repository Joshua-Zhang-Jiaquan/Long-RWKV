"""Per-arm measurement table for the bounded throughput calibration.

One rank cannot measure every arm the plan registers, because the controlled arms differ in
direction, objective and recycled depth. This module names exactly one arm per rank and records,
for each arm, whether its parameters actually exist in the code (``measured``) or whether its cost
must later be derived from another arm (A0 from A1, A4 from A3). A derived arm carries
``derived_from`` so the Task-6 ledger cannot present a derivation as a measurement.

Two further distinctions the table refuses to blur:

* The small (~0.4B) arms load **no checkpoint** — the HF weights are a warm start for the
  bidirectional wrapper, so their *compute shape* is real while their *accuracy* is untrained.
* The small-model root is on the volume qz workers mount. The project-volume copy of the same
  model is not reachable inside a job, so it must never be named here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .calibration_contracts import MeasuredGeometry, MeasuredParameters

RANK_COUNT: Final = 8

GLOBAL_USER: Final = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")
LARGE_MODEL_ROOT: Final = GLOBAL_USER / "models/RWKV7-Goose-World3-2.9B-HF"
LARGE_CHECKPOINT_ROOT: Final = GLOBAL_USER / "m2_baseline_triangle/m4loop_endpoint_ckpt"
SMALL_MODEL_ROOT: Final = GLOBAL_USER / "models/rwkv7-0.4B"

#: The step-4750 loop endpoint. The raw ``outputs_birwkv_diffusion`` directory was pruned; this
#: neutral copy is the surviving artifact and is the identity the accepted qualifications pinned.
LARGE_CHECKPOINT_STEP: Final = 4750
LARGE_CHECKPOINT_MODEL_SHA256: Final = (
    "ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07"
)
LARGE_CONFIG_SHA256: Final = (
    "08bf100980c8fe9e69d5c149fba30abc5903a57ad8e6d3166442a2efe7a25783"
)
LARGE_INDEX_SHA256: Final = (
    "d83256f944260c29571d24cdb5b25750536c7c759f3cb85acc82a5ea89371fe6"
)
SMALL_MODEL_WEIGHTS_SHA256: Final = (
    "e162387e439dfa3387a0ca7da61638749d00c9862b8cc0192ae5d366c8c1a524"
)
SMALL_CONFIG_SHA256: Final = (
    "1131b80caf1fe16e5f67dd8c2965fa61467a5188be52de479deecde3fabbfa5a"
)

#: The manifest can express exactly one HF root, so the ~0.4B model is pinned here instead.
SMALL_WEIGHTS_PIN: Final[dict[str, str]] = {
    "model.safetensors": SMALL_MODEL_WEIGHTS_SHA256,
    "config.json": SMALL_CONFIG_SHA256,
}

LARGE_LOOP_RANGE: Final = (16, 32)
SMALL_LOOP_RANGE: Final = (12, 24)


class ArmSpec(BaseModel):
    """One measurable arm, or one arm whose cost is declared derived from another."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    arm_id: str = Field(min_length=1)
    nominal_label: str = Field(min_length=1)
    model_root: Path
    checkpoint_dir: Path | None = None
    checkpoint_step: int | None = Field(default=None, gt=0)
    loop_range: tuple[int, int] | None = None
    loop_reps: int = Field(ge=0, le=2)
    loop_reps_measured: int | None = Field(default=None, ge=0, le=2)
    force_forward: bool
    optimizer_step_measured: bool
    measured: bool
    derived_from: str | None = None
    expected_weights_sha256: dict[str, str] = Field(default_factory=dict)
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def enforce_loop_declaration(self) -> Self:
        if self.loop_reps > 0 and self.loop_range is None:
            raise ValueError("loop_reps_requires_loop_range")
        if self.loop_reps == 0 and self.loop_range is not None:
            raise ValueError("loop_range_without_loop_reps")
        return self

    @model_validator(mode="after")
    def enforce_measured_loop_within_configured(self) -> Self:
        if self.loop_reps_measured is not None and self.loop_reps_measured > self.loop_reps:
            raise ValueError("loop_reps_measured_exceeds_configured")
        return self

    def measured_reps(self) -> int:
        """Recycled depth this arm actually executes, which may be below the configured depth."""
        return self.loop_reps if self.loop_reps_measured is None else self.loop_reps_measured

    @model_validator(mode="after")
    def enforce_checkpoint_declaration(self) -> Self:
        declared = self.checkpoint_dir is not None
        stepped = self.checkpoint_step is not None
        if declared != stepped:
            raise ValueError("checkpoint_dir_and_step_must_be_declared_together")
        return self

    @model_validator(mode="after")
    def enforce_measured_or_derived(self) -> Self:
        if self.measured == (self.derived_from is not None):
            raise ValueError("arm_must_be_either_measured_or_derived")
        return self

    @model_validator(mode="after")
    def enforce_derived_arms_measure_nothing(self) -> Self:
        if not self.measured and (self.checkpoint_dir is not None or self.optimizer_step_measured):
            raise ValueError("derived_arm_cannot_claim_measurements")
        return self

    def digest(self) -> str:
        """Stable identity of this arm's declaration, for binding into a sidecar."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ARM_SPECS: Final[tuple[ArmSpec, ...]] = (
    ArmSpec(
        arm_id="A1_small_noloop",
        nominal_label="A1 forward-only recurrent masked denoiser, no loop",
        model_root=SMALL_MODEL_ROOT,
        expected_weights_sha256=SMALL_WEIGHTS_PIN,
        loop_reps=0,
        force_forward=True,
        optimizer_step_measured=True,
        measured=True,
        notes="Small-model warm start only: no trained checkpoint is loaded, so this measures the arm's compute shape, not its accuracy.",
    ),
    ArmSpec(
        arm_id="A2_small_noloop",
        nominal_label="A2 bidirectional recurrent masked denoiser, no loop",
        model_root=SMALL_MODEL_ROOT,
        expected_weights_sha256=SMALL_WEIGHTS_PIN,
        loop_reps=0,
        force_forward=False,
        optimizer_step_measured=True,
        measured=True,
        notes="Small-model warm start only; the Task-7 control and the Task-8 baseline shape.",
    ),
    ArmSpec(
        arm_id="A3_small_loop",
        nominal_label="A3 bidirectional masked denoiser, tied loop",
        model_root=SMALL_MODEL_ROOT,
        expected_weights_sha256=SMALL_WEIGHTS_PIN,
        loop_range=SMALL_LOOP_RANGE,
        loop_reps=1,
        force_forward=False,
        optimizer_step_measured=True,
        measured=True,
        notes="Main candidate at small scale. The tied loop adds gate tensors only, not new blocks, so its parameter count matches A2 apart from those gates.",
    ),
    ArmSpec(
        arm_id="A5_small_loop_fwd",
        nominal_label="A5 forward-only masked denoiser, tied loop",
        model_root=SMALL_MODEL_ROOT,
        expected_weights_sha256=SMALL_WEIGHTS_PIN,
        loop_range=SMALL_LOOP_RANGE,
        loop_reps=1,
        force_forward=True,
        optimizer_step_measured=True,
        measured=True,
        notes="Interaction of looping and bidirectionality at small scale.",
    ),
    ArmSpec(
        arm_id="A2_large_noloop",
        nominal_label="A2 at the nominal-2.9B loop checkpoint, loop executed zero times",
        model_root=LARGE_MODEL_ROOT,
        checkpoint_dir=LARGE_CHECKPOINT_ROOT,
        checkpoint_step=LARGE_CHECKPOINT_STEP,
        loop_range=LARGE_LOOP_RANGE,
        loop_reps=1,
        loop_reps_measured=0,
        force_forward=False,
        optimizer_step_measured=False,
        measured=True,
        notes="Built with loop_reps=1 so the step-4750 checkpoint loads under strict=True, then measured with loop_reps_override=0. The loop module is present but executes zero times, so its gate tensors are counted but not timed. Single-GPU AdamW state for 4.09B parameters is not the training configuration, so no optimizer step is timed.",
    ),
    ArmSpec(
        arm_id="A3_large_loop",
        nominal_label="A3 at the nominal-2.9B step-4750 loop checkpoint",
        model_root=LARGE_MODEL_ROOT,
        checkpoint_dir=LARGE_CHECKPOINT_ROOT,
        checkpoint_step=LARGE_CHECKPOINT_STEP,
        loop_range=LARGE_LOOP_RANGE,
        loop_reps=1,
        force_forward=False,
        optimizer_step_measured=False,
        measured=True,
        notes="The arm the historical v7 pilot measured; identity pinned to the surviving neutral copy of step 4750.",
    ),
)

#: rank -> arm_id. Ranks 4 and 5 repeat the two decision-relevant small arms so the baseline and
#: the candidate each carry a second independent GPU reading rather than one.
RANK_ARM_TABLE: Final[dict[int, str]] = {
    0: "A1_small_noloop",
    1: "A2_small_noloop",
    2: "A3_small_loop",
    3: "A5_small_loop_fwd",
    4: "A2_small_noloop",
    5: "A3_small_loop",
    6: "A3_large_loop",
    7: "A2_large_noloop",
}

#: Registered arms whose cost this job cannot measure, with the arm each is derived from. A0 has
#: no causal-objective switch in the trainer; A4 has no untied-extra-block implementation.
DERIVED_ARMS: Final[dict[str, str]] = {
    "A0": "A1_small_noloop",
    "A4": "A3_small_loop",
}

_BY_ID: Final[dict[str, ArmSpec]] = {spec.arm_id: spec for spec in ARM_SPECS}


def arm_for_rank(rank: int) -> ArmSpec:
    """Resolve one rank to its arm, rejecting anything outside the declared world size."""
    if rank not in RANK_ARM_TABLE:
        raise ValueError(f"rank outside declared world size: {rank}")
    return _BY_ID[RANK_ARM_TABLE[rank]]


def arm_by_id(arm_id: str) -> ArmSpec:
    try:
        return _BY_ID[arm_id]
    except KeyError as error:
        raise ValueError(f"unknown arm: {arm_id}") from error


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Stream a file so a multi-gigabyte weight file is never materialised in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def measure_geometry(model) -> MeasuredGeometry:
    """Read the built model's geometry, without the qualifications' 2.9B-pinned literal type."""
    config = model.config
    return MeasuredGeometry(
        num_hidden_layers=int(config.num_hidden_layers),
        hidden_size=int(config.hidden_size),
        vocab_size=int(config.vocab_size),
        num_heads=None if config.num_heads is None else int(config.num_heads),
        head_dim=int(config.head_dim),
        intermediate_size=int(config.intermediate_size),
    )


def measure_parameters(model) -> MeasuredParameters:
    """Count tensors and elements, separating the loop's own tensors from the shared trunk."""
    total_tensors = 0
    total_numel = 0
    loop_tensors = 0
    loop_numel = 0
    for name, parameter in model.named_parameters():
        numel = int(parameter.numel())
        total_tensors += 1
        total_numel += numel
        if name.startswith("loop."):
            loop_tensors += 1
            loop_numel += numel
    return MeasuredParameters(
        total_tensors=total_tensors,
        total_numel=total_numel,
        loop_tensors=loop_tensors,
        loop_numel=loop_numel,
    )


def verify_arm_weights(arm: ArmSpec) -> dict[str, str]:
    """Hash this arm's weight files in-job, so the small arm is pinned where the manifest cannot.

    The runtime manifest expresses exactly one HF root (one index, and an exact shard-name match
    against it), so the ~0.4B model cannot be a manifest entry without weakening that contract.
    Its identity is pinned on the arm's own declaration instead, and recorded in the sidecar, so
    pointing an arm at different bytes is refused rather than silently measured.
    """
    weights = sorted(arm.model_root.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"arm_weights_absent:{arm.model_root}")
    observed = {path.name: sha256_file(path) for path in weights}
    config = arm.model_root / "config.json"
    if not config.is_file():
        raise ValueError(f"arm_config_absent:{config}")
    observed["config.json"] = sha256_file(config)
    for name, digest in arm.expected_weights_sha256.items():
        if observed.get(name) != digest:
            raise ValueError(f"arm_weights_digest_mismatch:{name}")
    return observed
