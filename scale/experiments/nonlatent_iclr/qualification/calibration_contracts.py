"""Bounded throughput contract for the Task-6 forecast ledger.

These measurements exist to fund Task 6's experiment forecast with observed units. They are
**not** performance, goodput or long-context evidence, and the contract says so explicitly so
a downstream reader cannot promote them. Every field is a measurement that a rank actually
observed; nothing is estimated or back-filled.

One job measures several arms, so the aggregate is grouped **per arm**: mixing a 0.4B timing and a
2.9B timing into one median would be meaningless. The contract requires all eight ranks to report,
keeps a failed rank visible as an unavailable arm member, and refuses to publish a passing
aggregate that silently drops one.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import ClassVar, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

RANK_COUNT = 8
CANVAS_TOKENS = (4096, 16384, 32768)
SCOPE: Literal["task6_forecast_input_only"] = "task6_forecast_input_only"
CLAIMS_NOT_MADE = (
    "long-context capability",
    "sustainable goodput",
    "quality-matched serving",
    "physical edge-device performance",
    "training throughput at production sharding",
)


class MeasuredGeometry(BaseModel):
    """Geometry as observed on an arbitrary arm.

    ``evidence_contracts.ModelGeometry`` pins the 2.9B identity with ``Literal`` fields, which is
    right for qualification evidence and useless for a 0.4B arm, so calibration records the
    observed values under their own type rather than relaxing that contract.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    num_hidden_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    num_heads: int | None = None
    head_dim: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)


class MeasuredParameters(BaseModel):
    """Parameter counts as observed on an arbitrary arm, with the loop's share separated."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    total_tensors: int = Field(gt=0)
    total_numel: int = Field(gt=0)
    loop_tensors: int = Field(ge=0)
    loop_numel: int = Field(ge=0)


class CalibrationMeasurement(BaseModel):
    """One canvas size as measured by one rank.

    The backward and optimizer durations are optional because a given arm may schedule neither.
    When a rank has not measured one, the record must say so explicitly rather than leave a zero
    or an estimate, so a reader can never mistake an unmeasured quantity for a fast one.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    canvas_tokens: int = Field(gt=0, le=65536)
    forward_tokens_per_second: float = Field(gt=0)
    backward_seconds: float | None = Field(default=None, gt=0)
    backward_not_measured_reason: str | None = Field(default=None, min_length=1)
    optimizer_step_seconds: float | None = Field(default=None, gt=0)
    optimizer_step_not_measured_reason: str | None = Field(default=None, min_length=1)
    selected_tokens: int = Field(ge=0)
    sampler_nfe_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def enforce_reserved_at_least_allocated(self) -> Self:
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("peak_reserved_bytes_below_peak_allocated_bytes")
        return self

    @model_validator(mode="after")
    def enforce_backward_measured_or_explained(self) -> Self:
        measured = self.backward_seconds is not None
        explained = self.backward_not_measured_reason is not None
        if measured == explained:
            raise ValueError("backward_measurement_must_be_either_measured_or_explained")
        return self

    @model_validator(mode="after")
    def enforce_optimizer_measured_or_explained(self) -> Self:
        measured = self.optimizer_step_seconds is not None
        explained = self.optimizer_step_not_measured_reason is not None
        if measured == explained:
            raise ValueError("optimizer_measurement_must_be_either_measured_or_explained")
        return self


class CalibrationSidecar(BaseModel):
    """One rank's terminal measurement record, bound to the arm that rank actually measured."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    rank: int = Field(ge=0, lt=RANK_COUNT)
    nonce: str = Field(min_length=8)
    status: Literal["PASSED", "FAILED"]
    detail: str = Field(min_length=1)
    scope: Literal["task6_forecast_input_only"] = SCOPE
    claims_not_made: tuple[str, ...] = CLAIMS_NOT_MADE
    arm_id: str = Field(min_length=1)
    arm_digest: str = Field(min_length=64, max_length=64)
    nominal_label: str = Field(min_length=1)
    measured: bool
    derived_from: str | None = None
    loop_reps_configured: int = Field(ge=0, le=2)
    loop_reps_measured: int = Field(ge=0, le=2)
    force_forward: bool
    gradient_checkpointing: bool
    model_geometry: MeasuredGeometry | None = None
    parameters: MeasuredParameters | None = None
    checkpoint_step: int | None = Field(default=None, gt=0)
    checkpoint_sha256: str | None = None
    weights_sha256: dict[str, str] = Field(default_factory=dict)
    measurements: tuple[CalibrationMeasurement, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def enforce_status_variant(self) -> Self:
        if self.status == "PASSED" and not self.measurements:
            raise ValueError("passed_sidecar_without_measurements")
        if self.status == "FAILED" and self.measurements:
            raise ValueError("failed_sidecar_carries_measurements")
        return self

    @model_validator(mode="after")
    def enforce_passing_ranks_report_identity(self) -> Self:
        if self.status == "PASSED" and (self.model_geometry is None or self.parameters is None):
            raise ValueError("passed_sidecar_without_a_measured_identity")
        return self

    @model_validator(mode="after")
    def enforce_checkpoint_declared_together(self) -> Self:
        declared = self.checkpoint_step is not None
        hashed = self.checkpoint_sha256 is not None
        if declared != hashed:
            raise ValueError("checkpoint_step_and_sha256_must_be_declared_together")
        return self

    @model_validator(mode="after")
    def enforce_only_measured_arms_publish(self) -> Self:
        if not self.measured:
            raise ValueError("derived_arm_cannot_publish_a_sidecar")
        return self

    @model_validator(mode="after")
    def enforce_measured_loop_within_configured(self) -> Self:
        if self.loop_reps_measured > self.loop_reps_configured:
            raise ValueError("loop_reps_measured_exceeds_configured")
        return self

    def write_once(self, output_dir: Path) -> Path:
        """Publish this record once; an identical rewrite is a no-op, a different one is refused."""
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"calibration-rank-{self.rank}.json"
        payload = (self.model_dump_json(indent=2) + "\n").encode("utf-8")
        if target.is_file():
            if target.read_bytes() != payload:
                raise ValueError(f"calibration sidecar already exists with different content: {target}")
            return target
        _ = target.write_bytes(payload)
        return target


class CanvasAggregate(BaseModel):
    """Median across the passing ranks of one arm for one canvas size."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    canvas_tokens: int
    contributing_ranks: int
    forward_tokens_per_second: float
    backward_seconds: float | None
    backward_contributing_ranks: int
    backward_not_measured_reason: str | None
    optimizer_step_seconds: float | None
    optimizer_step_contributing_ranks: int
    optimizer_step_not_measured_reason: str | None
    selected_tokens_min: int
    selected_tokens_max: int
    sampler_nfe_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


class ArmAggregate(BaseModel):
    """One arm's measurements. An arm whose every rank failed keeps zero contributions and names them."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    arm_id: str
    nominal_label: str
    derived_from: str | None = None
    contributing_ranks: int
    unavailable_ranks: tuple[int, ...] = ()
    parameters: MeasuredParameters | None = None
    checkpoint_step: int | None = None
    model_geometry: MeasuredGeometry | None = None
    canvases: tuple[CanvasAggregate, ...] = ()
    canvases_not_reported: tuple[int, ...] = ()

    @model_validator(mode="after")
    def enforce_empty_arm_claims_nothing(self) -> Self:
        if self.contributing_ranks == 0:
            if self.canvases or self.parameters is not None or self.model_geometry is not None:
                raise ValueError("arm_without_contributions_cannot_report_measurements")
        elif not self.canvases:
            raise ValueError("arm_with_contributions_must_report_a_canvas")
        return self


class CalibrationAggregate(BaseModel):
    """The whole-job summary, grouped by arm and bound to the eight rank records it summarises."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    scope: Literal["task6_forecast_input_only"] = SCOPE
    claims_not_made: tuple[str, ...] = CLAIMS_NOT_MADE
    status: Literal["PASSED", "PARTIAL"]
    gradient_checkpointing: bool
    ranks_passed: int
    ranks_failed: int
    arms: tuple[ArmAggregate, ...]
    derived_arms: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    sidecar_sha256: dict[str, str]

    @model_validator(mode="after")
    def enforce_status_matches_rank_outcome(self) -> Self:
        if self.status == "PASSED" and self.ranks_failed:
            raise ValueError("passed_aggregate_cannot_hide_a_failed_rank")
        if self.status == "PARTIAL" and not self.ranks_failed:
            raise ValueError("partial_aggregate_requires_a_failed_rank")
        return self


def aggregate(
    sidecars: tuple[CalibrationSidecar, ...],
    *,
    derived_arms: dict[str, str] | None = None,
) -> CalibrationAggregate:
    """Summarise all eight ranks, grouped by arm, keeping failures visible.

    Every rank must report and the ranks must cover 0..7 exactly; a job that cannot say which
    checkpoint or arm it measured fails closed. Ranks that failed are reported as unavailable
    members of their arm rather than removed, so the aggregate cannot read as a complete one.
    """
    if len(sidecars) != RANK_COUNT:
        raise ValueError(f"expected {RANK_COUNT} rank records, received {len(sidecars)}")
    ranks = sorted(sidecar.rank for sidecar in sidecars)
    if ranks != list(range(RANK_COUNT)):
        raise ValueError(f"rank records do not cover 0..{RANK_COUNT - 1}: {ranks}")
    geometry_owners = {sidecar.gradient_checkpointing for sidecar in sidecars}
    if len(geometry_owners) != 1:
        raise ValueError("ranks disagree on the gradient-checkpointing configuration")
    failed = tuple(sidecar.rank for sidecar in sidecars if sidecar.status != "PASSED")
    arms: list[ArmAggregate] = []
    for arm_id in sorted({sidecar.arm_id for sidecar in sidecars}):
        members = tuple(sidecar for sidecar in sidecars if sidecar.arm_id == arm_id)
        passing = tuple(sidecar for sidecar in members if sidecar.status == "PASSED")
        unavailable = tuple(sorted(sidecar.rank for sidecar in members if sidecar.status != "PASSED"))
        if not passing:
            arms.append(ArmAggregate(
                arm_id=arm_id,
                nominal_label=members[0].nominal_label,
                derived_from=members[0].derived_from,
                contributing_ranks=0,
                unavailable_ranks=unavailable,
            ))
            continue
        numels = {sidecar.parameters for sidecar in passing}
        steps = {sidecar.checkpoint_step for sidecar in passing}
        geometries = {sidecar.model_geometry for sidecar in passing}
        if len(numels) != 1 or len(steps) != 1 or len(geometries) != 1:
            raise ValueError(f"ranks of arm {arm_id} disagree on the model identity")
        reported = sorted({m.canvas_tokens for sidecar in passing for m in sidecar.measurements})
        if not reported:
            raise ValueError(f"no canvas was measured by any rank of arm {arm_id}")
        canvases: list[CanvasAggregate] = []
        for canvas in reported:
            rows = [m for sidecar in passing for m in sidecar.measurements if m.canvas_tokens == canvas]
            measured_backward = [row.backward_seconds for row in rows if row.backward_seconds is not None]
            backward_reasons = sorted({row.backward_not_measured_reason for row in rows if row.backward_not_measured_reason})
            measured_optimizer = [row.optimizer_step_seconds for row in rows if row.optimizer_step_seconds is not None]
            optimizer_reasons = sorted({row.optimizer_step_not_measured_reason for row in rows if row.optimizer_step_not_measured_reason})
            canvases.append(CanvasAggregate(
                canvas_tokens=canvas,
                contributing_ranks=len(rows),
                forward_tokens_per_second=median(row.forward_tokens_per_second for row in rows),
                backward_seconds=median(measured_backward) if measured_backward else None,
                backward_contributing_ranks=len(measured_backward),
                backward_not_measured_reason=None if measured_backward else "; ".join(backward_reasons) or "no rank reported a backward measurement",
                optimizer_step_seconds=median(measured_optimizer) if measured_optimizer else None,
                optimizer_step_contributing_ranks=len(measured_optimizer),
                optimizer_step_not_measured_reason=None if measured_optimizer else "; ".join(optimizer_reasons) or "no rank reported an optimizer measurement",
                selected_tokens_min=min(row.selected_tokens for row in rows),
                selected_tokens_max=max(row.selected_tokens for row in rows),
                sampler_nfe_seconds=median(row.sampler_nfe_seconds for row in rows),
                peak_allocated_bytes=int(median(row.peak_allocated_bytes for row in rows)),
                peak_reserved_bytes=int(median(row.peak_reserved_bytes for row in rows)),
            ))
        arms.append(ArmAggregate(
            arm_id=arm_id,
            nominal_label=passing[0].nominal_label,
            derived_from=passing[0].derived_from,
            contributing_ranks=len(passing),
            unavailable_ranks=unavailable,
            parameters=numels.pop(),
            checkpoint_step=steps.pop(),
            model_geometry=geometries.pop(),
            canvases=tuple(canvases),
            canvases_not_reported=tuple(canvas for canvas in CANVAS_TOKENS if canvas not in reported),
        ))
    return CalibrationAggregate(
        status="PARTIAL" if failed else "PASSED",
        gradient_checkpointing=geometry_owners.pop(),
        ranks_passed=len(sidecars) - len(failed),
        ranks_failed=len(failed),
        arms=tuple(arms),
        derived_arms=dict(derived_arms or {}),
        warnings=tuple(sorted({warning for sidecar in sidecars for warning in sidecar.warnings})),
        sidecar_sha256={f"rank-{sidecar.rank}": _digest(sidecar) for sidecar in sidecars},
    )


def _digest(sidecar: CalibrationSidecar) -> str:
    from hashlib import sha256

    return sha256(sidecar.model_dump_json(indent=2).encode("utf-8")).hexdigest()


def load_sidecar(path: Path) -> CalibrationSidecar:
    """Parse one published rank record, rejecting anything the contract forbids.

    JSON has no tuple type, so the two tuple-typed fields are normalised before validation.
    Strictness is preserved for every other field: wrong scalar types, missing fields and
    unknown fields are still refused.
    """
    raw = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    for field in ("measurements", "warnings", "claims_not_made"):
        value = raw.get(field)
        if isinstance(value, list):
            raw[field] = tuple(cast("list[object]", value))
    return CalibrationSidecar.model_validate(raw)
