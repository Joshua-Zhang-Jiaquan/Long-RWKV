"""Task-6 allocation ledger: turn measured per-arm rates into a funded experiment forecast.

Every number here is either read from a hash-bound calibration aggregate or computed from one by a
stated formula. Nothing is estimated from a nominal parameter count, a datasheet figure, or a
previous campaign's timing.

Three things the ledger refuses to do:

* It will not price a run from a measurement that does not exist. An arm whose training-length rung
  was never measured is reported as unforecastable with the reason, not filled in from a smaller
  canvas or a sibling arm.
* It will not present a derived arm as measured. A0 is derived from A1 and A4 from A3, and every
  row says which.
* It will not smooth over its own assumptions. The per-step time is a microbatch-1 measurement, so
  it is a conservative upper bound on the GPU-hours a batched run would use, and the ledger says so
  in the row rather than in a footnote.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from ..qualification.calibration_contracts import CalibrationAggregate, CanvasAggregate

#: The training sequence length the plan's runs use, so forecasts are priced at that canvas.
TRAINING_CANVAS: Final = 4096
CANDIDATE_TOKEN_BUDGETS: Final = (2_000_000_000, 4_000_000_000, 8_000_000_000)
CONTROLLED_ARMS: Final = ("A0", "A1", "A2", "A3", "A4", "A5")
TRAINING_SEEDS: Final = (17, 29, 43)
CONFIRMATION_ARMS: Final = ("A2", "A3")

#: Which measured arm each unmeasured arm's cost is taken from. Disclosed on every row.
DERIVATIONS: Final = {"A0": "A1_small_noloop", "A4": "A3_small_loop"}
#: The measured arms of the controlled small-model matrix. Named explicitly because a prefix test
#: would pick up the 2.9B confirmation arms too: ``A2_large_noloop`` starts with ``A2`` but carries
#: 4.09B parameters, which inflated the storage line by a factor of seven.
SMALL_SOURCE_ARMS: Final = (
    "A1_small_noloop", "A2_small_noloop", "A3_small_loop", "A5_small_loop_fwd",
)
ARM_SOURCE: Final = {
    "A1": "A1_small_noloop",
    "A2": "A2_small_noloop",
    "A3": "A3_small_loop",
    "A5": "A5_small_loop_fwd",
}
ARM_LABEL: Final = {
    "A0": "causal RWKV, autoregressive objective, no loop",
    "A1": "forward-only recurrent masked denoiser, no loop",
    "A2": "bidirectional recurrent masked denoiser, no loop",
    "A3": "bidirectional masked denoiser, tied loop",
    "A4": "bidirectional masked denoiser, untied extra blocks",
    "A5": "forward-only masked denoiser, tied loop",
}

#: bf16 weights plus two fp32 AdamW moments, per parameter, for a retained checkpoint pair.
CHECKPOINT_BYTES_PER_PARAM: Final = 2 + 8
MICROBATCH_ONE_ASSUMPTION: Final = (
    "priced from a microbatch-1 gradient step, so the row is a conservative upper bound on the "
    "GPU-hours a batched run would use"
)
CONTENDED_ASSUMPTION: Final = (
    "the underlying rates were measured while the job's other seven ranks were resident on the same "
    "node, so they are not solo-GPU peak"
)


class ArmForecast(BaseModel):
    """One arm's priced cost for one token budget, or the reason it could not be priced."""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True, strict=True)

    arm: str
    label: str
    source_arm: str
    source_kind: str = Field(pattern="^(measured|derived)$")
    forecastable: bool
    reason_unforecastable: str | None = None
    measured_canvas: int | None = None
    contributing_ranks: int = Field(ge=0)
    step_seconds: float | None = Field(default=None, gt=0)
    tokens_per_second: float | None = Field(default=None, gt=0)
    peak_allocated_bytes: int | None = Field(default=None, gt=0)
    token_budget: int | None = Field(default=None, gt=0)
    gpu_hours_per_seed: float | None = Field(default=None, gt=0)
    gpu_hours_all_seeds: float | None = Field(default=None, gt=0)
    assumptions: tuple[str, ...] = ()


class StorageLine(BaseModel):
    """Retained bytes per run, and the disk the plan is actually writing into."""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True, strict=True)

    parameters_numel: int = Field(gt=0)
    retained_checkpoints_per_seed: int = Field(gt=0)
    bytes_per_checkpoint: int = Field(gt=0)
    bytes_per_arm_seed: int = Field(gt=0)
    arms_in_matrix: int = Field(gt=0)
    total_bytes: int = Field(gt=0)
    retention_policy: str = Field(min_length=1)
    note: str = Field(min_length=1)


class LedgerDocument(BaseModel):
    """The published ledger: inputs, per-arm forecasts, storage, and what is deliberately missing."""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int = 1
    scope: str = Field(min_length=1)
    calibration_manifest_sha256: str = Field(min_length=64, max_length=64)
    calibration_status: str
    gradient_checkpointing: bool
    candidate_token_budgets: tuple[int, ...]
    derived_arms: dict[str, str]
    arms: tuple[ArmForecast, ...]
    storage: StorageLine
    confirmation_note: str = Field(min_length=1)
    unforecastable: tuple[str, ...] = ()
    unforecast_reasons: dict[str, str] = Field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    claims_not_made: tuple[str, ...] = ()


def training_step_seconds(canvas: CanvasAggregate) -> float | None:
    """One training step's time at the training canvas, from the measured components.

    A step is a forward, a backward and an optimizer step. The forward is only recorded as a rate,
    so it is inverted here; the other two are durations already.
    """
    if canvas.backward_seconds is None:
        return None
    forward_seconds = canvas.canvas_tokens / canvas.forward_tokens_per_second
    optimizer_seconds = canvas.optimizer_step_seconds or 0.0
    return forward_seconds + canvas.backward_seconds + optimizer_seconds


def _rung(arm_aggregate, canvas: int) -> CanvasAggregate | None:
    for measured in arm_aggregate.canvases:
        if measured.canvas_tokens == canvas:
            return measured
    return None


def forecast_arm(
    arm: str,
    aggregate: CalibrationAggregate,
    *,
    token_budget: int,
    seeds: int,
) -> ArmForecast:
    """Price one controlled arm at one token budget, or explain why it cannot be priced."""
    source_kind = "derived" if arm in DERIVATIONS else "measured"
    source_arm = DERIVATIONS.get(arm, ARM_SOURCE.get(arm, ""))
    label = ARM_LABEL[arm]
    by_id = {measured_arm.arm_id: measured_arm for measured_arm in aggregate.arms}
    arm_aggregate = by_id.get(source_arm)
    base = ArmForecast(
        arm=arm, label=label, source_arm=source_arm, source_kind=source_kind,
        forecastable=False, contributing_ranks=arm_aggregate.contributing_ranks if arm_aggregate else 0,
    )
    if arm_aggregate is None or arm_aggregate.contributing_ranks == 0:
        return base.model_copy(update={
            "reason_unforecastable": f"arm {source_arm} produced no measurement in this job",
        })
    measured = _rung(arm_aggregate, TRAINING_CANVAS)
    if measured is None:
        rungs = [row.canvas_tokens for row in arm_aggregate.canvases]
        return base.model_copy(update={
            "reason_unforecastable": (
                f"the training canvas {TRAINING_CANVAS} was not measured for {source_arm}; "
                f"measured rungs were {rungs}; a smaller canvas is not a substitute because "
                "per-token cost falls with canvas length"
            ),
        })
    step_seconds = training_step_seconds(measured)
    if step_seconds is None or step_seconds <= 0:
        return base.model_copy(update={"reason_unforecastable": "no backward duration was recorded"})
    tokens_per_second = TRAINING_CANVAS / step_seconds
    gpu_hours_per_seed = token_budget / tokens_per_second / 3600.0
    assumptions = [MICROBATCH_ONE_ASSUMPTION, CONTENDED_ASSUMPTION]
    if source_kind == "derived":
        assumptions.append(f"derived from {source_arm}: this arm was not run in the calibration job")
    if arm == "A4":
        assumptions.append(
            "A4 has extra untied blocks that do not exist in the code, so its parameter count is "
            "higher than the measured arm's and this row understates its cost"
        )
    return base.model_copy(update={
        "forecastable": True,
        "measured_canvas": measured.canvas_tokens,
        "step_seconds": step_seconds,
        "tokens_per_second": tokens_per_second,
        "peak_allocated_bytes": measured.peak_allocated_bytes,
        "token_budget": token_budget,
        "gpu_hours_per_seed": gpu_hours_per_seed,
        "gpu_hours_all_seeds": gpu_hours_per_seed * seeds,
        "assumptions": tuple(assumptions),
    })


def storage_line(aggregate: CalibrationAggregate, *, arms_in_matrix: int, retained: int = 2) -> StorageLine:
    """Retained bytes for the small-arm matrix, using the measured parameter count."""
    numel = 0
    for measured_arm in aggregate.arms:
        if measured_arm.parameters is not None and measured_arm.arm_id in SMALL_SOURCE_ARMS:
            numel = max(numel, measured_arm.parameters.total_numel)
    if numel == 0:
        numel = 1
    per_checkpoint = numel * CHECKPOINT_BYTES_PER_PARAM
    per_arm_seed = per_checkpoint * retained
    return StorageLine(
        parameters_numel=numel,
        retained_checkpoints_per_seed=retained,
        bytes_per_checkpoint=per_checkpoint,
        bytes_per_arm_seed=per_arm_seed,
        arms_in_matrix=arms_in_matrix,
        total_bytes=per_arm_seed * arms_in_matrix * len(TRAINING_SEEDS),
        retention_policy="keep-last plus keep-best per arm-seed, with model-only neutral copies",
        note=(
            "SAVE_ROOT is on the global_user volume, which was 98% full with ~240 GiB free when this "
            "ledger was written, and qz workers do not mount the project volume. Fix the retention "
            "policy before launching: a mid-run prune would destroy the failed-attempt evidence Task 7 "
            "must preserve, and it has happened before in this program."
        ),
    )


def confirmation_note(aggregate: CalibrationAggregate) -> str:
    """Describe the 2.9B gap from what the calibration actually recorded.

    Written from the aggregate, not as a fixed sentence: an earlier version asserted that both large
    arms stopped at 512 and OOM'd at 4096, and the aggregate contradicted it — one of them had
    measured 4096. A note that states calibration facts has to be computed from them.
    """
    large = [arm for arm in aggregate.arms if arm.arm_id in ("A2_large_noloop", "A3_large_loop")]
    reached = [arm for arm in large if any(row.canvas_tokens == TRAINING_CANVAS for row in arm.canvases)]
    missed = [arm for arm in large if arm not in reached]
    parts = [
        "The 2.9B confirmation runs are NOT covered by this ledger and are not extrapolated from it.",
    ]
    if reached:
        parts.append(
            "Reached the "
            f"{TRAINING_CANVAS}-token canvas: {', '.join(arm.arm_id for arm in reached)}. "
            "A 2.9B confirmation forecast could be built from that row, but this ledger prices only "
            "the controlled small-model matrix."
        )
    if not large:
        parts.append(
            "This calibration measured no large arm at all, so nothing here speaks to the 2.9B "
            "confirmation runs; a sharded calibration is required."
        )
    if missed:
        details = []
        for arm in missed:
            rungs = [row.canvas_tokens for row in arm.canvases]
            details.append(f"{arm.arm_id} (measured rungs {rungs})")
        parts.append(
            "Did not reach it: "
            + "; ".join(details)
            + ". Those arms stopped where a single 80 GiB GPU could no longer hold the step, so a "
            "sharded calibration is required rather than an extrapolation."
        )
    return " ".join(parts)


def build_ledger(
    aggregate: CalibrationAggregate,
    *,
    calibration_manifest_sha256: str,
    token_budget: int,
) -> LedgerDocument:
    """Assemble the ledger for one candidate common token budget."""
    arms = tuple(
        forecast_arm(arm, aggregate, token_budget=token_budget, seeds=len(TRAINING_SEEDS))
        for arm in CONTROLLED_ARMS
    )
    unforecastable = tuple(row.arm for row in arms if not row.forecastable)
    reasons = {row.arm: row.reason_unforecastable or "" for row in arms if not row.forecastable}
    return LedgerDocument(
        scope="Task-6 experiment forecast at the training canvas; not a measurement and not a performance claim",
        calibration_manifest_sha256=calibration_manifest_sha256,
        calibration_status=aggregate.status,
        gradient_checkpointing=aggregate.gradient_checkpointing,
        candidate_token_budgets=CANDIDATE_TOKEN_BUDGETS,
        derived_arms=dict(DERIVATIONS),
        arms=arms,
        storage=storage_line(aggregate, arms_in_matrix=len(CONTROLLED_ARMS)),
        confirmation_note=confirmation_note(aggregate),
        unforecastable=unforecastable,
        unforecast_reasons=reasons,
        assumptions=(
            "per-step cost is the sum of a forward, a backward and an optimizer step measured at the "
            "training canvas",
            "a run's GPU-hours are (token budget / measured tokens per second) / 3600",
            "no multi-GPU scaling factor is applied; the ledger is per-GPU",
            "no communication, checkpointing pause, evaluation, or restart cost is included",
        ),
        claims_not_made=(
            "sustainable goodput", "long-context capability", "quality",
            "physical edge-device performance", "measured training time at production sharding",
        ),
    )
