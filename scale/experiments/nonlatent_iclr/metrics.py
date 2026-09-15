from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class MetricName(StrEnum):
    MAX_RUN_FRAC = "max_run_frac"
    MASKED_TOKEN_ACCURACY = "masked_token_accuracy"
    WHOLE_SEQUENCE_EXACT_MATCH = "whole_sequence_exact_match"
    TAU_A_COMMIT_ORDER = "tau_a_commit_order"
    DISTINCT_FRAC = "distinct_frac"
    RESIDUE = "residue"


class MetricDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    DIAGNOSTIC = "diagnostic"


METRIC_DIRECTIONS: Final = {
    MetricName.MAX_RUN_FRAC: MetricDirection.LOWER_IS_BETTER,
    MetricName.MASKED_TOKEN_ACCURACY: MetricDirection.HIGHER_IS_BETTER,
    MetricName.WHOLE_SEQUENCE_EXACT_MATCH: MetricDirection.HIGHER_IS_BETTER,
    MetricName.TAU_A_COMMIT_ORDER: MetricDirection.DIAGNOSTIC,
    MetricName.DISTINCT_FRAC: MetricDirection.DIAGNOSTIC,
    MetricName.RESIDUE: MetricDirection.DIAGNOSTIC,
}


@dataclass(frozen=True, slots=True)
class MetricInputError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    max_run_frac: float
    masked_token_accuracy: float | None
    whole_sequence_exact_match: float | None


@dataclass(frozen=True, slots=True)
class Pair:
    document_id: str
    seed: int
    arm: str
    value: float


@dataclass(frozen=True, slots=True)
class PairedSummary:
    metric: MetricName
    direction: MetricDirection
    mean_delta: float
    pair_count: int
    interpretation: str


@dataclass(frozen=True, slots=True)
class CorruptionResult:
    corrupted: tuple[int, ...]
    condition_mask: tuple[bool, ...]
    prediction_mask: tuple[bool, ...]


def sequence_metrics(
    target: tuple[int, ...], prediction: tuple[int, ...], masked: tuple[bool, ...], *, pad_id: int
) -> SequenceMetrics:
    if len(target) != len(prediction) or len(target) != len(masked):
        raise MetricInputError("target, prediction, and masked positions must have equal length")
    active = tuple(index for index, token in enumerate(target) if token != pad_id)
    if not active:
        raise MetricInputError("sequence has no non-padding positions")
    if any((target[index] == pad_id) != (prediction[index] == pad_id) for index in range(len(target))):
        raise MetricInputError("target and prediction padding positions differ")
    run = _max_run(tuple(prediction[index] for index in active)) / len(active)
    masked_active = tuple(index for index in active if masked[index])
    accuracy = None if not masked_active else sum(
        prediction[index] == target[index] for index in masked_active
    ) / len(masked_active)
    exact = float(all(prediction[index] == target[index] for index in active))
    return SequenceMetrics(run, accuracy, exact)


def paired_delta(metric: MetricName, loop: tuple[Pair, ...], control: tuple[Pair, ...]) -> PairedSummary:
    loop_values = _unique_values(loop, "loop")
    control_values = _unique_values(control, "control")
    if loop_values.keys() != control_values.keys():
        raise MetricInputError("paired key sets differ; coverage loss must be reported")
    deltas = tuple(loop_values[key] - control_values[key] for key in sorted(loop_values))
    if not deltas:
        raise MetricInputError("paired comparison has no observations")
    mean = sum(deltas) / len(deltas)
    direction = METRIC_DIRECTIONS[metric]
    interpretation = _interpret(direction, mean)
    return PairedSummary(metric, direction, mean, len(deltas), interpretation)


def prompt_preserving_corruption(
    tokens: tuple[int, ...], condition: tuple[bool, ...], prediction: tuple[bool, ...], *, pad_id: int,
    mask_token_id: int,
) -> CorruptionResult:
    if len(tokens) != len(condition) or len(tokens) != len(prediction):
        raise MetricInputError("tokens and masks must have equal length")
    corrupted: list[int] = []
    for token, visible, requested in zip(tokens, condition, prediction, strict=True):
        if token == pad_id:
            if visible or requested:
                raise MetricInputError("padding cannot be conditioned or predicted")
            corrupted.append(token)
            continue
        if visible and requested:
            raise MetricInputError("known future answer cannot enter condition")
        if not visible and not requested:
            raise MetricInputError("every active token must be condition or prediction")
        corrupted.append(mask_token_id if requested else token)
    return CorruptionResult(tuple(corrupted), condition, prediction)


def _max_run(tokens: tuple[int, ...]) -> int:
    longest = 1
    current = 1
    for previous, token in zip(tokens, tokens[1:]):
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
    return longest


def _unique_values(pairs: tuple[Pair, ...], source: str) -> dict[tuple[str, int, str], float]:
    values: dict[tuple[str, int, str], float] = {}
    for pair in pairs:
        key = (pair.document_id, pair.seed, pair.arm)
        if key in values:
            raise MetricInputError(f"duplicate pair key in {source}: {key}")
        if not math.isfinite(pair.value):
            raise MetricInputError(f"non-finite value in {source}: {key}")
        values[key] = pair.value
    return values


def _interpret(direction: MetricDirection, delta: float) -> str:
    match direction:
        case MetricDirection.HIGHER_IS_BETTER:
            return "better" if delta > 0 else "worse" if delta < 0 else "equal"
        case MetricDirection.LOWER_IS_BETTER:
            return "worse" if delta > 0 else "better" if delta < 0 else "equal"
        case MetricDirection.DIAGNOSTIC:
            return "diagnostic_only"
        case unreachable:
            from typing import assert_never
            assert_never(unreachable)


def tau_a_commit_order(commit_steps: tuple[int, ...], masked: tuple[bool, ...]) -> float | None:
    if len(commit_steps) != len(masked):
        raise MetricInputError("commit steps and mask must have equal length")
    values = tuple(value for value, selected in zip(commit_steps, masked, strict=True) if selected)
    if len(values) < 2:
        return None
    concordant = 0
    discordant = 0
    for left_index, left in enumerate(values):
        for right in values[left_index + 1:]:
            if right > left:
                concordant += 1
            elif right < left:
                discordant += 1
    total_pairs = len(values) * (len(values) - 1) / 2
    return (concordant - discordant) / total_pairs
