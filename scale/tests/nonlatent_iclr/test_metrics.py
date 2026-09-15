from __future__ import annotations

import math

import pytest

from scale.experiments.nonlatent_iclr.metrics import (
    MetricInputError,
    MetricName,
    Pair,
    paired_delta,
    sequence_metrics,
    tau_a_commit_order,
)
from scale.experiments.nonlatent_iclr.metrics_task import failure_probe_task_two


def test_sequence_metrics_distinguish_masked_accuracy_from_exact_match() -> None:
    # Given: one wrong unmasked token and one wrong predicted masked token.
    metrics = sequence_metrics(
        target=(4, 5, 6, 0), prediction=(9, 5, 7, 0), masked=(False, True, True, False), pad_id=0
    )
    # When: metrics are calculated on the explicit prediction mask.
    # Then: masked accuracy and whole-sequence exact match retain distinct values.
    assert metrics.masked_token_accuracy == 0.5
    assert metrics.whole_sequence_exact_match == 0.0
    assert metrics.max_run_frac == 1 / 3


def test_sequence_metrics_returns_undefined_without_masked_positions() -> None:
    # Given: a non-padding sequence with no prediction positions.
    metrics = sequence_metrics(target=(1, 2, 0), prediction=(1, 2, 0), masked=(False, False, False), pad_id=0)
    # When: masked-token accuracy is requested.
    # Then: it is explicitly undefined rather than silently scored as perfect.
    assert metrics.masked_token_accuracy is None
    assert metrics.whole_sequence_exact_match == 1.0


def test_paired_delta_rejects_duplicate_and_mismatched_pair_sets() -> None:
    # Given: duplicate keys and a missing control key.
    duplicates = (Pair("doc-1", 42, "r100_s32", 0.1), Pair("doc-1", 42, "r100_s32", 0.2))
    missing = (Pair("doc-2", 42, "r100_s32", 0.1),)
    # When: paired comparisons are built.
    # Then: neither source can be silently dropped.
    with pytest.raises(MetricInputError):
        paired_delta(MetricName.MAX_RUN_FRAC, duplicates, missing)


def test_max_run_positive_delta_is_worse() -> None:
    # Given: loop has a longer repeated run than its paired control.
    loop = (Pair("doc-1", 42, "r100_s32", 0.75),)
    control = (Pair("doc-1", 42, "r100_s32", 0.25),)
    # When: the direction-aware paired summary is calculated.
    # Then: a positive loop-minus-control delta is worse for max-run fraction.
    summary = paired_delta(MetricName.MAX_RUN_FRAC, loop, control)
    assert summary.mean_delta == 0.5
    assert summary.interpretation == "worse"


def test_nonfinite_pair_is_rejected_without_verdict() -> None:
    # Given: a NaN raw observation.
    loop = (Pair("doc-1", 42, "r100_s32", math.nan),)
    control = (Pair("doc-1", 42, "r100_s32", 0.25),)
    # When: it enters a comparison.
    # Then: validation rejects it before a positive or negative conclusion exists.
    with pytest.raises(MetricInputError):
        paired_delta(MetricName.MASKED_TOKEN_ACCURACY, loop, control)


def test_repetition_uses_prediction_and_tau_a_handles_ties() -> None:
    # Given: distinct targets, repeated predictions, and a tied commit order.
    metrics = sequence_metrics((1, 2, 3, 4, 0), (9, 9, 9, 4, 0), (True, True, True, True, False), pad_id=0)
    # When: the decoded metrics are evaluated.
    # Then: repetition belongs to the prediction and tau-a is diagnostic 2/3.
    assert metrics.max_run_frac == 0.75
    assert tau_a_commit_order((1, 1, 2), (True, True, True)) == 2 / 3


def test_failure_probe_executes_finite_max_run_polarity_control() -> None:
    # Given: task-two's production failure probe.
    # When: its finite matched max-run control is evaluated.
    # Then: the output includes the actual positive delta and lower-is-better verdict.
    result = failure_probe_task_two()
    assert result.status == "EXPECTED_FAILURE_CONFIRMED"
    assert "delta=0.8" in result.detail
    assert "worse" in result.detail
