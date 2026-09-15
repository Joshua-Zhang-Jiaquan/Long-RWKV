from __future__ import annotations

from json import dumps
from typing import Literal, assert_never

import pytest
import torch
from pydantic import ValidationError

from scale.experiments.nonlatent_iclr.full_canvas import CanvasBoundaryError
from scale.experiments.nonlatent_iclr.qualification.lifecycle_checks import LifecycleAbort, LifecycleChecks
from scale.experiments.nonlatent_iclr.qualification.lifecycle_evidence import (
    LifecycleConfig, LifecycleEvidence, Scenario,
)
from scale.experiments.nonlatent_iclr.qualification.lifecycle import probe_lifecycle
from scale.tests.nonlatent_iclr.test_qualification_lifecycle import Fault, RecordingModel, Synchronizer


@pytest.mark.parametrize("corruption", ["empty", "reordered", "comparison"])
def test_evidence_cannot_pass_when_scenario_observations_are_missing_or_contradictory(
    corruption: Literal["empty", "reordered", "comparison"],
) -> None:
    # Given: a genuine CPU result altered at the serialized evidence boundary.
    model = RecordingModel()
    result = probe_lifecycle(model, LifecycleConfig(vocab_size=8), Synchronizer(model))
    payload = result.model_dump()
    match corruption:
        case "empty":
            payload["steps"] = ()
        case "reordered":
            payload["steps"] = tuple(reversed(payload["steps"]))
        case "comparison":
            payload["steps"][0]["all_finite"] = False
        case unreachable:
            assert_never(unreachable)
    # When: the standalone strict evidence type parses the altered observation.
    parsed = LifecycleEvidence.model_validate(payload)
    # Then: a completed flag or matching code alone cannot manufacture success.
    assert not parsed.passed


@pytest.mark.parametrize("reference", [
    torch.full((1, 48, 8), float("inf")),
    torch.zeros(1, 47, 8),
    torch.zeros(1, 48, 8, dtype=torch.float64),
])
def test_comparison_rejects_when_reference_is_invalid(reference: torch.Tensor) -> None:
    # Given: independently invalid reference finiteness, shape or dtype.
    model = RecordingModel()
    checks = LifecycleChecks(model, Synchronizer(model))
    # When: exact comparison inspects a real pair of CPU tensors.
    with pytest.raises(LifecycleAbort):
        checks.compare((torch.zeros(1, 48, 8), reference), LifecycleConfig(vocab_size=8))
    # Then: failure persists in typed observations, with no model calls.
    assert not checks.steps[0].passed
    assert checks.calls == 0


def test_rejection_fails_when_expected_exception_follows_a_model_call() -> None:
    # Given: an operation that improperly reaches the model before rejecting.
    model = RecordingModel()
    checks = LifecycleChecks(model, Synchronizer(model))

    def late_rejection() -> torch.Tensor:
        checks.direct(torch.zeros(1, 48, dtype=torch.int64))
        raise CanvasBoundaryError("stale_canvas")

    # When: a correct error code is paired with a nonzero model-call delta.
    with torch.inference_mode(), pytest.raises(LifecycleAbort, match="rejection_failed"):
        checks.reject(late_rejection, "stale_canvas")
    # Then: the code cannot hide the prohibited invocation.
    assert checks.calls == 1
    assert not checks.steps[0].passed


@pytest.mark.parametrize("vocabulary", [True, "8", 3, 65_537])
def test_config_rejects_when_vocabulary_is_untyped_or_out_of_bounds(vocabulary: int | str) -> None:
    # Given: untrusted configuration at the callable boundary.
    payload = {"vocab_size": vocabulary}
    # When: strict parsing applies the bounded probe contract.
    with pytest.raises(ValidationError):
        LifecycleConfig.model_validate(payload)
    # Then: validation ends before a model or device is required.


@pytest.fixture
def successful_evidence() -> LifecycleEvidence:
    model = RecordingModel()
    return probe_lifecycle(model, LifecycleConfig(vocab_size=8), Synchronizer(model))


@pytest.mark.parametrize("scenario", [
    "reference_a", "repeat_a", "caller_mutation", "edit_reference", "stale_revision",
    "closed_request", "reference_b", "old_session", "foreign_session", "a_b_a",
    "reset_retirement", "reset_reopen", "reset_old_session", "injected_forward",
    "caller_failure", "failure_retirement", "failure_reopen", "failure_old_session",
])
def test_evidence_rejects_success_when_scenario_kind_is_substituted(
    successful_evidence: LifecycleEvidence, scenario: Scenario,
) -> None:
    # Given: a record of another kind with the original scenario and call window.
    payload = successful_evidence.model_dump()
    index = next(i for i, step in enumerate(successful_evidence.steps) if step.scenario == scenario)
    original = successful_evidence.steps[index]
    match original.kind:
        case "comparison":
            replacement = next(step for step in successful_evidence.steps if step.kind == "injection").model_dump()
            replacement.update(synchronizations_before=original.calls_before, synchronizations_after=original.calls_after)
        case "rejection" | "injection" | "execution_error":
            replacement = next(step for step in successful_evidence.steps if step.kind == "comparison").model_dump()
        case unreachable:
            assert_never(unreachable)
    replacement.update(scenario=scenario, calls_before=original.calls_before, calls_after=original.calls_after)
    steps = list(payload["steps"])
    steps[index] = replacement
    payload["steps"] = tuple(steps)
    # When: the contradictory observations cross a real JSON serialize/reparse boundary.
    parsed = LifecycleEvidence.model_validate_json(dumps(payload))
    # Then: preserving windows cannot substitute for the required evidence kind.
    assert not parsed.passed


@pytest.mark.parametrize("scenario", [
    "stale_revision", "closed_request", "old_session", "foreign_session",
    "reset_retirement", "reset_old_session", "failure_retirement", "failure_old_session",
])
def test_evidence_rejects_success_when_rejection_codes_agree_but_are_wrong(
    successful_evidence: LifecycleEvidence, scenario: Scenario,
) -> None:
    # Given: expected and observed agree on an unrelated rejection code.
    payload = successful_evidence.model_dump()
    index = next(i for i, step in enumerate(successful_evidence.steps) if step.scenario == scenario)
    payload["steps"][index].update(expected_code="use_cache_forbidden", observed_code="use_cache_forbidden")
    # When: the changed observations are serialized and reparsed.
    parsed = LifecycleEvidence.model_validate_json(dumps(payload))
    # Then: the scenario's required code, not reported self-agreement, governs success.
    assert not parsed.passed


def test_evidence_rejects_success_when_top_level_vocabulary_changes(
    successful_evidence: LifecycleEvidence,
) -> None:
    # Given: all comparisons describe vocabulary8 but the report declares vocabulary7.
    payload = successful_evidence.model_dump()
    payload["vocab_size"] = 7
    # When: the contradictory report is serialized and reparsed.
    parsed = LifecycleEvidence.model_validate_json(dumps(payload))
    # Then: report geometry must bind every comparison.
    assert not parsed.passed


@pytest.mark.parametrize("scenario", [
    "reference_a", "repeat_a", "caller_mutation", "edit_reference", "reference_b",
    "a_b_a", "reset_reopen", "injected_forward", "failure_reopen",
])
def test_evidence_rejects_success_when_one_comparison_uses_foreign_geometry(
    successful_evidence: LifecycleEvidence, scenario: Scenario,
) -> None:
    # Given: one internally consistent comparison has a different vocabulary shape.
    payload = successful_evidence.model_dump()
    index = next(i for i, step in enumerate(successful_evidence.steps) if step.scenario == scenario)
    payload["steps"][index].update(expected_shape=(1, 48, 7), actual_shape=(1, 48, 7), reference_shape=(1, 48, 7))
    # When: the report crosses the JSON boundary.
    parsed = LifecycleEvidence.model_validate_json(dumps(payload))
    # Then: even non-first comparisons must match the declared report geometry.
    assert not parsed.passed


@pytest.mark.parametrize("counters", [(0, 1), (9, 10), (11, 12), (12, 13), (100, 101), (10, 12), (11, 11)])
def test_evidence_rejects_success_when_injection_synchronization_is_contradictory(
    successful_evidence: LifecycleEvidence, counters: tuple[int, int],
) -> None:
    # Given: synchronization counters are shifted, out of total bounds, or not one call.
    payload = successful_evidence.model_dump()
    index = next(i for i, step in enumerate(successful_evidence.steps) if step.kind == "injection")
    payload["steps"][index].update(synchronizations_before=counters[0], synchronizations_after=counters[1])
    # When: JSON serialization preserves those observations for the parser.
    parsed = LifecycleEvidence.model_validate_json(dumps(payload))
    # Then: delta-one alone cannot prove synchronized injection at calls10→11.
    assert not parsed.passed


@pytest.mark.parametrize("totals", [(11, 12), (12, 11), (13, 13)])
def test_evidence_rejects_success_when_completed_totals_are_contradictory(
    successful_evidence: LifecycleEvidence, totals: tuple[int, int],
) -> None:
    # Given: valid step windows but incorrect successful-run total counts.
    payload = successful_evidence.model_dump()
    payload.update(model_calls=totals[0], synchronizations=totals[1])
    # When: the aggregate observations are serialized and reparsed.
    parsed = LifecycleEvidence.model_validate_json(dumps(payload))
    # Then: the complete sequence requires twelve forwards and twelve synchronizations.
    assert not parsed.passed


def test_evidence_passes_when_genuine_observations_roundtrip(successful_evidence: LifecycleEvidence) -> None:
    # Given: the real adapter probe's deterministic CPU observations.
    serialized = successful_evidence.model_dump_json()
    # When: a future consumer reparses the standalone report.
    parsed = LifecycleEvidence.model_validate_json(serialized)
    # Then: exactly the prescribed observation mix passes without promotion.
    assert parsed.passed and parsed.runtime_promoted is False
    assert tuple(sum(step.kind == kind for step in parsed.steps)
                 for kind in ("comparison", "rejection", "injection")) == (9, 8, 1)


@pytest.mark.parametrize("fault", ["drift", "nan", "shape", "runtime"])
def test_evidence_preserves_partial_failure_when_reparsed(fault: Fault) -> None:
    # Given: an actual bounded CPU probe that stops on a forward failure.
    model = RecordingModel(fault)
    result = probe_lifecycle(model, LifecycleConfig(vocab_size=8), Synchronizer(model))
    # When: the partial observations are serialized and reparsed.
    parsed = LifecycleEvidence.model_validate_json(result.model_dump_json())
    # Then: failure remains representable and nonpassing, not rejected as incomplete JSON.
    assert not parsed.passed and not parsed.completed
    assert parsed.failure_code == result.failure_code
    assert parsed.model_calls == 2
