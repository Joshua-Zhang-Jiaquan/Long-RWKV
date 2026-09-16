"""Task 5: the preregistration record and its QA contract.

The plan's Task 5 acceptance is that "every comparison has an interpretable
control and matching rule; no future-looking AR baseline; practical thresholds
and non-inferiority bounds are frozen before confirmation". These tests hold the
record to that, and hold it to the property that makes a preregistration worth
anything: it cannot be sealed while a blocker stands, and it cannot be edited
after it is written.
"""

from __future__ import annotations

import json

import pytest

from scale.experiments.nonlatent_iclr import preregistration as pr


def _document(**kwargs) -> dict[str, object]:
    return pr.build_preregistration(**kwargs)


# --------------------------------------------------------------------------
# the record's contents
# --------------------------------------------------------------------------


def test_arms_are_exactly_a0_to_a5_and_each_names_a_control() -> None:
    doc = _document()
    assert [a["key"] for a in doc["arms"]] == ["A0", "A1", "A2", "A3", "A4", "A5"]
    for arm in doc["arms"]:
        assert arm["control_for"], f"{arm['key']} names no control"
        for field in ("direction", "objective", "depth"):
            assert arm[field], f"{arm['key']} leaves {field} unpinned"


def test_the_candidate_is_isolated_by_the_right_controls() -> None:
    """A3 is the candidate: A2 isolates looping, A4 isolates parameter count."""
    by_key = {a["key"]: a for a in _document()["arms"]}
    assert by_key["A2"]["depth"] == "no loop" and by_key["A2"]["direction"] == "bidirectional"
    assert by_key["A3"]["depth"] == "tied loop"
    assert by_key["A4"]["parameters_disclosed"] is True, (
        "the unequal-parameter control must disclose that it is unequal")
    assert by_key["A5"]["direction"] == "forward-only" and by_key["A5"]["depth"] == "tied loop"


def test_seeds_and_endpoints_are_frozen() -> None:
    doc = _document()
    assert tuple(doc["training_seeds"]) == (17, 29, 43)
    assert tuple(doc["synthetic_data_seeds"]) == (101, 102, 103, 104, 105)
    names = {e["endpoint"] for e in doc["primary_endpoints"]}
    assert names == {"depth_recycling_quality", "long_context_macro", "serving_goodput"}
    for endpoint in doc["primary_endpoints"] + doc["secondary_endpoints"]:
        assert endpoint["threshold"], endpoint["endpoint"]
        assert endpoint["failure_reading"], (
            f"{endpoint['endpoint']} states no failure reading; a target without one "
            f"cannot produce an honest negative")


def test_statistics_resample_clusters_not_tokens() -> None:
    stats = _document()["statistics"]
    assert "cluster" in stats["resampling_unit"]
    assert "hierarchical" in stats["bootstrap"]
    assert "Holm" in stats["correction"]
    assert stats["quality_non_inferiority"]


def test_stopping_rules_separate_sustainable_from_saturation() -> None:
    rules = _document()["stopping_rules"]
    assert "sustainable" in rules and "saturation" in rules
    assert "never a proven peak" in rules["saturation"]
    assert "underpowered" in rules


def test_released_and_controlled_results_are_separate_tables() -> None:
    assert "separate tables" in _document()["test_access"]["result_tables"]


# --------------------------------------------------------------------------
# sealing: the property that makes the record worth anything
# --------------------------------------------------------------------------


def test_a_draft_is_not_a_preregistration() -> None:
    doc = _document()
    assert doc["seal_state"] == "DRAFT_UNSEALED"
    with pytest.raises(pr.PreregistrationRefusal, match="governs nothing"):
        pr.require_sealed(doc)


def test_the_resolved_blocker_is_kept_not_deleted() -> None:
    """A blocker that vanishes without a record is indistinguishable from one
    that was edited away."""
    blockers = {b["id"]: b for b in _document()["sealing_blockers"]}
    assert pr.BLOCKER_ID in blockers
    resolved = blockers[pr.BLOCKER_ID]
    assert resolved["state"] == "RESOLVED"
    assert resolved["resolution"]["correct_convention"]
    assert resolved["resolution"]["evidence"]


def test_open_blockers_are_the_three_that_still_need_external_input() -> None:
    doc = _document()
    assert set(doc["open_blockers"]) == {
        "SLO_DEADLINES", "COMMON_TOKEN_BUDGET", "EXTERNAL_CHECKPOINTS"}


def test_sealing_requires_real_inputs_not_a_flag() -> None:
    """seal_state is derived, so a caller cannot declare the record sealed."""
    assert _document()["seal_state"] == "DRAFT_UNSEALED"
    # supplying the inputs closes the blockers
    sealed = _document(slo_deadline_ms=1000, common_token_budget=262_144,
                       checkpoints=({"version": "v1", "identity": "a" * 64},))
    assert sealed["seal_state"] == "SEALED"
    assert sealed["open_blockers"] == []
    pr.require_sealed(sealed)          # no refusal


def test_a_checkpoint_without_an_identity_does_not_close_the_blocker() -> None:
    doc = _document(checkpoints=({"version": "v1"},))
    assert "EXTERNAL_CHECKPOINTS" in doc["open_blockers"]


# --------------------------------------------------------------------------
# QA contract
# --------------------------------------------------------------------------


def test_happy_case_passes() -> None:
    assert pr.verify_preregistration(_document(), case="happy") == "PREREGISTRATION_HAPPY"


def test_failure_case_confirms_the_planted_violations() -> None:
    assert pr.verify_preregistration(_document(), case="failure") == \
        "EXPECTED_FAILURE_CONFIRMED"


def test_failure_case_refuses_a_sealed_record_on_task_selection() -> None:
    """A task-selection change after unsealing invalidates confirmation."""
    sealed = _document(slo_deadline_ms=1000, common_token_budget=262_144,
                       checkpoints=({"version": "v1", "identity": "a" * 64},))
    assert sealed["seal_state"] == "SEALED"
    with pytest.raises(pr.PreregistrationRefusal, match="invalidated"):
        pr.verify_preregistration(sealed, case="failure")


def test_unknown_case_refuses() -> None:
    with pytest.raises(pr.PreregistrationRefusal, match="expected happy or failure"):
        pr.verify_preregistration(_document(), case="maybe")


# --------------------------------------------------------------------------
# the matching rule
# --------------------------------------------------------------------------


def test_readouts_are_not_interchangeable() -> None:
    """Matched-token and matched-compute answer different questions."""
    doc = _document()
    _ = pr.require_comparable(doc, readout="matched_measured_compute")
    _ = pr.require_comparable(doc, readout="matched_nonpadding_tokens")
    with pytest.raises(pr.PreregistrationRefusal, match="published separately"):
        pr.require_comparable(doc, readout="average_accuracy")


def test_a_non_ar_control_arm_would_be_refused() -> None:
    """No future-looking AR baseline: A0 is the only autoregressive arm."""
    doc = _document()
    doc["arms"] = doc["arms"] + [dict(doc["arms"][0], key="A6")]
    with pytest.raises(pr.PreregistrationRefusal, match="A0 only"):
        pr.require_comparable(doc, readout="matched_measured_compute")


# --------------------------------------------------------------------------
# on-disk behaviour
# --------------------------------------------------------------------------


def test_write_once_then_read_round_trips(tmp_path) -> None:
    doc = _document()
    path = tmp_path / "preregistration.json"
    pr.write_preregistration(doc, path)
    assert pr.read_preregistration(path)["seal_state"] == "DRAFT_UNSEALED"


def test_a_second_write_refuses(tmp_path) -> None:
    path = tmp_path / "preregistration.json"
    pr.write_preregistration(_document(), path)
    with pytest.raises(pr.PreregistrationRefusal, match="written once"):
        pr.write_preregistration(_document(), path)


def test_an_edited_record_is_detected(tmp_path) -> None:
    """The digest is what makes 'frozen before confirmation' checkable."""
    path = tmp_path / "preregistration.json"
    pr.write_preregistration(_document(), path)
    document = json.loads(path.read_text())
    document["training_seeds"] = [17]          # quietly drop two seeds
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    with pytest.raises(pr.PreregistrationRefusal, match="edited since"):
        pr.read_preregistration(path)


def test_a_missing_or_wrong_schema_record_refuses(tmp_path) -> None:
    with pytest.raises(pr.PreregistrationRefusal, match="no preregistration"):
        pr.read_preregistration(tmp_path / "absent.json")
    wrong = tmp_path / "other.json"
    wrong.write_text(json.dumps({"schema": "something_else"}))
    with pytest.raises(pr.PreregistrationRefusal, match="does not declare"):
        pr.read_preregistration(wrong)
