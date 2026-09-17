"""Scoring tests: a harness failure must never be scored as a model failure.

The whole reason ``scoring.py`` exists is that "the completion could not be
parsed" and "the model answered wrong" are different facts, and a scorer that
conflates them reports the health of the harness as the accuracy of the model.
These tests pin that boundary from both sides: an unparseable completion must
become a MISSING record (never a wrong answer, never an empty-string match), and
``score_cell`` must keep the missing count out of the accuracy denominator.
"""

from __future__ import annotations

import math

import pytest

from scale.experiments.nonlatent_iclr.tasks import models
from scale.experiments.nonlatent_iclr.tasks.scoring import (
    FAMILY_ANSWER_PATTERN,
    ScoringRefusal,
    exact_match,
    extract_answer,
    normalize,
    score_cell,
)


def test_an_unparseable_completion_is_missing_not_wrong() -> None:
    """The defect this module exists to prevent, pinned directly."""
    # Given: one parseable correct answer, one parseable wrong answer, and one
    # completion with no answer-grammar token at all.
    records = [
        {"decoded": "The answer is v1234", "gold": "v1234"},
        {"decoded": "The answer is v9999", "gold": "v1234"},
        {"decoded": "I cannot find the key, sorry.", "gold": "v1234"},
    ]

    # When: the cell is scored.
    result = score_cell(records, "associative_recall")

    # Then: the unparseable completion is counted as MISSING, and is excluded
    # from the accuracy denominator rather than charged to the model as wrong --
    # accuracy is 1/2 over the two parseable records, not 1/3 over all three.
    assert result["n_missing"] == 1
    assert result["n_correct"] == 1
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["accuracy"] != pytest.approx(1 / 3)
    assert result["missing_fraction"] == pytest.approx(1 / 3)


def test_extract_answer_returns_none_rather_than_the_empty_string() -> None:
    """An empty-string sentinel would compare equal to an empty gold and match."""
    # Given: a completion with no value token.
    # When: the family-aware extractor reads it.
    # Then: the absence is None, not "", so downstream comparisons cannot turn
    # "unparseable" into a spurious match or a spurious wrong answer.
    assert extract_answer("no answer here", "associative_recall") is None
    assert extract_answer("", "code_dataflow") is None
    assert extract_answer("42", "associative_recall") is None


def test_a_trailing_period_is_not_an_answer_difference() -> None:
    # Given: a gold without a period and a prediction with one (plus casing and
    # collapsed whitespace differences).
    # When: they are compared after normalization.
    # Then: the difference in surface form is not an answer difference.
    assert exact_match("42.", "42") is True
    assert exact_match("v1234.", "v1234") is True
    assert normalize("  Forty   Two.  ") == "forty two"
    # and the extractor does not capture the sentence period in the first place
    assert extract_answer("The answer is 42.", "code_dataflow") == "42"


def test_an_empty_record_list_refuses() -> None:
    # Given: a cell with no records.
    # When / Then: scoring is refused rather than reporting an accuracy for an
    # empty measurement.
    with pytest.raises(ScoringRefusal, match="empty cell"):
        score_cell([], "associative_recall")


def test_an_unknown_family_refuses() -> None:
    # Given: a family that is not registered.
    # When / Then: both entry points refuse rather than guessing a grammar.
    with pytest.raises(ScoringRefusal, match="no answer grammar"):
        extract_answer("whatever", "needle_in_haystack")
    with pytest.raises(ScoringRefusal, match="no answer grammar"):
        score_cell([{"decoded": "v1", "gold": "v1"}], "needle_in_haystack")


def test_extraction_is_family_aware() -> None:
    # Given: completions written in each family's own answer grammar.
    # When: each is read under its family.
    # Then: the extractor binds the grammar that family uses and does not bind
    # another family's token.
    assert extract_answer("The answer is v1234", "associative_recall") == "v1234"
    assert extract_answer("The answer is 42", "code_dataflow") == "42"
    assert extract_answer("state is 3/7", "finite_hmm") == "3/7"
    # a bare integer is not a value token, and a fraction needs its slash
    assert extract_answer("The answer is 42", "associative_recall") is None
    assert extract_answer("The answer is 37", "finite_hmm") is None


def test_overwrite_extraction_takes_the_last_write() -> None:
    # Given: a completion that echoes two writes to the same key.
    # When: the overwrite family's answer is extracted.
    # Then: the LAST value wins, which is the overwrite task's own rule, so the
    # extractor agrees with the generator about what the answer is.
    assert extract_answer("first v111 then v222", "overwrite_delayed_query") == "v222"


def test_a_trailing_period_difference_scores_as_correct() -> None:
    # Given: a correct record whose completion ends a sentence.
    records = [{"decoded": "The answer is 42.", "gold": "42"}]
    # When / Then: it scores correct end to end, not missing and not wrong.
    result = score_cell(records, "code_dataflow")
    assert result["n_correct"] == 1
    assert result["n_missing"] == 0
    assert result["accuracy"] == 1.0


def test_a_cell_with_no_parseable_completion_reports_undefined_not_zero() -> None:
    # Given: a cell where every completion is unparseable.
    records = [
        {"decoded": "sorry", "gold": "v1"},
        {"decoded": "no key found", "gold": "v2"},
    ]
    # When: it is scored.
    result = score_cell(records, "associative_recall")
    # Then: accuracy is undefined (NaN), NOT 0.0 -- a 0.0 would present a total
    # extraction failure as a model that answered every query wrong.
    assert math.isnan(result["accuracy"])
    assert result["missing_fraction"] == 1.0
    assert result["n_missing"] == 2


def test_a_record_without_its_decoded_or_gold_field_refuses() -> None:
    # Given: records missing a required key.
    # When / Then: the malformed record is refused, because scoring it silently
    # would be the same class of error as scoring an unparsed completion.
    with pytest.raises(ScoringRefusal, match="record lacks"):
        score_cell([{"gold": "v1"}], "associative_recall")
    with pytest.raises(ScoringRefusal, match="record lacks"):
        score_cell([{"decoded": "v1"}], "associative_recall")


def test_the_scorer_covers_every_registered_family() -> None:
    # Given: the registry's family set and this module's answer grammars.
    # When / Then: they are the same set, so a newly registered family cannot be
    # added without an answer grammar (the module refuses the mismatch at import).
    assert set(FAMILY_ANSWER_PATTERN) == set(models.FAMILIES)


def test_case_and_internal_whitespace_do_not_change_a_match() -> None:
    # Given: a capitalised value token and a doubled internal space.
    # When / Then: normalization folds both without touching the answer.
    assert exact_match("V1234", "v1234") is True
    assert normalize("v 1234") == "v 1234"
