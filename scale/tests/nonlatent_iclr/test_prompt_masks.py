from __future__ import annotations

import pytest

from scale.experiments.nonlatent_iclr.metrics import MetricInputError, prompt_preserving_corruption


def test_prompt_corruption_preserves_condition_and_excludes_pad() -> None:
    # Given: a prompt, answer tokens, and trailing padding.
    tokens = (11, 12, 21, 22, 0, 0)
    condition = (True, True, False, False, False, False)
    prediction = (False, False, True, True, False, False)
    # When: deterministic CPU corruption is applied.
    result = prompt_preserving_corruption(tokens, condition, prediction, pad_id=0, mask_token_id=99)
    # Then: known condition stays visible, answer positions are masked, and padding stays padding.
    assert result.corrupted == (11, 12, 99, 99, 0, 0)
    assert result.condition_mask == condition
    assert result.prediction_mask == prediction


def test_prompt_corruption_rejects_known_future_answer() -> None:
    # Given: an answer position incorrectly admitted to the condition mask.
    tokens = (11, 12, 21, 0)
    condition = (True, True, True, False)
    prediction = (False, False, True, False)
    # When: prompt qualification is constructed.
    # Then: future-answer leakage is rejected.
    with pytest.raises(MetricInputError):
        prompt_preserving_corruption(tokens, condition, prediction, pad_id=0, mask_token_id=99)


def test_prompt_corruption_rejects_unclassified_active_token() -> None:
    # Given: an active token absent from both masks.
    # When: deterministic full-prediction corruption qualifies the prompt.
    # Then: hidden future information cannot remain visible.
    with pytest.raises(MetricInputError):
        prompt_preserving_corruption((1, 2, 0), (True, False, False), (False, False, False), pad_id=0, mask_token_id=99)
