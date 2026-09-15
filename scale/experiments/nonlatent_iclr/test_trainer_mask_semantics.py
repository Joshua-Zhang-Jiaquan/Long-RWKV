"""Behavioral checks of extracted corruption and selected-logit loss."""
import math
from unittest.mock import patch

import pytest
import torch

from scale.experiments.nonlatent_iclr import trainer_semantics as seam


@pytest.mark.parametrize("span_prob", [0.0, 1.0])
def test_eligibility_and_realized_fraction_when_final_block_is_partial(span_prob: float) -> None:
    # Given: controlled random draws choose low bucket, ratio .175, offset 0.
    functions = seam.load_functions()
    ids = torch.tensor([[10, 0, 12, 13, 14, 15, 16]])
    attention = torch.tensor([[1, 1, 0, 1, 1, 1, 1]])
    draws = [torch.full((1, 2), 0.5), torch.zeros(1, 2), torch.zeros(1, 7), torch.zeros(1), torch.zeros(1), torch.ones(1)]
    # When
    with patch("torch.multinomial", return_value=torch.zeros(2, 1, dtype=torch.long)), patch("torch.rand", side_effect=draws):
        corrupted, mask, bucket, gen_rows, fraction = functions.sample_corruption(ids, attention, 4, span_prob, torch.Generator())
    # Then
    expected = torch.tensor([[True, False, False, False, True, False, False]]) if span_prob else torch.tensor([[True, False, False, True, True, True, True]])
    assert torch.equal(mask, expected)
    assert torch.equal(corrupted[~mask], ids[~mask])
    assert torch.all(corrupted[mask] == 65535)
    assert torch.equal(bucket, torch.zeros(1, 2, dtype=torch.long))
    assert not gen_rows.any()
    expected_fraction = torch.tensor([[0.5, 1 / 3]]) if span_prob else torch.ones(1, 2)
    assert torch.allclose(fraction, expected_fraction)


def test_generation_canvas_when_prefix_draw_is_controlled() -> None:
    # Given: minimum prefix is 16 even when draw is zero.
    functions = seam.load_functions()
    ids = torch.arange(1, 33).view(1, 32)
    attention = torch.ones_like(ids)
    attention[0, 20] = 0
    draws = [torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 32), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    # When
    with patch("torch.multinomial", return_value=torch.zeros(2, 1, dtype=torch.long)), patch("torch.rand", side_effect=draws):
        corrupted, mask, bucket, rows, fraction = functions.sample_corruption(ids, attention, 16, 0.0, torch.Generator(), gen_prob=1.0)
    # Then
    expected = torch.arange(32).view(1, 32) >= 16
    expected[0, 20] = False
    assert torch.equal(mask, expected)
    assert torch.equal(corrupted[~mask], ids[~mask])
    assert rows.tolist() == [True]
    assert bucket.tolist() == [[2, 2]]
    assert fraction.tolist() == [[0.0, 1.0]]


@pytest.mark.parametrize("usable", [True, False])
def test_document_override_when_spans_are_present(usable: bool) -> None:
    # Given: one usable document on row 0; row 1 has no valid spans.
    functions = seam.load_functions()
    ids = torch.arange(1, 17).view(2, 8)
    attention = torch.ones_like(ids)
    if not usable:
        attention[0, 2:5] = 0
    spans = (torch.tensor([[0], [-1]]), torch.tensor([[2], [-1]]), torch.tensor([[5], [-1]]))
    generator = torch.Generator().manual_seed(42)
    baseline = functions.sample_corruption(ids, attention, 4, 0.0, torch.Generator().manual_seed(42))
    # When
    corrupted, mask, bucket, rows, fraction = functions.sample_corruption(ids, attention, 4, 0.0, generator, docgen_prob=1.0, doc_spans=spans)
    # Then
    assert torch.equal(mask[1], baseline[1][1])
    assert torch.equal(corrupted[~mask], ids[~mask])
    if usable:
        assert mask[0].tolist() == [False, False, True, True, True, False, False, False]
        assert rows.tolist() == [True, False]
        assert bucket[0].tolist() == [2, 2]
        assert fraction[0].tolist() == [0.5, 0.25]
    else:
        assert torch.equal(mask, baseline[1])
        assert not rows.any()


def test_zero_fraction_when_all_positions_are_ineligible() -> None:
    # Given
    functions = seam.load_functions()
    ids = torch.zeros(1, 3, dtype=torch.long)
    # When
    corrupted, mask, _, _, fraction = functions.sample_corruption(ids, torch.ones_like(ids), 2, 1.0, torch.Generator())
    # Then
    assert torch.equal(corrupted, ids)
    assert not mask.any()
    assert fraction.tolist() == [[0.0, 0.0]]


@pytest.mark.parametrize("empty", [False, True])
def test_selected_token_loss_when_logits_are_hand_computable(empty: bool) -> None:
    # Given: selected CE values log(2), log(4); middle logits are unselected.
    functions = seam.load_functions()
    logits = torch.tensor([[[0.0, 0.0], [100.0, -100.0], [0.0, math.log(3)]]], requires_grad=True)
    targets = torch.zeros(1, 3, dtype=torch.long)
    mask = torch.tensor([[not empty, False, not empty]])
    # When
    loss, diagnostics = functions.masked_diffusion_loss(logits, targets, mask, torch.tensor([[0, 2]]), 2)
    loss.backward()
    # Then: absent buckets are diagnostics NaNs, not evidence scalars.
    assert loss.item() == pytest.approx(0.0 if empty else math.log(8) / 2)
    assert diagnostics["mask_ce"] == pytest.approx(loss.item())
    assert math.isnan(diagnostics["ce_med"])
    assert logits.grad is not None
    assert torch.equal(logits.grad[:, 1], torch.zeros(1, 2))
    if empty:
        assert torch.count_nonzero(logits.grad).item() == 0
        assert all(math.isnan(diagnostics[key]) for key in ("ce_low", "ce_med", "ce_high"))
    else:
        assert diagnostics["ce_low"] == pytest.approx(math.log(2))
        assert diagnostics["ce_high"] == pytest.approx(math.log(4))
        assert torch.count_nonzero(logits.grad[:, [0, 2]]).item() == 4


def test_unselected_logits_excluded_when_values_change() -> None:
    # Given: this is a logit-loss check, not a context-token influence claim.
    functions = seam.load_functions()
    targets = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.tensor([[True, False]])
    bucket = torch.zeros(1, 1, dtype=torch.long)
    # When
    losses = [functions.masked_diffusion_loss(torch.tensor([[[0.0, 0.0], [value, -value]]]), targets, mask, bucket, 2)[0].item() for value in (0.0, 100.0, -100.0)]
    # Then
    assert losses == pytest.approx([math.log(2)] * 3)
