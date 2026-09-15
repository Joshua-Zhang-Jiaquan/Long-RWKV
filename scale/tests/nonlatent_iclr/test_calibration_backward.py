"""Behavioral checks of the calibration gradient step against the pinned loss path."""

from __future__ import annotations

import torch

from scale.experiments.nonlatent_iclr import trainer_semantics as seam
from scale.experiments.nonlatent_iclr.qualification import calibration_backward as back

CANVAS = 32


class _TinyMaskedModel(torch.nn.Module):
    """Small enough for a CPU test, wide enough to accept the pinned mask token id."""

    def __init__(self, vocab: int = 65_536, width: int = 4) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, width)
        self.head = torch.nn.Linear(width, vocab)

    def forward(self, input_ids, *, force_forward, z_slots, state_cache, use_cache):
        del force_forward, z_slots, state_cache, use_cache
        return self.head(self.embed(input_ids))


def test_doc_spans_are_the_middle_third() -> None:
    # Given / When: the production span builder over a canvas with a nonempty middle third
    spans = back._doc_spans(torch, torch.device("cpu"), CANVAS)

    # Then: it declares one document whose masked completion is exactly [10, 20)
    assert [span.tolist() for span in spans] == [[[10]], [[10]], [[20]]]
    assert [span.dtype for span in spans] == [torch.int64] * 3


def test_backward_step_selects_exactly_the_middle_third() -> None:
    # Given: a tiny trainable model driven through the real pinned loss, with the selection
    # observed. The extracted lane cannot be called cold (its restricted builtins lack
    # ``__import__`` until torch's lazy bindings are materialised), so it is exercised through
    # the production path, which is what the job actually runs.
    functions = seam.load_functions()
    model = _TinyMaskedModel()
    observed: list[int] = []
    backward, _, _ = back.build_gradient_step(
        model, functions, torch, torch.device("cpu"), observer=observed.append
    )

    # When: one canvas is measured
    elapsed = backward(CANVAS)

    # Then: the loss was taken over exactly the declared span, not the whole canvas
    assert elapsed > 0
    assert observed == [CANVAS // 3]
    assert observed[0] < CANVAS


def test_backward_step_returns_positive_seconds_and_populates_gradients() -> None:
    # Given: a tiny trainable model driven through the real pinned loss
    functions = seam.load_functions()
    model = _TinyMaskedModel()

    # When: the gradient step is built with no optimizer and one canvas is measured
    backward, optimizer, reason = back.build_gradient_step(model, functions, torch, torch.device("cpu"))

    # Then: the step is timed, gradients exist, and the absent optimizer is explained not zeroed
    elapsed = backward(CANVAS)
    assert elapsed > 0
    assert optimizer is None
    assert reason == back.OPTIMIZER_NOT_MEASURED_SCHEDULED_OFF
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert any(bool(torch.count_nonzero(gradient).item()) for gradient in gradients)


def test_optimizer_step_is_timed_when_an_optimizer_is_supplied() -> None:
    # Given: the same setup, but with a real AdamW supplied
    functions = seam.load_functions()
    model = _TinyMaskedModel()
    optimizer = back.build_adamw(model, torch)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    # When: the gradient step runs before the optimizer step
    backward, optimizer_step, reason = back.build_gradient_step(
        model, functions, torch, torch.device("cpu"), optimizer=optimizer
    )
    backward_seconds = backward(CANVAS)
    assert optimizer_step is not None
    optimizer_seconds = optimizer_step(CANVAS)

    # Then: both are timed, the reason is absent, and the step actually moved parameters
    assert backward_seconds > 0
    assert optimizer_seconds > 0
    assert reason is None
    after = [parameter.detach().clone() for parameter in model.parameters()]
    assert any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))


def test_sharded_optimizer_reason_is_preserved_verbatim() -> None:
    # Given: the large-arm reason, which stands in for a number that would misrepresent the cost
    functions = seam.load_functions()
    model = _TinyMaskedModel()

    # When: no optimizer is supplied and the caller explains why
    _, optimizer, reason = back.build_gradient_step(
        model, functions, torch, torch.device("cpu"), optimizer_reason=back.OPTIMIZER_NOT_MEASURED_SHARDED
    )

    # Then: the explanation survives rather than being replaced by the generic default
    assert optimizer is None
    assert reason == back.OPTIMIZER_NOT_MEASURED_SHARDED


def test_empty_corruption_mask_is_rejected_rather_than_timed(monkeypatch) -> None:
    # Given: a canvas of pure pad ids, so every position is ineligible
    functions = seam.load_functions()
    model = _TinyMaskedModel()

    def pad_batch(torch_module, device, canvas_tokens: int):
        ids = torch_module.zeros((1, canvas_tokens), dtype=torch.long, device=device)
        return ids, torch_module.ones_like(ids, dtype=bool)

    monkeypatch.setattr(back, "_canvas_batch", pad_batch)
    backward, _, _ = back.build_gradient_step(model, functions, torch, torch.device("cpu"))

    # When / Then: an empty selection is an error, never a fabricated timing
    try:
        _ = backward(CANVAS)
    except ValueError as error:
        assert str(error) == back.EMPTY_CORRUPTION_MASK
    else:
        raise AssertionError("expected the empty-corruption guard to reject the canvas")


def test_backward_step_restores_eval_mode_so_later_timings_stay_comparable() -> None:
    # Given: a model constructed in eval mode, as the calibration builds it.
    functions = seam.load_functions()
    model = _TinyMaskedModel()
    model.eval()
    backward, _, _ = back.build_gradient_step(model, functions, torch, torch.device("cpu"))

    # When: the gradient step runs.
    _ = backward(CANVAS)

    # Then: train mode does not leak into the next canvas's forward timing.
    assert model.training is False
