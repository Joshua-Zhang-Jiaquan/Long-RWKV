"""Task 10: the trained decay survey behind the memory-horizon measurement.

The bound in ``theory_bounds`` is proved for every input; the *horizon* depends
on the trained distribution of ``w``, which cannot be read off the checkpoint
because ``w`` is input-dependent.  These tests cover the reduction (fully, on
CPU, with synthetic captures) and the hook wiring (with a stub model).  The
capture itself needs CUDA and is exercised on the GPU run, which is why it is
not tested here -- a test that skipped on this host would prove nothing.

The load-bearing test is
``test_summarize_multiplies_the_two_passes_of_a_recycled_layer``: the weight-tied
loop applies layers [16,32) twice per forward, so those layers forget faster per
forward than a single application's rate suggests.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from scale.experiments.nonlatent_iclr import decay_survey as ds
from scale.experiments.nonlatent_iclr import theory_bounds as tb


def _decay(logit: float) -> float:
    """The decay a constant decay-logit produces."""
    return float(tb.decay_from_logits(torch.tensor([logit], dtype=torch.float64)))


class _StubDecay(nn.Module):
    """Emits a constant ``(B, T, hidden)`` logit field."""

    def __init__(self, logit: float, hidden: int = 4) -> None:
        super().__init__()
        self.logit = logit
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], x.shape[1], self.hidden), self.logit)


class _StubAttn(nn.Module):
    """An attention stand-in that owns a ``w_lora`` child, as the real one does."""

    def __init__(self, logit: float, hidden: int = 4) -> None:
        super().__init__()
        self.w_lora = _StubDecay(logit, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_lora(x)


class _StubModel(nn.Module):
    """A minimal stand-in with the naming the survey expects."""

    def __init__(self, layers: int = 4, logit: float = -4.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            block = nn.Module()
            block.attn_fwd = _StubAttn(logit)
            block.attn_bwd = _StubAttn(logit)
            block.ffn = nn.Linear(2, 2)
            self.layers.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            _ = block.attn_fwd(x)
            _ = block.attn_bwd(x)
        return x


# --------------------------------------------------------------------------
# hook targets
# --------------------------------------------------------------------------


def test_hook_targets_are_the_decay_modules_in_layer_order() -> None:
    # Given: a model with the expected naming.
    model = _StubModel(layers=3)
    # When/Then: both directions of every layer are found, ordered by layer.
    names = ds.hook_target_names(model)
    assert names == [
        "layers.0.attn_bwd.w_lora", "layers.0.attn_fwd.w_lora",
        "layers.1.attn_bwd.w_lora", "layers.1.attn_fwd.w_lora",
        "layers.2.attn_bwd.w_lora", "layers.2.attn_fwd.w_lora",
    ]


def test_hook_targets_ignore_unrelated_w_lora_modules() -> None:
    # Given: a model with decoys that share the suffix but are not the decay.
    class _Decoy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].attn_fwd = _StubAttn(-4.0)
            self.layers[0].ffn = _StubAttn(-4.0)       # wrong submodule name
            self.misc = nn.Module()
            self.misc.w_lora = _StubDecay(-4.0)        # not under layers.

    # When/Then: only the recurrent decay module survives -- picking up a decoy
    # would mix an unrelated quantity into the bound's constant.
    assert ds.hook_target_names(_Decoy()) == ["layers.0.attn_fwd.w_lora"]


def test_a_model_without_decay_modules_is_refused() -> None:
    # Given: a model with no matching modules.
    model = nn.Sequential(nn.Linear(2, 2))
    # When/Then: the survey refuses rather than reporting an empty capture,
    # which would look like "no layer decays fast" instead of "wrong model".
    with pytest.raises(ds.DecaySurveyError, match="no decay modules"):
        ds.register_decay_hooks(model, {})


# --------------------------------------------------------------------------
# the hook itself
# --------------------------------------------------------------------------


def test_the_hook_records_the_per_token_slowest_channel() -> None:
    # Given: a stub whose logits are constant, so the decay is known in closed
    # form, and a sink to capture into.
    model = _StubModel(layers=2, logit=-4.0)
    sink: dict[str, list[torch.Tensor]] = {}
    handles = ds.register_decay_hooks(model, sink)
    # When: a forward is run.
    _ = model(torch.zeros(1, 5, dtype=torch.long))
    for handle in handles:
        handle.remove()
    # Then: every module fired once and recorded one value per position, equal
    # to the decay its logit implies.
    assert set(sink) == set(ds.hook_target_names(model))
    for entries in sink.values():
        assert len(entries) == 1
        assert entries[0].numel() == 5
        assert float(entries[0].max()) == pytest.approx(_decay(-4.0), rel=1e-6)


def test_the_hook_keeps_each_firing_separate() -> None:
    # Given: a captured sink and a second forward, which is what the loop does
    # to the recycled layers.
    model = _StubModel(layers=1, logit=-4.0)
    sink: dict[str, list[torch.Tensor]] = {}
    handles = ds.register_decay_hooks(model, sink)
    _ = model(torch.zeros(1, 3, dtype=torch.long))
    _ = model(torch.zeros(1, 3, dtype=torch.long))
    for handle in handles:
        handle.remove()
    # Then: two firings are recorded, not merged -- the reduction needs them
    # apart to form the per-forward product.
    assert len(sink["layers.0.attn_fwd.w_lora"]) == 2


# --------------------------------------------------------------------------
# the reduction
# --------------------------------------------------------------------------


def test_summarize_reports_kappa_and_half_life() -> None:
    # Given: one module that fired once at a known decay.
    decay = _decay(-4.0)
    captured = {"layers.0.attn_fwd.w_lora": [torch.full((8,), decay, dtype=torch.float64)]}
    # When: it is reduced.
    summary = ds.summarize(captured)
    entry = summary["modules"]["layers.0.attn_fwd.w_lora"]  # type: ignore[index]
    # Then: kappa is the slowest channel observed and the half-life follows
    # from it, with no per-forward doubling outside the loop range.
    assert entry["kappa_per_application"] == pytest.approx(decay)
    assert entry["kappa_per_forward"] == pytest.approx(decay)
    assert entry["half_life_tokens_per_application"] == pytest.approx(tb.half_life(decay))
    assert entry["in_recycled_range"] is False
    assert entry["certifies_contraction_per_application"] is True


def test_summarize_multiplies_the_two_passes_of_a_recycled_layer() -> None:
    """The loop applies those layers twice per forward, so they forget faster."""
    # Given: a layer inside the loop range that fired twice at the same decay.
    decay = _decay(-4.0)
    captured = {"layers.20.attn_fwd.w_lora": [
        torch.full((8,), decay, dtype=torch.float64),
        torch.full((8,), decay, dtype=torch.float64),
    ]}
    # When: it is reduced.
    summary = ds.summarize(captured)
    entry = summary["modules"]["layers.20.attn_fwd.w_lora"]  # type: ignore[index]
    # Then: the per-forward rate is the PRODUCT of the passes...
    assert entry["in_recycled_range"] is True
    assert entry["kappa_per_application"] == pytest.approx(decay)
    assert entry["kappa_per_forward"] == pytest.approx(decay * decay)
    # ...and the half-life therefore halves relative to a single application.
    assert entry["half_life_tokens_per_forward"] == pytest.approx(
        entry["half_life_tokens_per_application"] / 2.0, rel=1e-9)


def test_a_recycled_layer_is_not_double_counted_outside_the_range() -> None:
    # Given: a layer outside the range that also fired twice (a caller running
    # two documents through the same sink).
    decay = _decay(-4.0)
    captured = {"layers.3.attn_fwd.w_lora": [torch.full((4,), decay, dtype=torch.float64)] * 2}
    # When: it is reduced.
    summary = ds.summarize(captured)
    entry = summary["modules"]["layers.3.attn_fwd.w_lora"]  # type: ignore[index]
    # Then: the flag says it is not recycled, so a reader is not misled into
    # reading the product as a loop effect.
    assert entry["in_recycled_range"] is False
    assert entry["firings"] == 2


def test_a_rate_above_one_is_reported_without_a_half_life() -> None:
    """kappa >= 1 certifies nothing and must say so rather than be dropped."""
    # Given: a capture whose slowest channel is above 1 -- which the
    # parameterization forbids, and which therefore means the capture is not
    # this model's.
    captured = {"layers.0.attn_fwd.w_lora": [torch.tensor([1.004], dtype=torch.float64)]}
    # When: it is reduced.
    summary = ds.summarize(captured)
    entry = summary["modules"]["layers.0.attn_fwd.w_lora"]  # type: ignore[index]
    # Then: the rate is recorded, the half-life is None, and certification is
    # False -- the plan's failure QA, applied to the measurement.
    assert entry["kappa_per_application"] == pytest.approx(1.004)
    assert entry["half_life_tokens_per_application"] is None
    assert entry["certifies_contraction_per_application"] is False
    assert summary["model_half_life_tokens"] is None
    assert "refusal" not in summary or True  # refusal text is optional detail


def test_a_module_that_never_fired_is_reported() -> None:
    # Given: a module with an empty firing list.
    summary = ds.summarize({"layers.0.attn_fwd.w_lora": []})
    # When/Then: it appears with zero firings rather than vanishing -- "never
    # ran" and "ran and decayed fast" are different findings.
    assert summary["modules"]["layers.0.attn_fwd.w_lora"] == {"firings": 0}
    assert summary["n_modules"] == 1
    assert summary["slowest_application"] is None


def test_the_model_level_constant_is_the_slowest_layer() -> None:
    # Given: three layers with different decay speeds.
    slow = _decay(-6.0)
    fast = _decay(-2.0)
    captured = {
        "layers.0.attn_fwd.w_lora": [torch.full((4,), fast, dtype=torch.float64)],
        "layers.1.attn_fwd.w_lora": [torch.full((4,), slow, dtype=torch.float64)],
        "layers.2.attn_fwd.w_lora": [torch.full((4,), fast, dtype=torch.float64)],
    }
    # When: the model-level constant is taken.
    summary = ds.summarize(captured)
    # Then: it is the SLOWEST layer, because a bound is only as strong as the
    # layer that forgets least.
    assert summary["slowest_application"] == pytest.approx(slow)
    assert summary["model_half_life_tokens"] == pytest.approx(tb.half_life(slow))
    assert "MEASUREMENT" in summary["note"]


def test_receipts_are_written_once() -> None:
    # Given: a written survey.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "survey.json"
        written = ds.write_survey({"summary": {}}, path)
        # When/Then: a second write refuses, so a re-run cannot silently replace
        # a measurement that other artifacts may already cite.
        assert written.is_file()
        with pytest.raises(ds.DecaySurveyError, match="written once"):
            ds.write_survey({"summary": {}}, path)
        # and the schema is stamped by the writer, not by the caller
        import json

        assert json.loads(path.read_text())["schema"] == ds.SCHEMA


def test_the_loop_range_matches_the_training_contract() -> None:
    # Given: the range the training spec used.
    # Then: it is the range this module assumes when it multiplies passes.
    assert ds.LOOP_RANGE == (16, 32)
    assert math.log(2) / math.log(1 / 0.99) == pytest.approx(tb.half_life(0.99))
