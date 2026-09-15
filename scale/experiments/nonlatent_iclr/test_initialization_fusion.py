"""Actual fusion-gate initialization statements on a tiny attribute sink."""
from __future__ import annotations

import pytest
import torch

from . import initialization_fusion as fusion


def test_fusion_factors_when_init_statements_execute() -> None:
    # Given: the real BiRWKV7Block __init__ fusion statements, tiny hidden.
    got = fusion.characterize(hidden=8, gate_bias_init=4.0)
    # Then: zero projection and exact bias fill, bit-for-bit.
    assert torch.equal(got.fuse_proj.weight, torch.zeros(8, 8))
    assert torch.equal(got.fuse_bias, torch.full((8,), 4.0))


def test_mean_alpha_when_diagnostic_method_executes() -> None:
    # Given: the real mean_forward_alpha diagnostic on the initialized factors.
    got = fusion.characterize(hidden=8, gate_bias_init=4.0)
    # Then: the documented near-forward midpoint sigmoid(4.0) ~ 0.982.
    assert got.mean_alpha == float(torch.sigmoid(torch.full((8,), 4.0)).mean())
    assert 0.98 < got.mean_alpha < 0.983


@pytest.mark.parametrize("gate_bias_init", [4.0, 2.0, 0.0, -3.0])
def test_alpha_input_independent_when_projection_zero(gate_bias_init: float) -> None:
    # Given: exactly-zero fuse_proj at init, two different input triples.
    got = fusion.characterize(hidden=4, gate_bias_init=gate_bias_init)
    g = torch.Generator().manual_seed(7)
    x2, o2_fwd, o2_bwd = (torch.randn(2, 3, 4, generator=g) for _ in range(3))
    alpha2, _ = fusion.fuse_at_init(got.fuse_proj, got.fuse_bias, x2, o2_fwd, o2_bwd)
    # Then: alpha is the bias sigmoid alone and ignores the input.
    expected = torch.sigmoid(torch.full((4,), gate_bias_init))
    assert torch.equal(got.alpha, expected.expand(2, 5, 4))
    assert torch.equal(alpha2, expected.expand(2, 3, 4))


def test_fused_output_when_streams_differ() -> None:
    # Given: the real mix statements with distinct forward/backward streams.
    got = fusion.characterize(hidden=8, gate_bias_init=4.0)
    expected_alpha = torch.sigmoid(torch.full((8,), 4.0))
    # Then: o = alpha*o_fwd + (1-alpha)*o_bwd, forward-dominant at init.
    assert torch.equal(got.output, expected_alpha * got.o_fwd + (1.0 - expected_alpha) * got.o_bwd)
    assert float((got.output - got.o_fwd).detach().norm() / got.o_fwd.norm().clamp_min(1e-6)) < 0.05


def test_gradients_when_residual_between_streams_nonzero() -> None:
    # Given: the fused output of the real mix statements at init.
    got = fusion.characterize(hidden=8, gate_bias_init=4.0)
    # When: differentiate through the original fusion equations.
    got.output.sum().backward()
    # Then: both fusion parameters receive nonzero gradients (no bilinear zero).
    proj_grad = got.fuse_proj.weight.grad
    bias_grad = got.fuse_bias.grad
    assert proj_grad is not None and float(proj_grad.abs().sum()) > 0.0
    assert bias_grad is not None and float(bias_grad.abs().sum()) > 0.0


def test_bias_changes_when_constructor_argument_changes() -> None:
    # Given: identical fixtures differing only in gate_bias_init.
    warm = fusion.characterize(hidden=4, gate_bias_init=4.0)
    cold = fusion.characterize(hidden=4, gate_bias_init=2.0)
    # Then: the constructor argument actually reaches the initialization.
    assert warm.mean_alpha > cold.mean_alpha
    assert cold.mean_alpha == float(torch.sigmoid(torch.full((4,), 2.0)).mean())


def test_rejection_when_fixture_limits_exceeded() -> None:
    # When / Then: tiny-CPU fixture limits fail closed.
    with pytest.raises(fusion.SourceContractError, match="tiny"):
        fusion.characterize(hidden=0, gate_bias_init=4.0)
    with pytest.raises(fusion.SourceContractError, match="tiny"):
        fusion.characterize(hidden=65, gate_bias_init=4.0)
