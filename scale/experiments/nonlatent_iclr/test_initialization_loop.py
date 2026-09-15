"""Original loop execution with explicitly non-FLA tiny residual blocks."""
from __future__ import annotations

import torch
import pytest

from . import initialization_loop as evidence


@pytest.mark.parametrize("reps", [1, 2, 4, 0, -2])
def test_identity_when_gates_zero(reps: int) -> None:
    # Given: finite, exactly representable CPU residuals; two trained gates.
    probe = evidence.TinyLoopProbe(32)
    loop = probe.model.attach_backbone_loop((16, 32), 2)
    h = torch.ones(1, 2, 3)
    # When: original model forward-loop region executes.
    result = probe.run(h, reps)
    # Then: extra passes preserve the base result, not the raw embedding.
    assert torch.equal(result.hidden, h + 32)
    assert torch.equal(loop.gates(), torch.zeros(2))
    assert (result.outer_invocations, result.base_calls, result.recycled_calls) == (1, 32, 16 * max(reps, 0))
    assert all(call.shape == (1, 2, 3) for call in result.calls)


def test_gradient_when_nonzero_recycled_residual() -> None:
    # Given: each block adds one; each extra pass adds 16 before gating.
    probe = evidence.TinyLoopProbe(32)
    loop = probe.model.attach_backbone_loop((16, 32), 2, 0.5)
    # When: differentiate the actual zero-gated loop.
    probe.run(torch.ones(1, 2, 3)).hidden.sum().backward()
    # Then: each raw gate derivative is 6 elements * 16 residual * .5.
    grad = loop.gates_raw.grad
    assert grad is not None and torch.equal(grad, torch.full((2,), 48.0))


def test_response_when_last_gate_reused_for_extrapolation() -> None:
    # Given: unequal gates distinguish last-gate reuse from cycling or first reuse.
    probe = evidence.TinyLoopProbe(4)
    loop = probe.model.attach_backbone_loop((2, 4), 2)
    with torch.no_grad():
        loop.gates_raw.copy_(torch.atanh(torch.tensor([0.25, 0.5])))
    # When: execute four extra passes using the historical override.
    result = probe.run(torch.ones(1, 2, 3), 4)
    # Then: base 5 + 2*(.25 + .5 + .5 + .5).
    assert torch.equal(result.hidden, torch.full((1, 2, 3), 8.5))


def test_sharing_and_highway_when_recycling_twice() -> None:
    # Given: independently allocated blocks, each with its own parameter.
    probe = evidence.TinyLoopProbe(4)
    _ = probe.model.attach_backbone_loop((2, 4), 2)
    # When: base plus two extra passes execute.
    result = probe.run(torch.ones(1, 2, 3))
    # Then: exact objects recur; each repetition inherits base highway, not previous rep.
    evidence.require_shared(result.calls[2:4], result.calls[4:6])
    evidence.require_shared(result.calls[2:4], result.calls[6:8])
    assert [c.highway for c in result.calls] == [0, 1, 2, 3, 4, 5, 4, 5]
    assert len({c.block_id for c in result.calls}) == 4
    assert len({c.parameter_id for c in result.calls}) == 4


def test_false_sharing_when_equal_values_separate_objects() -> None:
    # Given: value-equal independently allocated tiny blocks.
    left, right = evidence.TinyLoopProbe(2), evidence.TinyLoopProbe(2)
    a, b = left.run(torch.ones(1, 1, 1)), right.run(torch.ones(1, 1, 1))
    # When / Then: a false reuse claim is rejected despite identical outputs.
    assert torch.equal(a.hidden, b.hidden)
    with pytest.raises(evidence.SourceContractError, match="sharing"):
        evidence.require_shared(a.calls, b.calls)


def test_clone_tying_distinct_from_loop_reuse() -> None:
    # Given: actual recycled calls, plus warm-start-clone-style copies — value-
    # identical parameters in separately allocated objects, mirroring the
    # historical attn_bwd-is-a-clone-of-attn_fwd construction.
    probe = evidence.TinyLoopProbe(2)
    _ = probe.model.attach_backbone_loop((0, 2), 1)
    result = probe.run(torch.ones(1, 1, 1))
    original = []
    for block in probe.model.layers:
        assert isinstance(block, evidence.TinyResidualBlock)
        original.append(block.delta)
    clones = [delta.detach().clone() for delta in original]
    cloned_calls = tuple(
        evidence.BlockCall(call.block_id, id(clone), call.shape, call.highway)
        for call, clone in zip(result.calls[:2], clones, strict=True)
    )
    # When / Then: clones are value-tied but object-distinct, so the object-
    # identity check rejects them while accepting the actually recycled objects.
    assert all(torch.equal(a, b) for a, b in zip(original, clones, strict=True))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(original, clones, strict=True))
    with pytest.raises(evidence.SourceContractError, match="false sharing"):
        evidence.require_shared(cloned_calls, result.calls[2:])
    evidence.require_shared(result.calls[:2], result.calls[2:])


@pytest.mark.parametrize("bounds,reps", [((-1, 2), 1), ((2, 2), 1), ((3, 2), 1), ((0, 5), 1), ((0, 4), 0), ((0, 4), 3)])
def test_rejection_when_attachment_invalid(bounds: tuple[int, int], reps: int) -> None:
    # Given: actual model attachment seam with four tiny blocks.
    probe = evidence.TinyLoopProbe(4)
    # When / Then: preserve historical validation, not a new policy.
    with pytest.raises(ValueError):
        probe.model.attach_backbone_loop(bounds, reps)


def test_rejection_when_attached_twice() -> None:
    # Given: already attached loop.
    probe = evidence.TinyLoopProbe(4)
    _ = probe.model.attach_backbone_loop((0, 4), 1)
    # When / Then: actual repeated attachment fails.
    with pytest.raises(RuntimeError):
        probe.model.attach_backbone_loop((0, 4), 1)


@pytest.mark.parametrize("raw,expected", [(100.0, 0.5), (-100.0, -0.5)])
def test_bound_and_buffers_when_gate_saturates(raw: float, expected: float) -> None:
    # Given: actual control, custom scale.
    probe = evidence.TinyLoopProbe(4)
    loop = probe.model.attach_backbone_loop((1, 4), 2, 0.5)
    with torch.no_grad():
        loop.gates_raw.fill_(raw)
    # When: evaluate bounded gate and persistent state.
    gates, state = loop.gates(), loop.state_dict()
    # Then: actual saturation and serialization surface.
    assert torch.equal(gates, torch.full((2,), expected))
    assert set(state) == {"lo", "hi", "gates_raw"}
    assert state["lo"].dtype == torch.int64 and int(state["hi"]) == 4


@pytest.mark.parametrize("streams", [2, 4])
def test_stream_identity_when_raw_factors_zero(streams: int) -> None:
    # Given: safe exact integer fixture; no assertion for arbitrary floating cancellation.
    probe = evidence.TinyLoopProbe(4)
    mixer = probe.model.attach_residual_streams(streams)
    # When: actual model range executes the stream combination and update.
    result = probe.run(torch.ones(1, 2, 3))
    # Then: baseline preserved, block batch not multiplied by stream count.
    assert torch.equal(result.hidden, torch.full((1, 2, 3), 5.0))
    assert result.base_calls == 4 and result.recycled_calls == 0
    assert all(c.shape == (1, 2, 3) for c in result.calls)
    w, a, p = mixer.factors(0)
    assert torch.equal(w, torch.eye(streams)[0])
    assert torch.equal(a, torch.eye(streams))
    assert torch.equal(p, torch.ones(streams))


def test_zero_gate_gradient_when_recycled_residual_zero() -> None:
    # Given: the residual condition is intentionally absent.
    probe = evidence.TinyLoopProbe(2)
    loop = probe.model.attach_backbone_loop((0, 2), 1)
    for block in probe.model.layers:
        assert isinstance(block, evidence.TinyResidualBlock)
        with torch.no_grad():
            block.delta.zero_()
    # When: differentiate the real loop, rather than an analytical replacement.
    probe.run(torch.ones(1, 1, 1)).hidden.sum().backward()
    # Then: zero residual cannot provide a nonzero gate gradient.
    grad = loop.gates_raw.grad
    assert grad is not None and torch.equal(grad, torch.zeros(1))


def test_block_gradient_when_zero_gates_discard_extra_path() -> None:
    # Given: all blocks have independently learnable unit residuals.
    probe = evidence.TinyLoopProbe(4)
    _ = probe.model.attach_backbone_loop((2, 4), 2)
    # When: differentiate the original gated execution at init.
    probe.run(torch.ones(1, 2, 3)).hidden.sum().backward()
    # Then: block gradients include the base pass only, not the zero-gated extras.
    for block in probe.model.layers:
        assert isinstance(block, evidence.TinyResidualBlock)
        grad = block.delta.grad
        assert grad is not None and torch.equal(grad, torch.full((1,), 6.0))


@pytest.mark.parametrize("streams", [1, 3, 8])
def test_stream_rejection_when_count_unsupported(streams: int) -> None:
    # Given: actual stream attachment method.
    probe = evidence.TinyLoopProbe(2)
    # When / Then: single-stream means no attachment, not an accepted mixer count.
    with pytest.raises(ValueError):
        probe.model.attach_residual_streams(streams)
