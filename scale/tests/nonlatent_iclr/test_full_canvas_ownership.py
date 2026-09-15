from __future__ import annotations

import weakref

import pytest
import torch

from scale.experiments.nonlatent_iclr import full_canvas as boundary
from scale.tests.nonlatent_iclr.test_full_canvas import RecordingModel


def test_snapshot_isolated_when_caller_and_model_mutate_aliases() -> None:
    # Given: both original caller storage and a previous model input are mutated.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    canvas = torch.arange(4).reshape(1, 4)
    request = adapter.open(canvas)
    canvas.fill_(90)
    adapter.forward(request)
    model.calls[0][0].fill_(80)
    # When: the same unedited revision is recomputed.
    adapter.forward(request)
    # Then: neither external alias changes the owned snapshot.
    assert torch.equal(model.calls[1][0], torch.arange(4).reshape(1, 4))


def test_input_released_when_open_returns() -> None:
    # Given: the adapter has its own snapshot.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    canvas = torch.ones(1, 4, dtype=torch.long)
    reference = weakref.ref(canvas)
    adapter.open(canvas)
    # When: the caller releases its tensor.
    del canvas
    # Then: the adapter has not retained the caller tensor.
    assert reference() is None


def test_outputs_released_when_consumers_release_them() -> None:
    # Given: a completed call whose result is owned by the caller/spy only.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    request = adapter.open(torch.ones(1, 4, dtype=torch.long))
    output = adapter.forward(request)
    reference = weakref.ref(output[0])
    model.output = (torch.empty(0), torch.empty(0))
    # When: the caller releases the result.
    del output
    # Then: no logits or hidden states are retained by the adapter.
    assert reference() is None


def test_zero_calls_when_open_would_replace_active_session() -> None:
    # Given: an already active canvas.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    canvas = torch.ones(1, 4, dtype=torch.long)
    adapter.open(canvas)
    # When: another open attempts implicit replacement.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.open(canvas)
    # Then: lifecycle retirement must be explicit.
    assert error.value.code == "session_active"
    assert model.calls == []


@pytest.mark.parametrize("canvas", [torch.ones(4, dtype=torch.long), torch.ones(1, 4), torch.empty(1, 0, dtype=torch.long)])
def test_zero_calls_when_canvas_invalid(canvas: torch.Tensor) -> None:
    # Given: a rank, dtype, or empty-canvas violation.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    # When: ownership is requested for malformed token IDs.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.open(canvas)
    # Then: fail at the CPU boundary, not inside the model.
    assert error.value.code == "invalid_canvas"
    assert model.calls == []


def test_original_request_remains_valid_when_edit_shape_rejected() -> None:
    # Given: a shape-changing replacement is rejected before revision advancement.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    canvas = torch.ones(1, 4, dtype=torch.long)
    request = adapter.open(canvas)
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.edit_canvas(request, torch.ones(1, 3, dtype=torch.long))
    assert error.value.code == "canvas_shape_mismatch"
    assert model.calls == []
    # When: the original request is used after the failed edit.
    adapter.forward(request)
    # Then: failed edits are atomic.
    assert torch.equal(model.calls[0][0], canvas)


def test_reopened_session_works_when_reset_repeated() -> None:
    # Given: reset is idempotent and retires an edited session.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    canvas = torch.ones(1, 4, dtype=torch.long)
    old = adapter.open(canvas)
    adapter.edit_canvas(old, canvas)
    adapter.reset()
    adapter.reset()
    current = adapter.open(canvas)
    # When: a fresh request is used.
    adapter.forward(current)
    # Then: reset does not disable future valid sessions or reuse identities.
    assert current.session_id is not old.session_id
    assert len(model.calls) == 1
