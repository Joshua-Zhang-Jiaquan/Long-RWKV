from __future__ import annotations

from dataclasses import replace
from typing import Literal

import pytest
import torch

from scale.experiments.nonlatent_iclr import full_canvas as boundary


class RecordingModel:
    """Record the real callable seam without importing a GPU model or weights."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, bool, None, None, bool]] = []
        self.output: tuple[torch.Tensor, torch.Tensor] = (torch.ones(2, 4, 7), torch.zeros(2, 4, 3))

    # Signature intentionally mirrors the actual model's five independent inputs.
    def __call__(
        self, input_ids: torch.Tensor, *, force_forward: bool,
        z_slots: None, state_cache: None, use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((input_ids, force_forward, z_slots, state_cache, use_cache))
        return self.output


def test_output_preserved_when_full_canvas_forwarded() -> None:
    # Given: a batched, noncontiguous full canvas and an ordinary tuple output.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    canvas = torch.arange(16).reshape(2, 8)[:, ::2]
    request = adapter.open(canvas)
    # When: the public boundary is called.
    result = adapter.forward(request)
    # Then: no output wrapping/slicing and exactly the required explicit kwargs.
    assert result is model.output
    assert len(model.calls) == 1
    ids, force_forward, z_slots, state_cache, use_cache = model.calls[0]
    assert ids.shape == (2, 4)
    assert torch.equal(ids, canvas)
    assert ids.data_ptr() != canvas.data_ptr()
    assert (force_forward, z_slots, state_cache, use_cache) == (False, None, None, False)


@pytest.mark.parametrize("cache", [object(), {}, [], False, 0, torch.zeros(1)])
def test_zero_calls_when_any_cache_supplied(cache: object) -> None:
    # Given: even an empty/false-valued cache is still a supplied cache object.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    request = adapter.open(torch.ones(1, 4, dtype=torch.long))
    # When: the rejected object crosses the boundary.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.forward(request, state_cache=cache)
    # Then: no model call can consume a prefilled cache.
    assert error.value.code == "state_cache_forbidden"
    assert model.calls == []


@pytest.mark.parametrize("mode", ["full_model_prefix_cache", "loop_prefix_cache", "streaming", "unknown"])
def test_zero_calls_when_cache_mode_unsupported(mode: str) -> None:
    # Given: an active session, but no qualified cached execution mode.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    request = adapter.open(torch.ones(1, 4, dtype=torch.long))
    # When: a non-full-canvas mode is requested.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.forward(request, mode=mode)
    # Then: stable typed rejection, never fallback to the model.
    assert error.value.code == "unsupported_cache_mode"
    assert model.calls == []


@pytest.mark.parametrize("option", ["use_cache", "z_slots", "coordinator"])
def test_zero_calls_when_nonlatent_contract_violated(
    option: Literal["use_cache", "z_slots", "coordinator"],
) -> None:
    # Given: a valid session and prohibited legacy options.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    request = adapter.open(torch.ones(1, 4, dtype=torch.long))
    # When: either cache production or latent coordination is requested.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.forward(
            request, use_cache=option == "use_cache",
            z_slots=torch.zeros(1) if option == "z_slots" else None,
            coordinator=object() if option == "coordinator" else None,
        )
    # Then: exact rejection codes and no invocation.
    assert error.value.code == f"{option}_forbidden"
    assert model.calls == []


@pytest.mark.parametrize("operation", ["forward", "edit", "close"])
def test_zero_calls_when_request_crosses_sessions(operation: str) -> None:
    # Given: independent adapters whose revision numbers both start at zero.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    other = boundary.FullCanvasAdapter(model)
    canvas = torch.ones(1, 4, dtype=torch.long)
    adapter.open(canvas)
    foreign = other.open(canvas)
    # When: a foreign request attempts any identity-guarded operation.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        if operation == "forward":
            adapter.forward(foreign)
        elif operation == "edit":
            adapter.edit_canvas(foreign, canvas)
        else:
            adapter.close(foreign)
    # Then: neither a call nor session takeover occurs.
    assert error.value.code == "session_mismatch"
    assert model.calls == []


@pytest.mark.parametrize("offset", [-1, 1])
def test_zero_calls_when_revision_is_not_current(offset: int) -> None:
    # Given: an identity with a stale or future revision.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    request = adapter.open(torch.ones(1, 4, dtype=torch.long))
    invalid = replace(request, canvas_revision=boundary.CanvasRevision(offset))
    # When: recomputation is requested using the wrong revision.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.forward(invalid)
    # Then: wrong revisions do not invoke the model.
    assert error.value.code == "stale_canvas"
    assert model.calls == []


def test_full_edited_canvas_forwarded_when_revision_current() -> None:
    # Given: a replacement canvas, not a prefix/suffix cache update.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    original = adapter.open(torch.ones(2, 4, dtype=torch.long))
    edited = torch.arange(8).reshape(2, 4)
    current = adapter.edit_canvas(original, edited)
    edited.fill_(99)
    # When: the new revision is recomputed.
    adapter.forward(current)
    # Then: the complete owned edit is used, unaffected by external mutation.
    assert current.canvas_revision == original.canvas_revision + 1
    assert current.session_id is original.session_id
    assert torch.equal(model.calls[0][0], torch.arange(8).reshape(2, 4))


def test_zero_calls_when_edit_invalidates_previous_request() -> None:
    # Given: a request superseded by even an equal-valued edit.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    canvas = torch.ones(1, 4, dtype=torch.long)
    old = adapter.open(canvas)
    adapter.edit_canvas(old, canvas)
    # When: the old request is replayed.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.forward(old)
    # Then: equality of token contents does not resurrect revision identity.
    assert error.value.code == "stale_canvas"
    assert model.calls == []


@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("retire", ["close", "reset"])
def test_zero_calls_when_session_retired(retire: str, reopen: bool) -> None:
    # Given: a retired session, optionally replaced at the same revision zero.
    model = RecordingModel()
    adapter = boundary.FullCanvasAdapter(model)
    canvas = torch.ones(1, 4, dtype=torch.long)
    old = adapter.open(canvas)
    if retire == "close":
        adapter.close(old)
    else:
        adapter.reset()
    if reopen:
        adapter.open(canvas)
    # When: a previously valid request is replayed.
    with pytest.raises(boundary.CanvasBoundaryError) as error:
        adapter.forward(old)
    # Then: close/reset never permit revision/session ABA.
    assert error.value.code == ("session_mismatch" if reopen else "session_closed")
    assert model.calls == []
