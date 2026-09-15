from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
import torch

from scale.experiments.nonlatent_iclr.full_canvas import (
    CanvasBoundaryError,
    CanvasRequest,
    FullCanvasAdapter,
)
from scale.experiments.nonlatent_iclr.qualification.contracts import QualificationRuntimeError
from scale.experiments.nonlatent_iclr.qualification.model_checks import masked_forward


@dataclass(frozen=True, slots=True)
class ModelCall:
    input_ids: torch.Tensor
    force_forward: bool | None
    z_slots: Literal["omitted"] | None
    state_cache: Literal["omitted"] | None
    use_cache: bool | None
    inference_enabled: bool


class RecordingModel:
    """Record explicit versus omitted legacy kwargs without any GPU model imports."""

    def __init__(self, output: torch.Tensor | RuntimeError) -> None:
        self.output = output
        self.calls: list[ModelCall] = []

    # Mirrors the model seam; distinct defaults detect omitted kwargs on old callers.
    def __call__(
        self, input_ids: torch.Tensor, *, force_forward: bool | None = None,
        z_slots: Literal["omitted"] | None = "omitted",
        state_cache: Literal["omitted"] | None = "omitted",
        use_cache: bool | None = None,
    ) -> torch.Tensor:
        self.calls.append(ModelCall(
            input_ids, force_forward, z_slots, state_cache, use_cache,
            torch.is_inference_mode_enabled(),
        ))
        match self.output:
            case RuntimeError() as error:
                raise error
            case torch.Tensor() as output:
                return output
            case unreachable:
                from typing import assert_never

                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class OpenedSession:
    adapter: FullCanvasAdapter[torch.Tensor]
    request: CanvasRequest


@pytest.fixture
def opened_sessions(monkeypatch: pytest.MonkeyPatch) -> list[OpenedSession]:
    sessions: list[OpenedSession] = []
    real_open = FullCanvasAdapter[torch.Tensor].open

    def record_open(
        adapter: FullCanvasAdapter[torch.Tensor], input_ids: torch.Tensor,
    ) -> CanvasRequest:
        request = real_open(adapter, input_ids)
        sessions.append(OpenedSession(adapter, request))
        return request

    monkeypatch.setattr(FullCanvasAdapter, "open", record_open)
    return sessions


def assert_retired(sessions: list[OpenedSession], model: RecordingModel) -> None:
    assert len(sessions) == 1
    session = sessions[0]
    with pytest.raises(CanvasBoundaryError) as error:
        session.adapter.forward(session.request)
    assert error.value.code == "session_closed"
    assert len(model.calls) == 1


def test_masked_forward_routes_explicit_kwargs_when_output_valid(
    opened_sessions: list[OpenedSession],
) -> None:
    # Given: a CPU tensor-producing callable and real adapter lifecycle observation.
    model = RecordingModel(torch.zeros(()).expand(1, 48, 65_536))
    # When: the actual editable qualification caller runs its masked forward.
    result = masked_forward(model, torch, torch.device("cpu"))
    # Then: shape/finite contract, token generation and explicit flags are preserved.
    assert result == ((1, 48, 65_536), True)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call.input_ids.shape == (1, 48)
    assert call.input_ids.dtype == torch.int64
    assert call.input_ids.device.type == "cpu"
    assert torch.equal(call.input_ids == 65_535, (torch.arange(48).reshape(1, 48) >= 8) & (torch.arange(48).reshape(1, 48) < 12))
    assert bool(((call.input_ids >= 0) & (call.input_ids <= 65_535)).all())
    assert (call.force_forward, call.z_slots, call.state_cache, call.use_cache) == (False, None, None, False)
    assert call.inference_enabled
    assert_retired(opened_sessions, model)


@pytest.mark.parametrize("shape", [(1, 47, 65_536), (1, 48), (1, 48, 65_535)])
def test_masked_forward_retires_session_when_shape_invalid(
    opened_sessions: list[OpenedSession], shape: tuple[int, ...],
) -> None:
    # Given: actual tensor outputs with incorrect sequence/rank/vocabulary shape.
    model = RecordingModel(torch.zeros(()).expand(shape))
    # When: existing qualification shape validation rejects the output.
    with pytest.raises(QualificationRuntimeError) as error:
        masked_forward(model, torch, torch.device("cpu"))
    # Then: the original failure detail survives and the session cannot be reused.
    assert str(error.value) == f"masked_logits_shape_mismatch:{shape}"
    assert_retired(opened_sessions, model)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_masked_forward_retires_session_when_output_nonfinite(
    opened_sessions: list[OpenedSession], value: float,
) -> None:
    # Given: correct-shape logits that violate the existing finite-value contract.
    model = RecordingModel(torch.tensor(value).expand(1, 48, 65_536))
    # When: the actual caller checks finiteness.
    with pytest.raises(QualificationRuntimeError) as error:
        masked_forward(model, torch, torch.device("cpu"))
    # Then: it fails rather than emitting PASS, and retires the session.
    assert str(error.value) == "masked_logits_non_finite"
    assert_retired(opened_sessions, model)


def test_masked_forward_retires_session_when_model_raises(
    opened_sessions: list[OpenedSession],
) -> None:
    # Given: an exact model exception object, not a synthesized qualification result.
    failure = RuntimeError("recording_model_failure")
    model = RecordingModel(failure)
    inference_before = torch.is_inference_mode_enabled()
    # When: the model fails during the actual caller's adapter forward.
    with pytest.raises(RuntimeError) as error:
        masked_forward(model, torch, torch.device("cpu"))
    # Then: finally retirement neither swallows nor replaces the original exception.
    assert error.value is failure
    assert model.calls[0].inference_enabled
    assert torch.is_inference_mode_enabled() == inference_before
    assert_retired(opened_sessions, model)
