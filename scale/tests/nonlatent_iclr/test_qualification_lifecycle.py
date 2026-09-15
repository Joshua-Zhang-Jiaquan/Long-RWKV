from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from struct import pack
from typing import Literal, assert_never

import pytest
import torch

from scale.experiments.nonlatent_iclr.qualification import lifecycle
from scale.tests.nonlatent_iclr.test_qualification_masked_forward import (
    OpenedSession,
    opened_sessions as opened_sessions,
)
from scale.experiments.nonlatent_iclr.full_canvas import CanvasBoundaryError

type Fault = Literal["clean", "drift", "nan", "shape", "runtime", "alias_edit"]


class CPUFailure(RuntimeError):
    def __init__(self, code: Literal["cpu_forward_failure", "cpu_callback_failure"]) -> None:
        super().__init__(code)
        self.code: Literal["cpu_forward_failure", "cpu_callback_failure"] = code


@dataclass(frozen=True, slots=True)
class Call:
    ids: torch.Tensor
    inference: bool
    flags: tuple[bool | None, Literal["omitted"] | None, Literal["omitted"] | None, bool | None]


class RecordingModel:
    """Mutable CPU call ledger; outputs depend on the entire supplied canvas."""

    def __init__(self, fault: Fault = "clean") -> None:
        self.calls: list[Call] = []
        self.fault: Fault = fault
        self.buffer = torch.empty(1, 48, 8)

    # Independent legacy kwargs are intentional: omitted flags must fail tests.
    def __call__(
        self, input_ids: torch.Tensor, *, force_forward: bool | None = None,
        z_slots: Literal["omitted"] | None = "omitted",
        state_cache: Literal["omitted"] | None = "omitted",
        use_cache: bool | None = None,
    ) -> torch.Tensor:
        self.calls.append(Call(input_ids.clone(), torch.is_inference_mode_enabled(),
                               (force_forward, z_slots, state_cache, use_cache)))
        output = input_ids.float().unsqueeze(-1) + torch.arange(8).float()
        output = output + input_ids.sum().float()
        if len(self.calls) == 2:
            match self.fault:
                case "drift":
                    output[0, 0, 0] += 0.0001
                case "nan":
                    output[0, 0, 0] = float("nan")
                case "shape":
                    output = output[:, :-1]
                case "runtime":
                    raise CPUFailure("cpu_forward_failure")
                case "clean" | "alias_edit":
                    return output
                case unreachable:
                    assert_never(unreachable)
        match self.fault:
            case "alias_edit":
                if len(self.calls) == 5:
                    output += 1
                self.buffer.copy_(output)
                return self.buffer
            case "clean" | "drift" | "nan" | "shape" | "runtime":
                return output
            case unreachable:
                assert_never(unreachable)


class Synchronizer:
    """CPU ordering observer, explicitly not a CUDA synchronization substitute."""

    def __init__(self, model: RecordingModel) -> None:
        self.model = model
        self.observed: list[int] = []

    def __call__(self) -> None:
        self.observed.append(len(self.model.calls))
        assert torch.is_inference_mode_enabled()


def test_lifecycle_when_cpu_model_is_deterministic(opened_sessions: list[OpenedSession]) -> None:
    # Given: a loaded CPU callable and an explicit CPU synchronization observer.
    model = RecordingModel()
    sync = Synchronizer(model)
    # When: the public seam runs the bounded real-adapter scenarios.
    evidence = lifecycle.probe_lifecycle(model, lifecycle.LifecycleConfig(vocab_size=8), sync)
    # Then: exact evidence agrees with real calls and all sessions are retired.
    assert evidence.passed and evidence.completed
    assert evidence.model_calls == 12
    assert sync.observed == list(range(1, 13))
    assert evidence.synchronizations == 12
    assert evidence.device == "cpu"
    assert evidence.runtime_promoted is False
    assert len(set(item.sha256 for item in evidence.inputs)) == 3
    for observation, call_index in zip(evidence.inputs, (0, 6, 4), strict=True):
        tokens = tuple(int(token.item()) for token in model.calls[call_index].ids.flatten())
        assert observation.sha256 == sha256(pack("<48q", *tokens)).hexdigest()
    assert all(call.inference and call.flags == (False, None, None, False) for call in model.calls)
    assert all(call.ids.shape == (1, 48) for call in model.calls)
    assert not torch.equal(model.calls[0].ids, model.calls[6].ids)
    assert torch.equal(model.calls[0].ids, model.calls[8].ids)
    assert not torch.equal(model.calls[0].ids, model.calls[4].ids)
    comparisons = [step for step in evidence.steps if step.kind == "comparison"]
    assert len(comparisons) == 9
    assert all(step.code == "exact_match" and step.max_abs_error == 0.0 for step in comparisons)
    rejections = [step for step in evidence.steps if step.kind == "rejection"]
    assert {step.observed_code for step in rejections} == {"stale_canvas", "session_closed", "session_mismatch"}
    assert all(step.calls_after == step.calls_before and step.passed for step in rejections)
    assert {step.scenario for step in rejections} == {
        "stale_revision", "closed_request", "old_session", "foreign_session",
        "reset_retirement", "reset_old_session", "failure_retirement", "failure_old_session",
    }
    injection = next(step for step in evidence.steps if step.kind == "injection")
    assert injection.code == "caller_boundary_failure_injected"
    assert injection.calls_after - injection.calls_before == 1
    assert injection.synchronizations_after - injection.synchronizations_before == 1
    for session in opened_sessions:
        with pytest.raises(CanvasBoundaryError, match="session_closed"):
            session.adapter.forward(session.request)
    assert len(model.calls) == 12


@pytest.mark.parametrize("fault,code", [
    ("drift", "value_mismatch"), ("nan", "nonfinite"), ("shape", "shape_mismatch"),
    ("runtime", "model_runtime_error"),
])
def test_lifecycle_fails_closed_when_forward_is_bad(
    fault: Fault, code: str, opened_sessions: list[OpenedSession],
) -> None:
    # Given: a controlled CPU regression on the first adapter call.
    model = RecordingModel(fault)
    # When: the public probe sees a mismatch or an actual forward exception.
    evidence = lifecycle.probe_lifecycle(model, lifecycle.LifecycleConfig(vocab_size=8), Synchronizer(model))
    # Then: no tolerance widening, no subsequent call, and no active session survives.
    assert not evidence.passed and not evidence.completed
    assert evidence.failure_code == code
    assert evidence.model_calls == 2
    assert evidence.atol == evidence.rtol == 0.0
    for session in opened_sessions:
        with pytest.raises(CanvasBoundaryError, match="session_closed"):
            session.adapter.forward(session.request)
    assert len(model.calls) == 2


def test_lifecycle_preserves_edit_mismatch_when_model_reuses_output_storage() -> None:
    # Given: the adapter's edited output is wrong and aliases the next reference output.
    model = RecordingModel("alias_edit")
    # When: comparison observes both forwards rather than their final shared storage.
    evidence = lifecycle.probe_lifecycle(model, lifecycle.LifecycleConfig(vocab_size=8), Synchronizer(model))
    # Then: the next forward cannot overwrite the failure into an exact match.
    assert evidence.failure_code == "value_mismatch"
    assert evidence.model_calls == 6
    assert evidence.steps[-1].scenario == "edit_reference"


def test_lifecycle_cleans_up_when_synchronization_raises(opened_sessions: list[OpenedSession]) -> None:
    # Given: a CPU callback failure after a real adapter forward.
    model = RecordingModel()

    def synchronize() -> None:
        if len(model.calls) == 2:
            raise CPUFailure("cpu_callback_failure")

    # When: synchronization fails (not a simulated CUDA fault).
    evidence = lifecycle.probe_lifecycle(model, lifecycle.LifecycleConfig(vocab_size=8), synchronize)
    # Then: it stops rather than injecting/reopening and retires the real adapter.
    assert evidence.failure_code == "synchronization_error"
    assert evidence.model_calls == 2 and evidence.synchronizations == 1
    for session in opened_sessions:
        with pytest.raises(CanvasBoundaryError, match="session_closed"):
            session.adapter.forward(session.request)
