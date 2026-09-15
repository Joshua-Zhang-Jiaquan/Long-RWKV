"""Counted calls and exact fail-closed lifecycle observations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from ..full_canvas import CanvasBoundaryError, FullCanvasModel
from .lifecycle_evidence import (
    BoundaryErrorCode, ComparisonCode, ComparisonEvidence, ExecutionErrorEvidence,
    FailureCode, LifecycleConfig, RejectionEvidence, Scenario, StepEvidence,
)


class LifecycleAbort(Exception):
    def __init__(self, code: FailureCode) -> None:
        super().__init__(code)
        self.code: FailureCode = code


class CallerBoundaryFailure(Exception):
    """Controlled caller exception, never a CUDA/device fault."""


class LifecycleChecks:
    """Mutable per-invocation ledger; counts model attempts, not internal layer calls."""

    def __init__(self, model: FullCanvasModel[torch.Tensor], synchronize: Callable[[], None]) -> None:
        self.model = model
        self.synchronize = synchronize
        self.calls = 0
        self.synchronizations = 0
        self.steps: list[StepEvidence] = []
        self.scenario: Scenario = "reference_a"
        self.start = 0

    # This signature mirrors the approved model protocol's independent legacy flags.
    def __call__(
        self, input_ids: torch.Tensor, *, force_forward: Literal[False],
        z_slots: None, state_cache: None, use_cache: Literal[False],
    ) -> torch.Tensor:
        self.calls += 1
        try:
            output = self.model(input_ids=input_ids, force_forward=force_forward,
                                z_slots=z_slots, state_cache=state_cache, use_cache=use_cache)
        except RuntimeError as error:
            raise LifecycleAbort(self.execution_error("model_runtime_error", error)) from error
        try:
            self.synchronize()
        except RuntimeError as error:
            raise LifecycleAbort(self.execution_error("synchronization_error", error)) from error
        self.synchronizations += 1
        return output

    def execution_error(
        self, code: Literal["model_runtime_error", "synchronization_error", "boundary_error"],
        error: RuntimeError | CanvasBoundaryError,
    ) -> FailureCode:
        self.steps.append(ExecutionErrorEvidence(
            scenario=self.scenario, calls_before=self.start, calls_after=self.calls,
            code=code, exception_type=type(error).__name__, detail=str(error),
        ))
        return code

    def begin(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.start = self.calls

    def direct(self, ids: torch.Tensor) -> torch.Tensor:
        return self(input_ids=ids.clone(), force_forward=False,
                    z_slots=None, state_cache=None, use_cache=False)

    @staticmethod
    def snapshot(output: torch.Tensor) -> torch.Tensor:
        """Own reference bytes before another model call can reuse output storage."""
        return output.detach().to(device="cpu", copy=True)

    def compare(self, pair: tuple[torch.Tensor, torch.Tensor], config: LifecycleConfig) -> None:
        actual, reference = pair
        shape = (1, 48, config.vocab_size)
        finite = bool(torch.isfinite(actual).all().item()) and bool(torch.isfinite(reference).all().item())
        shape_valid = tuple(actual.shape) == tuple(reference.shape) == shape
        equal = False
        delta: float | None = None
        code: ComparisonCode = "shape_mismatch"
        if shape_valid:
            code = "nonfinite"
            if finite:
                left, right = actual.detach().cpu(), reference.detach().cpu()
                equal = torch.equal(left, right)
                difference = (left.to(torch.float64) - right.to(torch.float64)).abs().max()
                if bool(torch.isfinite(difference).item()):
                    delta = float(difference.item())
                code = "dtype_mismatch"
                if actual.dtype == reference.dtype:
                    code = "exact_match" if equal else "value_mismatch"
        self.steps.append(ComparisonEvidence(
            scenario=self.scenario, calls_before=self.start, calls_after=self.calls,
            code=code, expected_shape=shape, actual_shape=tuple(actual.shape),
            reference_shape=tuple(reference.shape), actual_dtype=str(actual.dtype),
            reference_dtype=str(reference.dtype), all_finite=finite,
            exact_equal=equal, max_abs_error=delta,
        ))
        if code != "exact_match":
            raise LifecycleAbort(code)

    def reject(self, operation: Callable[[], torch.Tensor], expected: BoundaryErrorCode) -> None:
        observed: BoundaryErrorCode | None = None
        try:
            operation()
        except CanvasBoundaryError as error:
            observed = error.code
        evidence = RejectionEvidence(
            scenario=self.scenario, calls_before=self.start, calls_after=self.calls,
            expected_code=expected, observed_code=observed,
        )
        self.steps.append(evidence)
        if not evidence.passed:
            raise LifecycleAbort("rejection_failed")
