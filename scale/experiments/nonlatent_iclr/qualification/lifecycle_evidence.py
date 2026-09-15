"""Independent engineering observations; never a qualification release schema."""

from __future__ import annotations

from typing import Annotated, Final, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..full_canvas import BoundaryErrorCode

type Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type ComparisonCode = Literal[
    "exact_match", "shape_mismatch", "nonfinite", "dtype_mismatch", "value_mismatch",
]
type FailureCode = ComparisonCode | Literal[
    "rejection_failed", "model_runtime_error", "synchronization_error", "boundary_error",
]
type Scenario = Literal[
    "reference_a", "repeat_a", "caller_mutation", "edit_reference", "stale_revision",
    "closed_request", "reference_b", "old_session", "foreign_session", "a_b_a",
    "reset_retirement", "reset_reopen", "reset_old_session", "injected_forward", "caller_failure",
    "failure_retirement", "failure_reopen", "failure_old_session",
]

type StepContract = tuple[
    Scenario, int, int, Literal["comparison", "rejection", "injection"],
    Literal["exact_match", "stale_canvas", "session_closed", "session_mismatch",
            "caller_boundary_failure_injected"],
]

REQUIRED_STEPS: Final[tuple[StepContract, ...]] = (
    ("reference_a", 0, 2, "comparison", "exact_match"),
    ("repeat_a", 2, 3, "comparison", "exact_match"),
    ("caller_mutation", 3, 4, "comparison", "exact_match"),
    ("edit_reference", 4, 6, "comparison", "exact_match"),
    ("stale_revision", 6, 6, "rejection", "stale_canvas"),
    ("closed_request", 6, 6, "rejection", "session_closed"),
    ("reference_b", 6, 8, "comparison", "exact_match"),
    ("old_session", 8, 8, "rejection", "session_mismatch"),
    ("foreign_session", 8, 8, "rejection", "session_mismatch"),
    ("a_b_a", 8, 9, "comparison", "exact_match"),
    ("reset_retirement", 9, 9, "rejection", "session_closed"),
    ("reset_reopen", 9, 10, "comparison", "exact_match"),
    ("reset_old_session", 10, 10, "rejection", "session_mismatch"),
    ("injected_forward", 10, 11, "comparison", "exact_match"),
    ("caller_failure", 10, 11, "injection", "caller_boundary_failure_injected"),
    ("failure_retirement", 11, 11, "rejection", "session_closed"),
    ("failure_reopen", 11, 12, "comparison", "exact_match"),
    ("failure_old_session", 12, 12, "rejection", "session_mismatch"),
)


class StrictEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class LifecycleConfig(StrictEvidence):
    """Fixed 48-token probe; caller supplies the loaded model's vocabulary/device."""

    vocab_size: int = Field(ge=4, le=65_536)
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda:[0-9]+)$")


class InputEvidence(StrictEvidence):
    name: Literal["a", "b", "edited_a"]
    sha256: Digest
    shape: tuple[Literal[1], Literal[48]] = (1, 48)
    dtype: Literal["int64"] = "int64"
    hash_encoding: Literal["row_major_little_endian_int64"] = "row_major_little_endian_int64"


class CallWindow(StrictEvidence):
    scenario: Scenario
    calls_before: int = Field(ge=0)
    calls_after: int = Field(ge=0)


class ComparisonEvidence(CallWindow):
    kind: Literal["comparison"] = "comparison"
    code: ComparisonCode
    expected_shape: tuple[int, int, int]
    actual_shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    actual_dtype: str
    reference_dtype: str
    all_finite: bool
    exact_equal: bool
    max_abs_error: float | None = Field(ge=0.0)

    @property
    def passed(self) -> bool:
        return (self.code == "exact_match" and self.all_finite and self.exact_equal
                and self.max_abs_error == 0.0 and self.actual_dtype == self.reference_dtype
                and self.actual_shape == self.reference_shape == self.expected_shape)


class RejectionEvidence(CallWindow):
    kind: Literal["rejection"] = "rejection"
    expected_code: BoundaryErrorCode
    observed_code: BoundaryErrorCode | None

    @property
    def passed(self) -> bool:
        return self.expected_code == self.observed_code and self.calls_before == self.calls_after


class InjectionEvidence(CallWindow):
    kind: Literal["injection"] = "injection"
    code: Literal["caller_boundary_failure_injected"] = "caller_boundary_failure_injected"
    synchronizations_before: int = Field(ge=0)
    synchronizations_after: int = Field(ge=0)
    cuda_device_fault_recovery: Literal[False] = False

    @property
    def passed(self) -> bool:
        return (self.calls_after - self.calls_before == 1
                and self.synchronizations_before == self.calls_before
                and self.synchronizations_after == self.calls_after)


class ExecutionErrorEvidence(CallWindow):
    kind: Literal["execution_error"] = "execution_error"
    code: Literal["model_runtime_error", "synchronization_error", "boundary_error"]
    exception_type: str
    detail: str

    @property
    def passed(self) -> bool:
        return False


type StepEvidence = Annotated[
    ComparisonEvidence | RejectionEvidence | InjectionEvidence | ExecutionErrorEvidence,
    Field(discriminator="kind"),
]


class LifecycleEvidence(StrictEvidence):
    scope: Literal["unbound_full_canvas_lifecycle_observation"] = "unbound_full_canvas_lifecycle_observation"
    runtime_promoted: Literal[False] = False
    device: str
    vocab_size: int
    inputs: tuple[InputEvidence, InputEvidence, InputEvidence]
    steps: tuple[StepEvidence, ...]
    model_calls: int = Field(ge=0)
    synchronizations: int = Field(ge=0)
    completed: bool
    failure_code: FailureCode | None
    atol: float = Field(default=0.0, ge=0.0, le=0.0)
    rtol: float = Field(default=0.0, ge=0.0, le=0.0)

    @property
    def passed(self) -> bool:
        if not (self.completed and self.failure_code is None and self.model_calls == 12
                and self.synchronizations == 12 and all(step.passed for step in self.steps)
                and len(self.steps) == len(REQUIRED_STEPS)):
            return False
        for step, (scenario, before, after, kind, required_code) in zip(self.steps, REQUIRED_STEPS, strict=True):
            if (step.scenario, step.calls_before, step.calls_after, step.kind) != (scenario, before, after, kind):
                return False
            match step:
                case ComparisonEvidence():
                    if step.expected_shape != (1, 48, self.vocab_size):
                        return False
                    code = step.code
                case RejectionEvidence():
                    code = step.expected_code
                case InjectionEvidence():
                    code = step.code
                case ExecutionErrorEvidence():
                    return False
                case unreachable:
                    assert_never(unreachable)
            if code != required_code:
                return False
        return True
