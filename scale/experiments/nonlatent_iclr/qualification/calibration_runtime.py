"""Bounded throughput calibration for the Task-6 forecast ledger.

The runtime is deliberately hardware-agnostic: callers supply a *closure* that performs one
masked-canvas forward, so nothing here imports torch or names a tensor type. The GPU specifics
live in ``calibration.py``. That keeps this module exactly testable on a CPU-only host.

Three deliberate behaviours:

* The canvas ladder runs **small to large** and stops at the first out-of-memory. A canvas that
  does not fit is recorded as unavailable with its reason, never as a crash and never as a
  fabricated number. The proven smoke path uses a 48-token canvas while the ladder requests up
  to 32768, so where a rank stops is a real, reportable outcome.
* A backward/optimizer-step measurement is **only** taken when a factory is supplied. When none
  is provided the record says so explicitly rather than reporting a zero.
* When a wall-clock budget is supplied, canvases that would start after it are recorded as
  unavailable (``time_budget_exhausted``) instead of being silently dropped, so a truncated rank
  is still a complete record of what it did.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from .calibration_contracts import CANVAS_TOKENS, CalibrationMeasurement

CANVAS_LADDER: Final = (512, *CANVAS_TOKENS)
FORWARD_REPETITIONS: Final = 3
BACKWARD_NOT_MEASURED: Final = (
    "no single-gradient-step runner was supplied; the arm's backward step was not timed"
)
TIME_BUDGET_EXHAUSTED: Final = "time_budget_exhausted"

#: Performs one masked-canvas forward for a given canvas size, raising on failure.
ForwardOnce = Callable[[int], None]
#: Measures one gradient step for a given canvas size and returns wall-clock seconds.
BackwardStep = Callable[[int], float]
#: Reads how many positions the last corruption selected, or None when a rank has none yet.
SelectedTokens = Callable[[], int]


class CudaLike(Protocol):
    """The ``torch.cuda`` surface this module uses; satisfied by the real module.

    Every device-scoped call takes the device explicitly. Calling them without one measures the
    process's *current* device, which on a rank working on another GPU reports zero allocations,
    and a zero peak reads as "unmeasurable" rather than as the bug it is.
    """

    def synchronize(self, device: object = None) -> None: ...
    def reset_peak_memory_stats(self, device: object = None) -> None: ...
    def max_memory_allocated(self, device: object = None) -> int: ...
    def max_memory_reserved(self, device: object = None) -> int: ...
    def empty_cache(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CanvasOutcome:
    """Either a measurement, or the reason the canvas could not be measured."""

    canvas_tokens: int
    measurement: CalibrationMeasurement | None
    unavailable_reason: str | None


def is_out_of_memory(error: BaseException) -> bool:
    """Recognise the CUDA OOM shape without importing torch at module scope."""
    if type(error).__name__ == "OutOfMemoryError":
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def measure_canvas(
    forward_once: ForwardOnce,
    cuda: CudaLike,
    canvas_tokens: int,
    *,
    device: object = None,
    repetitions: int = FORWARD_REPETITIONS,
    backward_step: BackwardStep | None = None,
    optimizer_step: BackwardStep | None = None,
    optimizer_not_measured_reason: str | None = None,
    selected_tokens: SelectedTokens | None = None,
    warmup: bool = False,
) -> CanvasOutcome:
    """Time one canvas size, or report why it could not be measured.

    Peak memory counters are reset per canvas so each row describes its own canvas rather than
    the high-water mark of everything measured before it. ``device`` must be the device the work
    actually runs on, for the reason given on ``CudaLike``.

    With ``warmup`` an extra untimed forward is taken on this canvas first, because a rung's first
    touch pays for cold kernels, autotuning and allocator growth, and because a rung's timing would
    otherwise be perturbed by the previous rung's backward leaving memory changed. The warm-up runs
    inside the out-of-memory guard, so a canvas too large to warm is recorded as unavailable rather
    than discarding the rungs measured before it. Its allocation is excluded from the peak by
    resetting the counters afterwards.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    try:
        if warmup:
            # Warm the canvas's first touch (kernel autotune, allocator growth), then return those
            # blocks to the device. Warming the backward here as well was tried and reverted: it
            # needs a second activation set live at the same time as the timed one, and on a 79 GiB
            # device that pushed the small arms into CUDA OOM during the dplr backward. The
            # one-time backward warm-up before the ladder covers that cost instead.
            #
            # The warm-up sits inside this guard and is deliberately *not* continued past a failure.
            # A warm-up OOM means "this canvas does not fit", which is a recorded outcome; left
            # outside the guard it would throw away every rung the rank had already measured, and
            # catching it only to keep running kernels on the same context produced
            # "CUDA error: an illegal memory access was encountered" on the looped arms.
            forward_once(canvas_tokens)
            cuda.synchronize(device)
            cuda.empty_cache()
        cuda.reset_peak_memory_stats(device)
        cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repetitions):
            forward_once(canvas_tokens)
        cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_allocated = int(cuda.max_memory_allocated(device))
        peak_reserved = int(cuda.max_memory_reserved(device))
    except BaseException as error:  # noqa: BLE001 - OOM must not abort the whole rank
        if not is_out_of_memory(error):
            raise
        cuda.empty_cache()
        return CanvasOutcome(canvas_tokens, None, f"out_of_memory:{type(error).__name__}")
    forward_rate = (canvas_tokens * repetitions) / elapsed if elapsed > 0 else 0.0
    if forward_rate <= 0 or peak_allocated <= 0:
        return CanvasOutcome(canvas_tokens, None, "measurement_not_positive")
    backward_seconds: float | None = None
    reason: str | None = BACKWARD_NOT_MEASURED
    if backward_step is not None:
        measured = float(backward_step(canvas_tokens))
        if measured <= 0:
            raise ValueError("backward step returned a non-positive duration")
        backward_seconds, reason = measured, None
    optimizer_seconds: float | None = None
    optimizer_reason: str | None = optimizer_not_measured_reason or BACKWARD_NOT_MEASURED
    if optimizer_step is not None:
        measured_optimizer = float(optimizer_step(canvas_tokens))
        if measured_optimizer <= 0:
            raise ValueError("optimizer step returned a non-positive duration")
        optimizer_seconds, optimizer_reason = measured_optimizer, None
    observed_selection = 0
    if selected_tokens is not None and backward_step is not None:
        observed_selection = int(selected_tokens())
    return CanvasOutcome(
        canvas_tokens,
        CalibrationMeasurement(
            canvas_tokens=canvas_tokens,
            forward_tokens_per_second=forward_rate,
            backward_seconds=backward_seconds,
            backward_not_measured_reason=reason,
            optimizer_step_seconds=optimizer_seconds,
            optimizer_step_not_measured_reason=optimizer_reason,
            selected_tokens=observed_selection,
            sampler_nfe_seconds=elapsed / repetitions,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=max(peak_reserved, peak_allocated),
        ),
        None,
    )


def run_ladder(
    forward_once: ForwardOnce,
    cuda: CudaLike,
    *,
    device: object = None,
    ladder: tuple[int, ...] = CANVAS_LADDER,
    backward_step: BackwardStep | None = None,
    optimizer_step: BackwardStep | None = None,
    optimizer_not_measured_reason: str | None = None,
    selected_tokens: SelectedTokens | None = None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    warmup: bool = False,
) -> tuple[tuple[CalibrationMeasurement, ...], tuple[str, ...], tuple[int, ...]]:
    """Measure the canvas ladder, stopping at the first canvas that does not fit.

    Returns the measurements taken, the warnings, and the canvases that were not measured. The
    stopping canvas ends the ladder because a larger one cannot be expected to fit either. A
    canvas that would begin after ``deadline`` is recorded as unavailable for that reason.
    """
    measurements: list[CalibrationMeasurement] = []
    warnings: list[str] = []
    for index, canvas in enumerate(ladder):
        if deadline is not None and clock() >= deadline:
            warnings.extend(f"canvas_{blocked}_unavailable:{TIME_BUDGET_EXHAUSTED}" for blocked in ladder[index:])
            return tuple(measurements), tuple(warnings), tuple(ladder[index:])
        outcome = measure_canvas(
            forward_once,
            cuda,
            canvas,
            device=device,
            backward_step=backward_step,
            optimizer_step=optimizer_step,
            optimizer_not_measured_reason=optimizer_not_measured_reason,
            selected_tokens=selected_tokens,
            warmup=warmup,
        )
        # Release this rung's cached blocks before the next, larger rung asks for new ones. Without
        # this the allocator holds every previous canvas size, and the peak creeps upward until a
        # rung that would fit on its own fails.
        cuda.empty_cache()
        if outcome.measurement is None:
            warnings.append(f"canvas_{canvas}_unavailable:{outcome.unavailable_reason}")
            return tuple(measurements), tuple(warnings), tuple(ladder[index:])
        measurements.append(outcome.measurement)
    return tuple(measurements), tuple(warnings), ()
