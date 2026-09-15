"""Calibration-runtime tests: ladder behaviour, OOM handling, backward honesty.

These use closures rather than a GPU, so they verify the control flow that decides what gets
recorded -- not the CUDA timing itself, which only a real device can exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from scale.experiments.nonlatent_iclr.qualification import calibration_runtime as cr


class _FakeOutOfMemory(RuntimeError):
    pass


@dataclass
class _FakeCuda:
    allocated: int = 1_024
    reserved: int = 2_048
    resets: int = 0
    syncs: int = 0
    empties: int = 0
    seen_devices: list[object] = field(default_factory=list)

    def synchronize(self, device: object = None) -> None:
        self.syncs += 1
        self.seen_devices.append(device)

    def reset_peak_memory_stats(self, device: object = None) -> None:
        self.resets += 1
        self.seen_devices.append(device)

    def max_memory_allocated(self, device: object = None) -> int:
        self.seen_devices.append(device)
        return self.allocated

    def max_memory_reserved(self, device: object = None) -> int:
        self.seen_devices.append(device)
        return self.reserved

    def empty_cache(self) -> None:
        self.empties += 1

    def only_device(self) -> bool:
        """True when every device-scoped call named the same single device."""
        return len(set(map(repr, self.seen_devices))) == 1 and self.seen_devices != []


@dataclass
class _Forward:
    """A forward closure that records the canvases it was asked for, and can fail."""

    oom_at: int | None = None
    failure: BaseException | None = None
    seen: list[int] = field(default_factory=list)

    def __call__(self, canvas_tokens: int) -> None:
        self.seen.append(canvas_tokens)
        if self.failure is not None:
            raise self.failure
        if self.oom_at is not None and canvas_tokens >= self.oom_at:
            raise _FakeOutOfMemory("CUDA out of memory. Tried to allocate 4.30 GiB")


def test_ladder_measures_every_canvas_when_they_all_fit() -> None:
    # Given: a device that can hold the whole ladder.
    forward, cuda = _Forward(), _FakeCuda()
    # When: the ladder runs.
    measurements, warnings, unavailable = cr.run_ladder(forward, cuda)
    # Then: every requested canvas is measured, with nothing reported unavailable.
    assert [row.canvas_tokens for row in measurements] == list(cr.CANVAS_LADDER)
    assert warnings == ()
    assert unavailable == ()
    assert forward.seen == [canvas for canvas in cr.CANVAS_LADDER for _ in range(cr.FORWARD_REPETITIONS)]


def test_ladder_stops_at_the_first_out_of_memory_and_records_the_rest() -> None:
    # Given: a device that cannot hold 16384 tokens.
    forward, cuda = _Forward(oom_at=16_384), _FakeCuda()
    # When: the ladder runs.
    measurements, warnings, unavailable = cr.run_ladder(forward, cuda)
    # Then: the canvases that fit are measured, the rest are recorded as unavailable.
    assert [row.canvas_tokens for row in measurements] == [512, 4096]
    assert unavailable == (16_384, 32_768)
    assert any("16384" in warning and "out_of_memory" in warning for warning in warnings)
    # The cache is released after every rung and again on the OOM path, so at least one release
    # happened rather than the allocator holding the failed rung's blocks.
    assert cuda.empties >= 1


def test_out_of_memory_does_not_abort_the_rank_before_smaller_canvases() -> None:
    # Given: a device that only holds the smallest canvas.
    forward, cuda = _Forward(oom_at=4096), _FakeCuda()
    # When: the ladder runs.
    measurements, _warnings, unavailable = cr.run_ladder(forward, cuda)
    # Then: the rank still reports what it did measure rather than failing outright.
    assert [row.canvas_tokens for row in measurements] == [512]
    assert unavailable == (4096, 16_384, 32_768)


def test_a_non_memory_failure_still_propagates() -> None:
    # Given: a device failing for a reason that is not memory pressure.
    forward, cuda = _Forward(failure=ValueError("device lost")), _FakeCuda()
    # When / Then: the rank fails loudly rather than mislabelling it as an unavailable canvas.
    with pytest.raises(ValueError, match="device lost"):
        _ = cr.run_ladder(forward, cuda)


def test_backward_is_reported_as_unmeasured_when_no_step_is_supplied() -> None:
    # Given: no gradient-step measurement.
    forward, cuda = _Forward(), _FakeCuda()
    # When: the ladder runs.
    measurements, _warnings, _unavailable = cr.run_ladder(forward, cuda)
    # Then: every row explains the omission rather than reporting a zero.
    assert measurements
    for row in measurements:
        assert row.backward_seconds is None
        assert row.backward_not_measured_reason == cr.BACKWARD_NOT_MEASURED


def test_backward_is_measured_when_a_step_is_supplied() -> None:
    # Given: a step that reports a duration.
    forward, cuda = _Forward(), _FakeCuda()
    # When: the ladder runs with that step.
    measurements, _warnings, _unavailable = cr.run_ladder(forward, cuda, backward_step=lambda _canvas: 1.25)
    # Then: the duration is recorded and the omission reason is cleared.
    assert measurements
    for row in measurements:
        assert row.backward_seconds == 1.25
        assert row.backward_not_measured_reason is None


def test_a_non_positive_backward_duration_is_refused() -> None:
    # Given: a step reporting an impossible duration.
    forward, cuda = _Forward(), _FakeCuda()
    # When / Then: the measurement is refused rather than recorded as instantaneous.
    with pytest.raises(ValueError, match="non-positive"):
        _ = cr.run_ladder(forward, cuda, backward_step=lambda _canvas: 0.0)


def test_peak_memory_is_reset_for_each_canvas() -> None:
    # Given: a device that can hold the whole ladder.
    forward, cuda = _Forward(), _FakeCuda()
    # When: the ladder runs.
    _ = cr.run_ladder(forward, cuda)
    # Then: each canvas row describes its own peak, not a running high-water mark.
    assert cuda.resets == len(cr.CANVAS_LADDER)


def test_a_canvas_with_no_positive_reading_is_not_recorded() -> None:
    # Given: a device reporting no allocated memory.
    forward, cuda = _Forward(), _FakeCuda(allocated=0)
    # When: the ladder runs.
    measurements, warnings, unavailable = cr.run_ladder(forward, cuda)
    # Then: nothing is invented; the canvas is reported unavailable.
    assert measurements == ()
    assert unavailable == cr.CANVAS_LADDER
    assert any("measurement_not_positive" in warning for warning in warnings)


def test_repetitions_must_be_positive() -> None:
    # Given: a request for zero repetitions.
    forward, cuda = _Forward(), _FakeCuda()
    # When / Then: it is refused rather than dividing by zero.
    with pytest.raises(ValueError, match="repetitions"):
        _ = cr.measure_canvas(forward, cuda, 512, repetitions=0)


def test_measured_rows_satisfy_the_published_contract() -> None:
    # Given: a measured ladder.
    forward, cuda = _Forward(), _FakeCuda()
    measurements, _warnings, _unavailable = cr.run_ladder(forward, cuda)
    # When / Then: every row validates against the contract that the job publishes.
    assert measurements
    for row in measurements:
        assert row.peak_reserved_bytes >= row.peak_allocated_bytes
        assert row.forward_tokens_per_second > 0
        assert row.canvas_tokens in cr.CANVAS_LADDER


def test_the_forward_is_called_once_per_repetition() -> None:
    # Given: a device holding one canvas.
    forward, cuda = _Forward(), _FakeCuda()
    # When: a single canvas is measured with an explicit repetition count.
    outcome = cr.measure_canvas(forward, cuda, 4096, repetitions=5)
    # Then: the timing loop ran exactly that many forwards, all at the requested size.
    assert outcome.measurement is not None
    assert forward.seen == [4096] * 5
    assert outcome.measurement.canvas_tokens == 4096


def test_a_non_positive_forward_rate_is_not_recorded() -> None:
    # Given: a device whose clock does not advance, so no rate can be derived.
    forward, cuda = _Forward(), _FakeCuda()
    # When: the measurement is taken with the clock frozen.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cr.time, "perf_counter", lambda: 0.0)
        outcome = cr.measure_canvas(forward, cuda, 512)
    # Then: an unmeasurable rate is reported as unavailable, never as zero or infinite.
    assert outcome.measurement is None
    assert outcome.unavailable_reason == "measurement_not_positive"


def test_an_optimizer_step_is_timed_only_when_supplied() -> None:
    # Given: a device and one canvas.
    forward, cuda = _Forward(), _FakeCuda()
    # When: no optimizer step is supplied, then one that reports a duration.
    without = cr.measure_canvas(forward, cuda, 512, backward_step=lambda _: 0.5)
    calls: list[int] = []
    with_step = cr.measure_canvas(
        forward, cuda, 512, backward_step=lambda _: 0.5,
        optimizer_step=lambda canvas: calls.append(canvas) or 0.25,
    )
    # Then: the absent step carries a reason, and the supplied one is recorded with its duration.
    assert without.measurement is not None
    assert without.measurement.optimizer_step_seconds is None
    assert without.measurement.optimizer_step_not_measured_reason is not None
    assert with_step.measurement is not None
    assert with_step.measurement.optimizer_step_seconds == 0.25
    assert with_step.measurement.optimizer_step_not_measured_reason is None
    assert calls == [512]


def test_a_caller_supplied_optimizer_reason_is_preserved() -> None:
    # Given: a rank that legitimately cannot time an optimizer step.
    forward, cuda = _Forward(), _FakeCuda()
    # When: the reason is supplied rather than defaulted.
    outcome = cr.measure_canvas(
        forward, cuda, 512, backward_step=lambda _: 0.5,
        optimizer_not_measured_reason="sharded training is not a single-GPU step",
    )
    # Then: the specific explanation survives instead of the generic default.
    assert outcome.measurement is not None
    assert outcome.measurement.optimizer_step_not_measured_reason == "sharded training is not a single-GPU step"


def test_the_selection_size_is_recorded_from_the_backward_step() -> None:
    # Given: a rank whose corruption selected 41 positions on the last canvas.
    forward, cuda = _Forward(), _FakeCuda()
    # When: the measurement is taken with a selection reader.
    outcome = cr.measure_canvas(
        forward, cuda, 512, backward_step=lambda _: 0.5, selected_tokens=lambda: 41
    )
    # Then: the evidence that the loss was non-vacuous travels with the row.
    assert outcome.measurement is not None
    assert outcome.measurement.selected_tokens == 41


def test_no_selection_is_reported_when_no_backward_was_taken() -> None:
    # Given: a forward-only row.
    forward, cuda = _Forward(), _FakeCuda()
    # When / Then: no selection is claimed, rather than the reader being consulted.
    outcome = cr.measure_canvas(forward, cuda, 512, selected_tokens=lambda: 41)
    assert outcome.measurement is not None
    assert outcome.measurement.selected_tokens == 0


def test_a_ladder_past_its_deadline_records_the_remaining_canvases() -> None:
    # Given: a clock already beyond the budget.
    forward, cuda = _Forward(), _FakeCuda()
    # When: the ladder runs with an exhausted deadline.
    measurements, warnings, unavailable = cr.run_ladder(
        forward, cuda, deadline=0.0, clock=lambda: 10.0
    )
    # Then: nothing is measured and every canvas is named as budget-blocked, not dropped.
    assert measurements == ()
    assert unavailable == cr.CANVAS_LADDER
    assert len(warnings) == len(cr.CANVAS_LADDER)
    assert all(cr.TIME_BUDGET_EXHAUSTED in warning for warning in warnings)
    assert forward.seen == []


def test_device_scoped_calls_name_the_device_they_measured() -> None:
    # Given: a rank whose work runs on a GPU other than the process-default device.
    forward, cuda = _Forward(), _FakeCuda()
    device = "cuda:3"

    # When: one canvas is measured with an explicit device.
    outcome = cr.measure_canvas(forward, cuda, 512, device=device, backward_step=lambda _: 0.5)

    # Then: every device-scoped call named it, so the peak reading describes that GPU.
    # Calling these without a device reports the current device's allocations, which on another
    # GPU is zero -- and a zero peak was read as "unmeasurable" instead of as this bug.
    assert outcome.measurement is not None
    assert cuda.only_device()
    assert set(map(repr, cuda.seen_devices)) == {"'cuda:3'"}


def test_ladder_forwards_the_device_to_every_canvas() -> None:
    # Given: a ladder run on an explicit device.
    forward, cuda = _Forward(), _FakeCuda()

    # When: the ladder measures its canvases.
    measurements, _warnings, _unavailable = cr.run_ladder(
        forward, cuda, device="cuda:5", backward_step=lambda _: 0.5
    )

    # Then: no canvas was measured against the wrong device's counters.
    assert measurements
    assert cuda.only_device()
    assert set(map(repr, cuda.seen_devices)) == {"'cuda:5'"}


def test_a_warmup_call_precedes_the_timed_canvas_and_is_excluded_from_the_peak() -> None:
    # Given: a device that counts cache releases.
    forward, cuda = _Forward(), _FakeCuda()

    # When: a canvas is measured with warm-up enabled.
    outcome = cr.measure_canvas(forward, cuda, 512, repetitions=2, warmup=True)

    # Then: the closure ran once extra, the peak was reset after that warm-up, and the warm-up's
    # blocks were released so they cannot inflate the timed section or the next rung.
    assert outcome.measurement is not None
    assert forward.seen == [512, 512, 512]
    assert cuda.resets == 1
    assert cuda.empties == 1


def test_warmup_is_off_by_default() -> None:
    # Given: a caller that did not ask for warm-up.
    forward, cuda = _Forward(), _FakeCuda()

    # When: a canvas is measured.
    outcome = cr.measure_canvas(forward, cuda, 512, repetitions=2)

    # Then: only the timed repetitions ran.
    assert outcome.measurement is not None
    assert forward.seen == [512, 512]


def test_each_rung_releases_its_cache_before_the_next() -> None:
    # Given: a two-rung ladder.
    forward, cuda = _Forward(), _FakeCuda()

    # When: both rungs are measured.
    measurements, _warnings, _unavailable = cr.run_ladder(forward, cuda, ladder=(512, 4096))

    # Then: the allocator cache is released once per rung, so a larger rung is not charged for the
    # smaller rung's cached blocks -- the accumulation that pushed the small arms into CUDA OOM.
    assert len(measurements) == 2
    assert cuda.empties == 2


def test_a_warmup_out_of_memory_is_recorded_as_unavailable() -> None:
    # Given: a canvas too large for the warm-up pass.
    forward, cuda = _Forward(oom_at=512), _FakeCuda()

    # When: the canvas is measured with warm-up enabled.
    outcome = cr.measure_canvas(forward, cuda, 512, warmup=True)

    # Then: it is recorded, not raised, and not retried cold either.
    assert outcome.measurement is None
    assert outcome.unavailable_reason is not None and "out_of_memory" in outcome.unavailable_reason


def test_a_warmup_out_of_memory_stops_the_ladder_without_continuing() -> None:
    # Given: a device that cannot even run the extra warm-up pass on the second rung.
    class _WarmupFailsAt4096(_Forward):
        def __call__(self, canvas_tokens: int) -> None:
            if canvas_tokens == 4096 and not getattr(self, "_warmed", False):
                self._warmed = True
                raise _FakeOutOfMemory("CUDA out of memory: warm-up pass does not fit")
            super().__call__(canvas_tokens)

    forward, cuda = _WarmupFailsAt4096(), _FakeCuda()

    # When: the ladder runs with warm-up enabled.
    measurements, warnings, unavailable = cr.run_ladder(forward, cuda, ladder=(512, 4096, 16384), warmup=True)

    # Then: the earlier rung survives, the failing rung is recorded, and nothing is attempted after
    # it -- continuing on a context that has just failed an allocation is what produced an illegal
    # memory access on the real device.
    assert [row.canvas_tokens for row in measurements] == [512]
    assert unavailable == (4096, 16384)
    assert any("out_of_memory" in warning for warning in warnings)
