"""Efficiency-probe tests: inference-mode accounting, OOM-as-a-row, warm-up exclusion.

These drive the runner with adapter fakes and a fake device-memory surface, so they pin the control
flow that decides what a row contains -- not the CUDA timing itself, which only a real device can
produce. The one property they can assert about the device path without a GPU is that the timed
forward ran under ``torch.inference_mode()``; an adapter records that directly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from scale.experiments.nonlatent_iclr.efficiency import probe_inference as pp

#: The fields a row must carry, from the protocol's minimum. Locked here so a later edit that drops
#: one fails a test rather than producing a table with a hole in it.
REQUIRED_FIELDS = {
    "model", "family", "objective", "nfe", "context", "status", "wall_ms", "tokens_per_second",
    "peak_allocated_bytes", "peak_reserved_bytes", "analytic_state_bytes", "error",
}


@dataclass
class _FakeMemory:
    """A device-memory surface that reports a fixed peak and records the devices it was asked about.

    It deliberately keeps no call counters: a count cannot distinguish a reset that ran after the
    warm-up from one that ran before it, so the ordering contracts use :class:`_Trace` instead. A
    counter here would only invite a test that reports coverage it does not have.
    """

    allocated: int = 8_000_000_000
    reserved: int = 9_000_000_000
    seen_devices: list[str] = field(default_factory=list)

    def synchronize(self, device: str) -> None:
        self.seen_devices.append(device)

    def reset_peak_memory_stats(self, device: str) -> None:
        self.seen_devices.append(device)

    def max_memory_allocated(self, device: str) -> int:
        self.seen_devices.append(device)
        return self.allocated

    def max_memory_reserved(self, device: str) -> int:
        self.seen_devices.append(device)
        return self.reserved

    def empty_cache(self) -> None:
        return None


@pytest.fixture
def fake_memory(monkeypatch: pytest.MonkeyPatch) -> _FakeMemory:
    """Replace the one hardware seam so the memory calls are observable without a GPU."""
    memory = _FakeMemory()
    monkeypatch.setattr(pp, "device_memory", lambda device: memory)
    return memory


@dataclass
class _Identity:
    """The metadata every fake arm carries; the row is built from it."""

    model: str = "unit-arm"
    family: str = "associative_recall"
    objective: str = "masked_diffusion"
    nfe: int = 1
    analytic_weights_bytes: int = 2_000_000_000
    analytic_state_bytes: int = 1_048_576


@dataclass
class _SleepAdapter(_Identity):
    """A forward that sleeps a fixed time and records how it was called."""

    sleep_seconds: float = 0.004
    calls: list[int] = field(default_factory=list)
    inference_mode_seen: list[bool] = field(default_factory=list)

    def forward(self, context: int) -> None:
        self.calls.append(context)
        self.inference_mode_seen.append(torch.is_inference_mode_enabled())
        time.sleep(self.sleep_seconds)


@dataclass
class _OomAdapter(_Identity):
    """A forward that raises a genuine CUDA OOM at or above ``oom_at`` tokens."""

    oom_at: int = 0

    def forward(self, context: int) -> None:
        if context >= self.oom_at:
            raise torch.cuda.OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 14.00 GiB "
                "(GPU 0; 79.00 GiB total capacity; 60.00 GiB already allocated)"
            )


@dataclass
class _FaultAdapter(_Identity):
    """A forward that fails for a reason that is not memory pressure."""

    def forward(self, context: int) -> None:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


@dataclass
class _Trace:
    """One ordered log that a fake forward and a fake memory surface both append to.

    Two of this file's strongest contracts are about *order*, not counts: the peak reset must fall
    after the warm-up forwards and before the first timed one, and the ladder must release the cache
    between rungs. A call counter cannot see either -- a reset moved ahead of the warm-up still
    counts as one call, and an ``empty_cache`` the ladder never made is simply absent -- so the fakes
    write into a single interleaved log and the test reads positions out of it.
    """

    events: list[tuple[str, int | None]] = field(default_factory=list)

    def record(self, name: str, context: int | None = None) -> None:
        self.events.append((name, context))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def forward_indices(self, context: int | None = None) -> list[int]:
        return [
            index
            for index, (name, seen) in enumerate(self.events)
            if name == "forward" and (context is None or seen == context)
        ]


@dataclass
class _TracedAdapter(_Identity):
    """A forward that records itself into a shared trace instead of sleeping."""

    trace: _Trace = field(default_factory=_Trace)

    def forward(self, context: int) -> None:
        self.trace.record("forward", context)


@dataclass
class _TracedMemory:
    """A device-memory surface that records every call into the same trace."""

    trace: _Trace
    allocated: int = 8_000_000_000
    reserved: int = 9_000_000_000

    def synchronize(self, device: str) -> None:
        self.trace.record("sync")

    def reset_peak_memory_stats(self, device: str) -> None:
        self.trace.record("reset")

    def max_memory_allocated(self, device: str) -> int:
        self.trace.record("allocated")
        return self.allocated

    def max_memory_reserved(self, device: str) -> int:
        self.trace.record("reserved")
        return self.reserved

    def empty_cache(self) -> None:
        self.trace.record("empty")


@pytest.fixture
def traced(monkeypatch: pytest.MonkeyPatch) -> tuple[_Trace, _TracedMemory]:
    """Bind a shared trace and the memory surface that both write into it."""
    trace = _Trace()
    memory = _TracedMemory(trace=trace)
    monkeypatch.setattr(pp, "device_memory", lambda device: memory)
    return trace, memory


def test_a_measured_row_carries_every_required_field(fake_memory: _FakeMemory) -> None:
    # Given: an arm that runs instantly.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When: one cell is measured.
    row = pp.measure_cell(adapter, context=4096, device="cuda:0")
    # Then: the row is complete and labelled as a measurement, not an omission.
    assert REQUIRED_FIELDS <= set(row)
    assert row["status"] == pp.STATUS_OK
    assert row["error"] is None
    assert row["context"] == 4096
    assert row["model"] == "unit-arm"


def test_the_timed_forward_runs_under_inference_mode(fake_memory: _FakeMemory) -> None:
    # Given: an arm that records whether autograd was enabled around it.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When: a cell is measured with a warm-up and repeats.
    _ = pp.measure_cell(adapter, context=1024, device="cuda:0", warmup=2, repeats=3)
    # Then: every forward -- warm-up included -- saw inference mode on, so no autograd graph was
    # ever built and the timing is an inference timing rather than a training-shape one.
    assert adapter.inference_mode_seen == [True] * 5


def test_a_cuda_out_of_memory_becomes_a_row_with_its_error_text() -> None:
    # Given: an arm that cannot fit this context in the warm-up pass.
    adapter = _OomAdapter(oom_at=4096)
    # When: the cell is measured.
    row = pp.measure_cell(adapter, context=4096, device="cpu")
    # Then: it is recorded, not raised, and the row says which arm and context failed.
    assert row["status"] == pp.STATUS_OOM
    assert row["context"] == 4096
    assert row["model"] == "unit-arm"
    assert row["error"] is not None and "out of memory" in row["error"]
    # And no timing is invented for a rung that never completed.
    assert row["wall_ms"] is None
    assert row["tokens_per_second"] is None


def test_a_non_out_of_memory_exception_is_not_swallowed() -> None:
    # Given: an arm failing for a reason that is not memory pressure.
    adapter = _FaultAdapter()
    # When / Then: the defect propagates rather than being filed as "it did not fit".
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _ = pp.measure_cell(adapter, context=4096, device="cpu")


def test_the_ladder_keeps_the_oom_row_and_stops_there(tmp_path) -> None:
    # Given: a ladder whose second rung does not fit.
    adapter = _OomAdapter(oom_at=16_384)
    # When: the default ladder runs.
    rows = pp.probe_ladder(adapter, device="cpu", out_dir=tmp_path)
    # Then: the fitting rung is measured, the failing rung is present as an OOM row, and nothing
    # larger is attempted -- so the table shows four contexts attempted, not two.
    assert [row["context"] for row in rows] == [4096, 16_384]
    assert rows[0]["status"] == pp.STATUS_OK
    assert rows[1]["status"] == pp.STATUS_OOM


def test_the_ladder_measures_every_rung_when_they_all_fit(fake_memory: _FakeMemory, tmp_path) -> None:
    # Given: an arm that fits every context.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When: a two-rung ladder runs.
    rows = pp.probe_ladder(adapter, contexts=(1024, 2048), device="cuda:0", out_dir=tmp_path)
    # Then: both rungs are measured and their contexts match what was asked, in order.
    assert [row["context"] for row in rows] == [1024, 2048]
    assert all(row["status"] == pp.STATUS_OK for row in rows)


def test_tokens_per_second_matches_a_hand_computed_value(fake_memory: _FakeMemory) -> None:
    # Given: an arm whose per-forward cost is a fixed 4 ms, so the rate is context / 4 ms regardless
    # of how many repeats are taken.
    adapter = _SleepAdapter(sleep_seconds=0.004)
    # When: one cell is measured.
    row = pp.measure_cell(adapter, context=2000, device="cuda:0", warmup=1, repeats=5)
    # Then: the reported rate is the hand-computed 2000 / 0.004 = 500k tokens/s, within the slack a
    # real sleep needs.
    expected = 2000 / 0.004
    assert row["tokens_per_second"] == pytest.approx(expected, rel=0.3)
    # The wall time covers exactly the five timed repeats (~20 ms), no more.
    assert row["wall_ms"] == pytest.approx(5 * 0.004 * 1000, rel=0.3)


def test_warmup_calls_are_exactly_the_argument_and_are_not_timed(fake_memory: _FakeMemory) -> None:
    # Given: an arm with a known 10 ms cost per forward.
    adapter = _SleepAdapter(sleep_seconds=0.01)
    # When: the cell is measured with four warm-ups and two timed repeats.
    row = pp.measure_cell(adapter, context=1000, device="cuda:0", warmup=4, repeats=2)
    # Then: the forward ran exactly 4 + 2 times, at the requested context.
    assert adapter.calls == [1000] * 6
    # And the rate reflects two timed repeats, not six forwards: 1000 / 0.01 = 100k, whereas billing
    # the warm-ups would give 1000 * 2 / (6 * 0.01) = ~33k. The band excludes the latter.
    assert row["tokens_per_second"] == pytest.approx(1000 / 0.01, rel=0.3)
    # The timed wall time is two repeats (~20 ms), not six (~60 ms).
    assert row["wall_ms"] is not None and row["wall_ms"] <= 2 * 0.01 * 1000 * 1.5


def test_the_peaks_are_reset_after_the_warmups_and_before_the_timed_forwards(
    traced: tuple[_Trace, _TracedMemory],
) -> None:
    # Given: an arm measured with three warm-ups and two timed repeats, its forwards and the memory
    # calls landing in one ordered trace.
    trace, _ = traced
    adapter = _TracedAdapter(trace=trace)
    # When: the cell is measured.
    _ = pp.measure_cell(adapter, context=1024, device="cuda:0", warmup=3, repeats=2)
    # Then: the peaks were reset exactly once -- and crucially *where*: after the three warm-up
    # forwards and before either timed forward. Counting the reset is not enough; moving the reset
    # ahead of the warm-up loop still resets once, yet lets the warm-up's allocation inflate the very
    # reading the reset exists to keep clean. The ordered trace is the only instrument that sees it.
    names = trace.names()
    assert names.count("reset") == 1
    reset_index = names.index("reset")
    assert len([name for name in names[:reset_index] if name == "forward"]) == 3
    assert len([name for name in names[reset_index:] if name == "forward"]) == 2
    # And the warm-up's blocks were handed back before the reset, not after it.
    assert "empty" in names[:reset_index]


def test_the_ladder_releases_each_rungs_cache_before_the_next_rung(
    traced: tuple[_Trace, _TracedMemory], tmp_path
) -> None:
    # Given: a two-rung ladder whose rungs are distinguishable at the forward.
    trace, _ = traced
    adapter = _TracedAdapter(trace=trace)
    # When: the ladder runs.
    rows = pp.probe_ladder(adapter, contexts=(1024, 2048), device="cuda:0", out_dir=tmp_path)
    # Then: both rungs were measured, and an empty_cache fell between the last small forward and the
    # first large one. Without that release the allocator still holds the 1024 blocks when the 2048
    # rung warms up, so a rung that fits on its own could be recorded as a false "oom" row -- the one
    # entry the table cannot afford, because the row is the reader's only evidence of what failed.
    assert [row["status"] for row in rows] == [pp.STATUS_OK, pp.STATUS_OK]
    last_small = max(trace.forward_indices(1024))
    first_large = min(trace.forward_indices(2048))
    between = [name for name, _ in trace.events[last_small + 1 : first_large]]
    assert "empty" in between


def test_peak_allocated_bytes_clears_the_analytic_weights(fake_memory: _FakeMemory) -> None:
    # Given: an arm whose analytic parameter bytes are 2 GB and a surface reporting an 8 GB allocated
    # peak against 9 GB reserved.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When: the cell is measured.
    row = pp.measure_cell(adapter, context=4096, device="cuda:0")
    # Then: the allocated reading is the surface's own number and cannot be below the weights that
    # must be resident for the forward to run at all -- the accounting sanity that catches a
    # mislabelled or mis-scoped reading.
    assert row["peak_allocated_bytes"] == fake_memory.allocated
    assert row["peak_allocated_bytes"] >= adapter.analytic_weights_bytes
    # And the reserved reading is the surface's *reserved* number, not a copy of the allocated one:
    # the fake reports 9 GB reserved against 8 GB allocated, so a row that echoed the allocated value
    # would erase the allocator headroom the table's memory story rests on.
    assert row["peak_reserved_bytes"] == fake_memory.reserved
    assert row["peak_reserved_bytes"] >= row["peak_allocated_bytes"]


def test_peak_reserved_is_clamped_up_when_the_surface_reports_less_than_allocated(
    fake_memory: _FakeMemory,
) -> None:
    # Given: a surface whose two readings disagree, reserved below allocated -- which no allocator
    # should report, but which a mis-scoped or misread surface can.
    fake_memory.allocated = 8_000_000_000
    fake_memory.reserved = 7_000_000_000
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When: the cell is measured.
    row = pp.measure_cell(adapter, context=4096, device="cuda:0")
    # Then: the reserved reading is raised to the allocated one, so the row cannot claim the run
    # reserved less than it had in use.
    assert row["peak_reserved_bytes"] == fake_memory.allocated


def test_the_device_is_named_on_every_memory_call(fake_memory: _FakeMemory) -> None:
    # Given: a rank whose work runs on a GPU other than the process default.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When: the cell is measured with an explicit device.
    _ = pp.measure_cell(adapter, context=1024, device="cuda:3", warmup=1, repeats=2)
    # Then: every device-scoped call named it, so the peak reading describes that GPU and not the
    # process default -- which on another GPU reports zero and reads as unmeasurable.
    assert fake_memory.seen_devices and set(fake_memory.seen_devices) == {"cuda:3"}


def test_repeats_must_be_positive() -> None:
    # Given: a request for zero repeats.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When / Then: it is refused rather than yielding a rate from an empty timed section.
    with pytest.raises(pp.ProbeRefusal, match="repeats"):
        _ = pp.measure_cell(adapter, context=1024, device="cpu", repeats=0)


def test_warmup_must_not_be_negative() -> None:
    # Given: a request for a negative number of warm-ups.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When / Then: it is refused rather than silently treated as zero.
    with pytest.raises(pp.ProbeRefusal, match="warmup"):
        _ = pp.measure_cell(adapter, context=1024, device="cpu", warmup=-1)


def test_context_must_be_positive() -> None:
    # Given: a zero-length context.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When / Then: it is refused rather than timed as a degenerate pass.
    with pytest.raises(pp.ProbeRefusal, match="context"):
        _ = pp.measure_cell(adapter, context=0, device="cpu")


def test_the_ladder_must_strictly_increase(fake_memory: _FakeMemory, tmp_path) -> None:
    # Given: a ladder that repeats a rung and then goes backwards.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When / Then: it is refused, because stop-on-first-OOM is only meaningful on a monotone ladder.
    with pytest.raises(pp.ProbeRefusal, match="strictly increase"):
        _ = pp.probe_ladder(adapter, contexts=(4096, 4096), device="cuda:0", out_dir=tmp_path)


def test_an_empty_ladder_is_refused(tmp_path) -> None:
    # Given: a ladder that names no context at all.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When / Then: it is refused rather than writing an artifact whose empty row list would read as
    # "every rung fit" -- a table with no rows must be a loud error, not a silent empty table.
    with pytest.raises(pp.ProbeRefusal, match="at least one context"):
        _ = pp.probe_ladder(adapter, contexts=(), device="cpu", out_dir=tmp_path)


def test_a_clock_that_does_not_advance_is_refused(
    fake_memory: _FakeMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: an arm whose forwards are instantaneous on a frozen clock, so the timed section spans
    # zero seconds.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    monkeypatch.setattr(pp.time, "perf_counter", lambda: 5.0)
    # When / Then: the cell is refused rather than yielding an infinite tokens-per-second from a
    # zero-length interval. A fabricated rate is worse than a missing one, so this is a named refusal
    # and not a row with an infinite field in it.
    with pytest.raises(pp.ProbeRefusal, match="clock did not advance"):
        _ = pp.measure_cell(adapter, context=1024, device="cuda:0")


def test_the_module_imports_with_torch_blocked() -> None:
    # Given: a host where torch cannot be imported at all, modeled with a meta-path finder that
    # refuses it. This is the CPU-suite case: every module under this package is imported on a host,
    # and a module-scope torch or torch.cuda import would make the whole package unimportable there.
    module_path = Path(pp.__file__)
    script = (
        "import importlib.util, sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch blocked by test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"spec = importlib.util.spec_from_file_location('probe_isolated', {str(module_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "assert module.CONTEXTS == (4096, 16384, 32768, 65536)\n"
        "print('imported-without-torch')\n"
    )
    # When: the module is loaded in a subprocess with torch blocked before its bytecode runs.
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    # Then: it imports and exposes its surface -- so at module scope it depends on neither torch nor
    # torch.cuda. (The measurement path still imports torch inside measure_cell; that is deliberate
    # and is what keeps the import contract above true for callers who only want the protocol types.)
    assert result.returncode == 0, result.stderr
    assert "imported-without-torch" in result.stdout


def test_a_non_out_of_memory_exception_is_not_swallowed_by_the_ladder(
    fake_memory: _FakeMemory, tmp_path
) -> None:
    # Given: a ladder whose only rung faults.
    adapter = _FaultAdapter()
    # When / Then: the fault aborts the ladder loudly rather than ending it as an OOM stop.
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _ = pp.probe_ladder(adapter, contexts=(1024,), device="cuda:0", out_dir=tmp_path)


def test_the_artifact_records_the_rows_and_whether_the_ladder_stopped(
    fake_memory: _FakeMemory, tmp_path
) -> None:
    # Given: a ladder where only the first two rungs fit.
    adapter = _OomAdapter(oom_at=4096)
    # When: it runs and writes its artifact.
    rows = pp.probe_ladder(adapter, contexts=(1024, 2048, 4096, 8192), device="cuda:0", out_dir=tmp_path)
    # Then: the file holds the rows, the measured contexts, and the early stop -- so the table's
    # scope is stated in the artifact itself rather than inferred from what is absent.
    payload = json.loads((tmp_path / pp.PROBE_FILENAME).read_text(encoding="utf-8"))
    assert payload["rows"] == rows
    assert payload["contexts_requested"] == [1024, 2048, 4096, 8192]
    assert payload["contexts_measured"] == [1024, 2048]
    assert payload["stopped_at_oom"] is True


def test_the_artifact_reports_a_complete_ladder_as_not_stopped(
    fake_memory: _FakeMemory, tmp_path
) -> None:
    # Given: a ladder that fits end to end.
    adapter = _SleepAdapter(sleep_seconds=0.0)
    # When: it runs and writes its artifact.
    _ = pp.probe_ladder(adapter, contexts=(1024, 2048), device="cuda:0", out_dir=tmp_path)
    # Then: nothing claims an early stop, so a complete table is distinguishable from a truncated
    # one by the artifact alone.
    payload = json.loads((tmp_path / pp.PROBE_FILENAME).read_text(encoding="utf-8"))
    assert payload["stopped_at_oom"] is False
    assert payload["contexts_measured"] == [1024, 2048]
