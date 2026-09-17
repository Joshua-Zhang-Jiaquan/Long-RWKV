"""Inference-mode probe runner for the paper's efficiency matrix (contribution 1).

Every number in the efficiency table has to be an *inference* number. The training-shape
calibration this program already has (``qualification.calibration_runtime``) builds an autograd
graph and times ``loss.backward()``, so pricing inference from it overstates the cost by whatever
the backward and optimizer add -- and its 4096 rung OOMed for the loop arm because activation and
gradient state were live at the same time. This module therefore measures the forward alone, under
``torch.inference_mode()``, so no autograd graph is ever built.

Two behaviours are load-bearing for the table being defensible rather than merely present:

* **An OOM is a row, not an exception.** The reported protocol forbids silent truncation. A ladder
  that stopped at the first OOM and dropped the larger rungs would let a table of four contexts be
  read as "four contexts were measured" when two were. So the failing rung is recorded with
  ``status == "oom"`` and its error text, :func:`probe_ladder` stops there, and the surviving rows
  keep it.
* **Only a genuine CUDA OOM is recorded as an OOM.** Catching a broader ``RuntimeError`` would let
  a real defect -- an illegal memory access, a device-side assert -- be filed as "it did not fit",
  which is exactly the silent mislabelling this table cannot afford. Any other exception
  propagates.

The probe is deliberately hardware-agnostic in the same way ``calibration_runtime`` is: the caller
supplies an *adapter* whose ``forward`` performs one full-context call, and the device-memory
surface is resolved by :func:`device_memory`, which is the one seam a CPU-only test replaces.
Neither ``torch`` nor ``torch.cuda`` is imported at module scope, because the CPU suite imports
every module under this package and a module-scope CUDA import would break that import.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Protocol, cast

#: The four contexts the efficiency matrix reports, smallest first. The ladder is monotonically
#: increasing because it stops at the first rung that does not fit: a shuffled ladder would stop at
#: the wrong rung and the remaining, larger contexts would never be attempted.
CONTEXTS: Final = (4096, 16384, 32768, 65536)

#: One warm-up forward per rung by default, then five timed repeats. The default is a parameter so
#: a caller can tighten the loop, not so a caller can silently drop the warm-up.
DEFAULT_WARMUP: Final = 1
DEFAULT_REPEATS: Final = 5

STATUS_OK: Final = "ok"
STATUS_OOM: Final = "oom"

#: The artifact :func:`probe_ladder` writes beside its return value, so a reviewer can diff the
#: table against the run that produced it rather than trusting a copy-paste of stdout.
PROBE_FILENAME: Final = "inference_probe.json"


class ProbeRefusal(ValueError):
    """The probe was asked for a measurement that cannot describe a real inference pass.

    Raised for a non-positive context or repeat count, a negative warm-up, an unsorted ladder, and
    a clock that did not advance. Each is a caller or environment defect that must fail loudly:
    recording it as a slow-but-finite measurement would put a number in the efficiency table that
    nothing produced.
    """


class DeviceMemory(Protocol):
    """The ``torch.cuda`` surface this module uses; every call names the device explicitly.

    Binding the device matters here for the same reason it does in the calibration runtime: calling
    these without a device reads the *process-default* device, so a rank working on another GPU
    would report a zero peak, and a zero peak reads as "unmeasurable" rather than as the bug it is.
    """

    def synchronize(self, device: str) -> None: ...
    def reset_peak_memory_stats(self, device: str) -> None: ...
    def max_memory_allocated(self, device: str) -> int: ...
    def max_memory_reserved(self, device: str) -> int: ...
    def empty_cache(self) -> None: ...


class _NoDeviceMemory:
    """The stand-in for a device with no allocator (a CPU run, or a test that wants no CUDA).

    It reports zero for both peaks rather than raising, so a caller that only wanted the timing
    path -- a CPU correctness run -- is not forced to touch CUDA. A zero peak is *not* a measurement
    and the row that carries it must be read as a CPU timing, never as a device allocation.
    """

    def synchronize(self, device: str) -> None:
        return None

    def reset_peak_memory_stats(self, device: str) -> None:
        return None

    def max_memory_allocated(self, device: str) -> int:
        return 0

    def max_memory_reserved(self, device: str) -> int:
        return 0

    def empty_cache(self) -> None:
        return None


_NO_DEVICE_MEMORY: Final[DeviceMemory] = _NoDeviceMemory()


def device_memory(device: str) -> DeviceMemory:
    """Resolve the memory surface for ``device``, importing ``torch.cuda`` only for a CUDA device.

    This is the module's single hardware seam. A test replaces it to observe the reset-before-timing
    and peak-reading calls without a GPU; production relies on it returning the real ``torch.cuda``
    for a CUDA device and the zero surface otherwise. ``torch`` is imported *here*, not at module
    scope, so importing this module on a CPU-only host cannot fail.
    """
    if device.startswith("cuda"):
        import torch  # local: a module-scope CUDA import would break CPU-only importers

        return cast(DeviceMemory, torch.cuda)
    return _NO_DEVICE_MEMORY


class InferenceAdapter(Protocol):
    """One efficiency arm: how to run it, and the identity and accounting the row must carry.

    ``forward`` performs exactly one full-context masked inference pass and returns nothing the
    probe needs; the probe only times it and reads the device peaks. Carrying ``model``, ``family``,
    ``objective`` and ``nfe`` on the adapter -- rather than passing them to the runner -- keeps a
    row impossible to build for an arm whose identity was never stated, which is how a table row
    silently stops describing the run that produced it.
    """

    #: The checkpoint or arm label the row is about.
    model: str
    #: The task family this arm is evaluated on, matching the frozen protocol's families.
    family: str
    #: The training objective the arm was trained under.
    objective: str
    #: Number of function evaluations the sampler performs per generated token.
    nfe: int
    #: Analytic parameter bytes, computed from the geometry; the floor a peak reading must clear.
    analytic_weights_bytes: int
    #: Analytic recurrent-state bytes, CONSTANT in context length -- the point the table makes.
    analytic_state_bytes: int

    def forward(self, context: int) -> None:
        """Run one inference pass over ``context`` tokens, on the device the adapter owns."""
        ...


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ProbeRefusal(f"{name} must be positive, got {value}")


def _require_context(context: int) -> None:
    _require_positive(context, "context")


def _row_identity(adapter: InferenceAdapter, context: int) -> dict[str, object]:
    """The part of a row that is known before any device work, shared by the ok and oom cases.

    The OOM row carries the same identity as a measured row, so a reader can tell *which* arm and
    which context failed to fit without cross-referencing anything else.
    """
    return {
        "model": adapter.model,
        "family": adapter.family,
        "objective": adapter.objective,
        "nfe": adapter.nfe,
        "context": context,
        "status": STATUS_OK,
        "wall_ms": None,
        "tokens_per_second": None,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "analytic_weights_bytes": adapter.analytic_weights_bytes,
        "analytic_state_bytes": adapter.analytic_state_bytes,
        "error": None,
    }


def measure_cell(
    adapter: InferenceAdapter,
    *,
    context: int,
    device: str,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, object]:
    """Time one adapter at one context under inference mode, or record why it did not fit.

    The warm-up forwards run first and are *not* timed: the first touch of a context pays for cold
    kernels, autotuner selection and allocator growth, and this program has already been bitten once
    by billing that one-time cost to a measured rung. The peak counters are reset only after the
    warm-up, so the warm-up's own allocation cannot inflate the timed reading either.

    A ``torch.cuda.OutOfMemoryError`` from any of those forwards becomes a row with
    ``status == "oom"`` and the error text; every other exception propagates, because a CUDA fault
    recorded as "out of memory" would hide a defect behind a plausible table entry.
    """
    _require_context(context)
    if warmup < 0:
        raise ProbeRefusal(f"warmup must not be negative, got {warmup}")
    _require_positive(repeats, "repeats")

    import torch  # local: keeps this module importable on a host without torch

    memory = device_memory(device)
    row = _row_identity(adapter, context)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                adapter.forward(context)
        memory.synchronize(device)
        # Release the warm-up's blocks before measuring, so the timed envelope is the rung's own
        # footprint rather than the warm-up plus it. Then reset the peaks so they describe only the
        # timed section.
        memory.empty_cache()
        memory.reset_peak_memory_stats(device)
        memory.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                adapter.forward(context)
        memory.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_allocated = int(memory.max_memory_allocated(device))
        peak_reserved = int(memory.max_memory_reserved(device))
    except torch.cuda.OutOfMemoryError as error:
        memory.empty_cache()
        row["status"] = STATUS_OOM
        row["error"] = f"{type(error).__name__}: {error}"
        return row

    if elapsed <= 0.0:
        # A clock that did not advance cannot yield a rate; inventing one would be worse than
        # refusing, so this is a named refusal rather than a zero-token-per-second row.
        raise ProbeRefusal(f"the clock did not advance while measuring context {context}")

    tokens = context * repeats
    row["wall_ms"] = elapsed * 1000.0
    row["tokens_per_second"] = tokens / elapsed
    row["peak_allocated_bytes"] = peak_allocated
    # The allocator never reserves less than it hands out; recording max() keeps the reserved
    # reading from ever contradicting the allocated one on a surface that reports them separately.
    row["peak_reserved_bytes"] = max(peak_reserved, peak_allocated)
    return row


def _require_ladder(contexts: Iterable[int]) -> tuple[int, ...]:
    """Refuse a ladder that is not a strictly increasing sequence of positive contexts.

    The stop-on-first-OOM contract is only meaningful on a monotone ladder: on a shuffled one the
    probe would stop at whichever rung happened to come first and never attempt the rungs after it,
    and the resulting table would understate the range the arm actually covers.
    """
    ordered = tuple(contexts)
    if not ordered:
        raise ProbeRefusal("the ladder must name at least one context")
    previous = 0
    for context in ordered:
        _require_context(context)
        if context <= previous:
            raise ProbeRefusal(
                f"the ladder must strictly increase; {context} follows {previous}"
            )
        previous = context
    return ordered


def probe_ladder(
    adapter: InferenceAdapter,
    *,
    contexts: Iterable[int] = CONTEXTS,
    device: str,
    out_dir,
) -> list[dict[str, object]]:
    """Measure every rung in order, stopping at the first OOM but keeping that rung's row.

    The stopping row is what makes the table honest: it says explicitly "this context did not fit"
    instead of the larger rungs simply being absent, so a reader counts four contexts attempted, not
    two measured. The rows -- OOM row included -- are written to ``out_dir`` as the artifact the
    paper table is built from.
    """
    ladder = _require_ladder(contexts)
    memory = device_memory(device)
    rows: list[dict[str, object]] = []
    for context in ladder:
        row = measure_cell(adapter, context=context, device=device)
        rows.append(row)
        # Release this rung's cached blocks before the next, larger rung asks for new ones. Without
        # this the allocator still holds every previous context's blocks, so a rung that would fit on
        # its own can fail during its warm-up and be written as an "oom" row -- a false OOM, which is
        # the one entry this table cannot afford because the row is the reader's only evidence.
        memory.empty_cache()
        if row["status"] == STATUS_OOM:
            break
    _write_probe(out_dir, adapter, ladder, device, rows)
    return rows


def _write_probe(
    out_dir,
    adapter: InferenceAdapter,
    ladder: tuple[int, ...],
    device: str,
    rows: list[dict[str, object]],
) -> Path:
    """Persist the ladder's rows with the identity of what produced them.

    The rows alone would not say which device, which arm or how many contexts were requested, so the
    artifact bundles those with them and records whether the ladder stopped early. A table rebuilt
    from this file can therefore state its own scope without a separate note.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / PROBE_FILENAME
    stopped_at_oom = bool(rows) and rows[-1]["status"] == STATUS_OOM
    payload = {
        "model": adapter.model,
        "device": device,
        "contexts_requested": list(ladder),
        "contexts_measured": [row["context"] for row in rows if row["status"] == STATUS_OK],
        "stopped_at_oom": stopped_at_oom,
        "rows": rows,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
