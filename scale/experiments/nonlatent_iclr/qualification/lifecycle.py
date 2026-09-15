"""Opt-in lifecycle probe seam (not wired into qualification)."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from struct import pack

import torch

from ..full_canvas import CanvasBoundaryError, FullCanvasAdapter, FullCanvasModel
from .lifecycle_checks import CallerBoundaryFailure, LifecycleAbort, LifecycleChecks
from .lifecycle_evidence import (
    FailureCode, InjectionEvidence, InputEvidence, LifecycleConfig, LifecycleEvidence,
)


def probe_lifecycle(
    model: FullCanvasModel[torch.Tensor], config: LifecycleConfig,
    synchronize: Callable[[], None],
) -> LifecycleEvidence:
    """Observe an already-loaded, eval-mode model using at most twelve full forwards.

    The caller owns model/device suitability, exclusive access and synchronization
    semantics. This does not load weights, set eval/RNG/backend modes, promote a
    runtime, or recover a device fault. RuntimeError stops with partial evidence;
    other unexpected exceptions propagate after finally retirement. GPU integration
    must supply a real device synchronization callback, never a CPU no-op.
    """
    rows = (
        tuple(index % 3 if not 8 <= index < 12 else config.vocab_size - 1 for index in range(48)),
        tuple((index + 1) % 3 if not 8 <= index < 12 else config.vocab_size - 1 for index in range(48)),
    )
    edited = (2, *rows[0][1:])
    inputs = tuple(InputEvidence(name=name, sha256=sha256(pack("<48q", *row)).hexdigest())
                   for name, row in zip(("a", "b", "edited_a"), (*rows, edited), strict=True))
    checks = LifecycleChecks(model, synchronize)
    adapter = FullCanvasAdapter(checks)
    foreign = FullCanvasAdapter(checks)
    failure: FailureCode | None = None
    completed = False
    with torch.inference_mode():
        try:
            a = torch.tensor([rows[0]], dtype=torch.int64, device=config.device)
            b = torch.tensor([rows[1]], dtype=torch.int64, device=config.device)
            edit = torch.tensor([edited], dtype=torch.int64, device=config.device)
            reference = checks.snapshot(checks.direct(a))
            caller = a.clone()
            request = adapter.open(caller)
            checks.compare((adapter.forward(request), reference), config)

            checks.begin("repeat_a")
            checks.compare((adapter.forward(request), reference), config)
            caller.copy_(b)
            checks.begin("caller_mutation")
            checks.compare((adapter.forward(request), reference), config)

            stale = request
            request = adapter.edit_canvas(request, edit)
            edit.copy_(b)
            checks.begin("edit_reference")
            edit_reference_ids = torch.tensor([edited], dtype=torch.int64, device=config.device)
            edit_output = checks.snapshot(adapter.forward(request))
            checks.compare((edit_output, checks.direct(edit_reference_ids)), config)
            del edit_output
            checks.begin("stale_revision")
            checks.reject(lambda: adapter.forward(stale), "stale_canvas")
            adapter.close(request)
            checks.begin("closed_request")
            checks.reject(lambda: adapter.forward(request), "session_closed")

            old = request
            request = adapter.open(b)
            checks.begin("reference_b")
            reference_b = checks.snapshot(checks.direct(b))
            checks.compare((adapter.forward(request), reference_b), config)
            del reference_b
            checks.begin("old_session")
            checks.reject(lambda: adapter.forward(old), "session_mismatch")
            foreign_request = foreign.open(a)
            checks.begin("foreign_session")
            checks.reject(lambda: adapter.forward(foreign_request), "session_mismatch")
            foreign.reset()
            adapter.close(request)

            request = adapter.open(a)
            checks.begin("a_b_a")
            checks.compare((adapter.forward(request), reference), config)
            adapter.reset()
            checks.begin("reset_retirement")
            checks.reject(lambda: adapter.forward(request), "session_closed")
            old = request
            request = adapter.open(a)
            checks.begin("reset_reopen")
            checks.compare((adapter.forward(request), reference), config)
            checks.begin("reset_old_session")
            checks.reject(lambda: adapter.forward(old), "session_mismatch")
            adapter.close(request)

            checks.begin("injected_forward")
            calls_before, sync_before = checks.calls, checks.synchronizations
            request = adapter.open(a)
            try:
                try:
                    checks.compare((adapter.forward(request), reference), config)
                    raise CallerBoundaryFailure
                finally:
                    adapter.reset()
            except CallerBoundaryFailure:
                checks.steps.append(InjectionEvidence(
                    scenario="caller_failure", calls_before=calls_before, calls_after=checks.calls,
                    synchronizations_before=sync_before, synchronizations_after=checks.synchronizations,
                ))
            checks.begin("failure_retirement")
            checks.reject(lambda: adapter.forward(request), "session_closed")
            old = request
            request = adapter.open(a)
            checks.begin("failure_reopen")
            checks.compare((adapter.forward(request), reference), config)
            checks.begin("failure_old_session")
            checks.reject(lambda: adapter.forward(old), "session_mismatch")
            completed = True
        except CanvasBoundaryError as error:
            failure = checks.execution_error("boundary_error", error)
        except LifecycleAbort as abort:
            failure = abort.code
        finally:
            adapter.reset()
            foreign.reset()
    return LifecycleEvidence(
        device=config.device, vocab_size=config.vocab_size,
        inputs=(inputs[0], inputs[1], inputs[2]), steps=tuple(checks.steps),
        model_calls=checks.calls, synchronizations=checks.synchronizations,
        completed=completed, failure_code=failure,
    )
