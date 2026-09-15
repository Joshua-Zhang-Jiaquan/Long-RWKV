"""Task3-local full-canvas caller boundary; CPU engineering, not qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, NewType, Protocol, TypeVar

import torch

CanvasRevision = NewType("CanvasRevision", int)
Output = TypeVar("Output", covariant=True)
BoundaryErrorCode = Literal[
    "state_cache_forbidden", "use_cache_forbidden", "z_slots_forbidden",
    "coordinator_forbidden", "unsupported_cache_mode", "session_active",
    "session_closed", "session_mismatch", "stale_canvas", "invalid_canvas",
    "canvas_shape_mismatch",
]


class CanvasBoundaryError(Exception):
    """Machine-readable rejection; never retains the rejected tensor/cache."""

    def __init__(self, code: BoundaryErrorCode) -> None:
        super().__init__(code)
        self.code: BoundaryErrorCode = code


@dataclass(frozen=True, slots=True, eq=False)
class CanvasSession:
    """Opaque in-process identity, compared by identity, never by recycled IDs."""


@dataclass(frozen=True, slots=True)
class CanvasRequest:
    session_id: CanvasSession
    canvas_revision: CanvasRevision


class FullCanvasModel(Protocol[Output]):
    """Actual model call seam; additional model options stay at their defaults."""

    # Five inputs are required by the legacy model signature, not a new abstraction.
    def __call__(
        self, input_ids: torch.Tensor, *, force_forward: Literal[False],
        z_slots: None, state_cache: None, use_cache: Literal[False],
    ) -> Output: ...


@dataclass(frozen=True, slots=True)
class _ActiveCanvas:
    request: CanvasRequest
    input_ids: torch.Tensor


class FullCanvasAdapter(Generic[Output]):
    """Mutable, single-owner synchronous session; callers must serialize operations.

    open/edit own detached copies of rank-2, nonempty int32/int64 token IDs.
    Each forward receives a further copy, never the owned snapshot. External
    tensor mutation cannot be detected by revision assertions alone: it is
    isolated here, and only edit_canvas changes the owned canvas. Do not mutate
    input concurrently with copying (including unsynchronized device streams).

    No logits, hidden/recurrent state, or caller tensor is retained. The trusted
    model must honor the supplied kwargs; this cannot police hidden model caches
    or reentrant lifecycle calls. This API is opt-in, not qualification-v5 wiring.
    """

    __slots__ = ("_model", "_active")

    def __init__(self, model: FullCanvasModel[Output]) -> None:
        self._model: FullCanvasModel[Output] = model
        self._active: _ActiveCanvas | None = None

    def open(self, input_ids: torch.Tensor) -> CanvasRequest:
        """Open only when inactive; each opening issues a fresh identity object."""
        if self._active is not None:
            raise CanvasBoundaryError("session_active")
        owned = self._copy_canvas(input_ids)
        request = CanvasRequest(CanvasSession(), CanvasRevision(0))
        self._active = _ActiveCanvas(request, owned)
        return request

    def edit_canvas(self, request: CanvasRequest, input_ids: torch.Tensor) -> CanvasRequest:
        """Atomically replace the full same-shape canvas and advance its revision."""
        active = self._require_active(request)
        if input_ids.shape != active.input_ids.shape:
            raise CanvasBoundaryError("canvas_shape_mismatch")
        owned = self._copy_canvas(input_ids)
        current = CanvasRequest(request.session_id, CanvasRevision(request.canvas_revision + 1))
        self._active = _ActiveCanvas(current, owned)
        return current

    # Explicit rejection-only legacy knobs cannot be grouped without hiding the
    # call boundary. object deliberately accepts *any* cache/coordinator solely
    # to reject it; these opaque values never enter the model or owned state.
    def forward(
        self, request: CanvasRequest, *, mode: str = "full_canvas",
        state_cache: object = None, use_cache: bool = False,
        z_slots: object = None, coordinator: object = None,
    ) -> Output:
        """Recompute the whole canvas; reject prohibited knobs before invocation."""
        if state_cache is not None:
            raise CanvasBoundaryError("state_cache_forbidden")
        if use_cache is not False:
            raise CanvasBoundaryError("use_cache_forbidden")
        if z_slots is not None:
            raise CanvasBoundaryError("z_slots_forbidden")
        if coordinator is not None:
            raise CanvasBoundaryError("coordinator_forbidden")
        if mode != "full_canvas":
            raise CanvasBoundaryError("unsupported_cache_mode")
        active = self._require_active(request)
        return self._model(
            input_ids=active.input_ids.clone(), force_forward=False,
            z_slots=None, state_cache=None, use_cache=False,
        )

    def close(self, request: CanvasRequest) -> None:
        """Retire only the current session/revision and release its canvas."""
        self._require_active(request)
        self._active = None

    def reset(self) -> None:
        """Owner-only idempotent retirement; reopening never reuses the identity."""
        self._active = None

    def _require_active(self, request: CanvasRequest) -> _ActiveCanvas:
        active = self._active
        if active is None:
            raise CanvasBoundaryError("session_closed")
        if request.session_id is not active.request.session_id:
            raise CanvasBoundaryError("session_mismatch")
        if request.canvas_revision != active.request.canvas_revision:
            raise CanvasBoundaryError("stale_canvas")
        return active

    @staticmethod
    def _copy_canvas(input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.numel() == 0 or input_ids.dtype not in (torch.int32, torch.int64):
            raise CanvasBoundaryError("invalid_canvas")
        return input_ids.detach().clone()
