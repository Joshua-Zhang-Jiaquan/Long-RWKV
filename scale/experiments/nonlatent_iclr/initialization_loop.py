"""Tiny CPU evidence at the historical hidden-state loop boundary, not model inference."""
from __future__ import annotations

from types import ModuleType
from typing import TypeGuard

import torch
from torch import nn

from .initialization_contract import (
    BlockCall, FixtureBase, HistoricalLoopModel, LoopEvidence,
    SourceContractError as SourceContractError,
)
from .initialization_extract import SourceSnapshot, control_namespace, execute_region


def is_loop_model(obj: object) -> TypeGuard[HistoricalLoopModel]:
    """Structural check; runtime Protocol isinstance cannot see nn.Module submodules."""
    return (
        isinstance(getattr(obj, "layers", None), nn.ModuleList)
        and callable(getattr(obj, "attach_backbone_loop", None))
        and callable(getattr(obj, "attach_residual_streams", None))
    )


class TinyResidualBlock(nn.Module):
    """Non-FLA block: add a scalar parameter and record calls; mutation is the probe."""

    def __init__(self, calls: list[BlockCall]) -> None:
        super().__init__()
        self.delta: nn.Parameter = nn.Parameter(torch.ones(1))
        self.calls: list[BlockCall] = calls

    # Arity must match the historical block call, including inactive options.
    def forward(self, h: torch.Tensor, vf: torch.Tensor, vb: torch.Tensor,
                force_forward: bool, film: tuple[torch.Tensor, ...] | None,
                state_cache: None, use_cache: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if force_forward or film is not None or state_cache is not None or use_cache:
            raise SourceContractError("tiny fixture only supports unconditioned cache-free calls")
        self.calls.append(BlockCall(id(self), id(self.delta), tuple(h.shape), float(vf.flatten()[0])))
        return h + self.delta, vf + 1, vb - 1


class _UnavailableConditioner:
    """Uninstantiated type binding for unreachable latent branches; no fake computation."""


def require_shared(base: tuple[BlockCall, ...], recycled: tuple[BlockCall, ...]) -> None:
    if not base or len(base) != len(recycled):
        raise SourceContractError("sharing requires equal nonempty call ranges")
    if any((a.block_id, a.parameter_id) != (b.block_id, b.parameter_id)
           for a, b in zip(base, recycled, strict=True)):
        raise SourceContractError("false sharing: block or parameter object differs")


class TinyLoopProbe:
    """Execute original methods with tiny blocks; retain per-invocation call evidence."""

    def __init__(self, layers: int) -> None:
        if not 1 <= layers <= 32:
            raise SourceContractError("tiny fixture requires 1..32 layers")
        namespace = control_namespace()
        namespace.__dict__.update(LatentFiLMConditioner=_UnavailableConditioner,
                                  LatentCrossAttnConditioner=_UnavailableConditioner,
                                  _grad_checkpoint=None, FixtureBase=FixtureBase)
        self.snapshot: SourceSnapshot = SourceSnapshot.read("birwkv7_diffusion.py")
        execute_region(self.snapshot, "methods", namespace)
        model_type = namespace.__dict__["HistoricalMethods"]
        if not isinstance(model_type, type) or not issubclass(model_type, FixtureBase):
            raise SourceContractError("historical method shell mismatch")
        self.calls: list[BlockCall] = []
        model = model_type(nn.ModuleList(TinyResidualBlock(self.calls) for _ in range(layers)))
        if not is_loop_model(model):
            raise SourceContractError("historical model interface mismatch")
        self.model: HistoricalLoopModel = model
        self.namespace: ModuleType = namespace
        self.layers: int = layers

    def run(self, h: torch.Tensor, reps: int | None = None) -> LoopEvidence:
        """One synthetic outer invocation; original base, recycle, readout statements."""
        if h.device.type != "cpu" or h.ndim != 3 or not 0 < h.numel() <= 4096:
            raise SourceContractError("tiny CPU [B,T,C] input required")
        if h.dtype != torch.float32 or not bool(torch.isfinite(h).all()):
            raise SourceContractError("finite fp32 fixture required")
        if reps is not None and reps > 16:
            raise SourceContractError("tiny fixture override limit exceeded")
        self.calls.clear()
        namespace = ModuleType("historical_loop_invocation")
        namespace.__dict__.update(self.namespace.__dict__)
        namespace.__dict__.update(self=self.model, h=h, force_forward=False, film_params=None,
                                  xattn_kv=None, state_cache=None, use_cache=False,
                                  loop_reps_override=reps)
        execute_region(self.snapshot, "loop", namespace)
        hidden = namespace.__dict__["h"]
        if not isinstance(hidden, torch.Tensor):
            raise SourceContractError("historical loop result mismatch")
        calls = tuple(self.calls)
        base_calls = len(calls[:self.layers])
        return LoopEvidence(hidden, calls, 1, base_calls, len(calls) - base_calls)
