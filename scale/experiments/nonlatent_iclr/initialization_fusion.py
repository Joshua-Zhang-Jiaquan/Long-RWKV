"""Actual fusion-gate initialization via extracted statements; no FLA import."""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Protocol, TypeGuard

import torch
from torch import nn

from .initialization_contract import SourceContractError as SourceContractError
from .initialization_extract import SourceSnapshot, execute_region


class FusionDiagnostic(Protocol):
    def __call__(self, shell: SimpleNamespace) -> float: ...


def _is_diagnostic(obj: object) -> TypeGuard[FusionDiagnostic]:
    return callable(obj)


@dataclass(frozen=True, slots=True)
class FusionCharacterization:
    fuse_proj: nn.Linear
    fuse_bias: nn.Parameter
    mean_alpha: float
    alpha: torch.Tensor
    output: torch.Tensor
    x: torch.Tensor
    o_fwd: torch.Tensor
    o_bwd: torch.Tensor


def _require_tiny_hidden(hidden: int) -> None:
    if not 1 <= hidden <= 64:
        raise SourceContractError("tiny fixture requires 1..64 hidden units")


def _require_tiny_stream(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cpu" or tensor.ndim != 3 or not bool(torch.isfinite(tensor).all()):
        raise SourceContractError(f"tiny finite CPU [B,T,C] required: {name}")


def fuse_at_init(
    fuse_proj: nn.Linear, fuse_bias: nn.Parameter,
    x: torch.Tensor, o_fwd: torch.Tensor, o_bwd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the original mix statements; returns (alpha, fused output)."""
    for name, tensor in (("x", x), ("o_fwd", o_fwd), ("o_bwd", o_bwd)):
        _require_tiny_stream(name, tensor)
    shell = SimpleNamespace(fuse_proj=fuse_proj, fuse_bias=fuse_bias)
    namespace = ModuleType("historical_fusion_mix_cpu")
    namespace.__dict__.update(torch=torch, self=shell, x=x, o_fwd=o_fwd, o_bwd=o_bwd)
    execute_region(SourceSnapshot.read("birwkv7_diffusion.py"), "fusion_mix", namespace)
    alpha = namespace.__dict__["alpha"]
    output = namespace.__dict__["o"]
    if not isinstance(alpha, torch.Tensor) or not isinstance(output, torch.Tensor):
        raise SourceContractError("historical fusion result mismatch")
    return alpha, output


def characterize(hidden: int, gate_bias_init: float) -> FusionCharacterization:
    """Run the actual init statements, then the diagnostic and one mix pass."""
    _require_tiny_hidden(hidden)
    snapshot = SourceSnapshot.read("birwkv7_diffusion.py")
    shell = SimpleNamespace()
    namespace = ModuleType("historical_fusion_cpu")
    namespace.__dict__.update(
        torch=torch, nn=nn, self=shell, hidden=hidden, gate_bias_init=gate_bias_init,
    )
    execute_region(snapshot, "fusion_init", namespace)
    fuse_proj = shell.fuse_proj
    fuse_bias = shell.fuse_bias
    if not isinstance(fuse_proj, nn.Linear) or not isinstance(fuse_bias, nn.Parameter):
        raise SourceContractError("historical fusion attribute mismatch")
    execute_region(snapshot, "fusion_alpha", namespace)
    diagnostic = namespace.__dict__["mean_forward_alpha"]
    if not _is_diagnostic(diagnostic):
        raise SourceContractError("historical diagnostic missing")
    generator = torch.Generator().manual_seed(20260913)
    x = torch.randn(2, 5, hidden, generator=generator)
    o_fwd = torch.randn(2, 5, hidden, generator=generator)
    o_bwd = torch.randn(2, 5, hidden, generator=generator)
    alpha, output = fuse_at_init(fuse_proj, fuse_bias, x, o_fwd, o_bwd)
    return FusionCharacterization(
        fuse_proj=fuse_proj, fuse_bias=fuse_bias, mean_alpha=diagnostic(shell),
        alpha=alpha, output=output, x=x, o_fwd=o_fwd, o_bwd=o_bwd,
    )
