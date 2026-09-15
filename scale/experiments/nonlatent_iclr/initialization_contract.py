"""Pinned source regions and typed interfaces for CPU-only historical seams."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

import torch
from torch import nn

from .trainer_semantics_contract import SourceContractError as SourceContractError

SOURCES: Final = MappingProxyType({
    "residual_streams.py": "5ba645d5924b1782c60fccd8a17dec26bc904e3675f2f1834d9e948cac7e9b68",
    "birwkv7_diffusion.py": "2591f58b1dfb7b567b979fb1f05cbb8c1e9a60490fc9fba293fcb2da48e0154a",
})
REGIONS: Final = MappingProxyType({
    "controls": ("residual_streams.py", 64, 183, "ee7d801860d02f2f1924d79183f4f7a4d1fa72d0d6801cfe57b34e6ec5812922"),
    "methods": ("birwkv7_diffusion.py", 742, 882, "f55c5b0778283d78eda8088f720a1ddc5d1d811f9daab57aece02c576f02ab33"),
    "loop": ("birwkv7_diffusion.py", 963, 1004, "dcc016ca67f260024310d4efd8f0cbe74941f8b98c4f59655f803642b519deee"),
    "fusion_init": ("birwkv7_diffusion.py", 474, 476, "b3aedca6f81930b9d6d2db01da8b33ce61ac090d3ff1f48f920563840fd7739f"),
    "fusion_mix": ("birwkv7_diffusion.py", 560, 562, "446abffd778a3839737bdd1d2c4e0cc3d8fa6bcd7a0028a49918edc45adae7f0"),
    "fusion_alpha": ("birwkv7_diffusion.py", 576, 578, "8eeb1a3786d94de41b99a0cf9b451e680db661e8f8cc14ac4d71664ad11ebbe5"),
})


class LoopControl(Protocol):
    gates_raw: nn.Parameter
    def gates(self) -> torch.Tensor: ...
    def state_dict(self) -> dict[str, torch.Tensor]: ...


class StreamControl(Protocol):
    def factors(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class HistoricalLoopModel(Protocol):
    # These signatures preserve the actual historical API, not a new config layer.
    layers: nn.ModuleList
    def attach_backbone_loop(self, loop_range: tuple[int, int], loop_reps: int,
                             loop_scale: float = 1.0) -> LoopControl: ...
    def attach_residual_streams(self, n_streams: int, mix_scale: float = 1.0) -> StreamControl: ...


@dataclass(frozen=True, slots=True)
class TinyConfig:
    num_hidden_layers: int


class FixtureBase(nn.Module):
    """Only fixture state; all attachment and layer execution methods are extracted."""

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers: nn.ModuleList = layers
        self.config: TinyConfig = TinyConfig(len(layers))
        self.residual_streams: nn.Module | None = None
        self.loop: nn.Module | None = None
        self.latent_cond: None = None
        self.gradient_checkpointing: bool = False


@dataclass(frozen=True, slots=True)
class BlockCall:
    block_id: int
    parameter_id: int
    shape: tuple[int, ...]
    highway: float


@dataclass(frozen=True, slots=True)
class LoopEvidence:
    hidden: torch.Tensor
    calls: tuple[BlockCall, ...]
    outer_invocations: int
    base_calls: int
    recycled_calls: int
