"""Pinned AST identities and typed historical verification inputs."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

import torch

SOURCE_SHA256: Final = "0bee15b5af05a785b70ddfeffa3064161f04beccea36b44e9bfd01029e286b60"
SOURCE_PATH: Final = "DAN/v7_arch_round/code/train/train_birwkv_diffusion.py"
SEGMENTS: Final = MappingProxyType({
    "constants": (78, 81, "857885f01b3c32e17feb2bcd65db99a25336cca1b744a05214c606b7b6416b26"),
    "corruption": (134, 271, "c05b772d7ae15900e8e64e7f8045a6e2dcc8e380c07a580183a84ec956992c00"),
    "loss": (277, 315, "7175eaf35e070ab456d69ab9f7b73aa7a0bd655bd61584e561cfcfe00d4c4ed6"),
    "regex": (430, 430, "9137480669a8a6ded3bee57eb8fb3890557f22796f9626bf4463a38684da25e8"),
    "predicate": (433, 443, "68e5bf2bd632a93208ab8ea52153d590fac8cacdc306870a86084714ce728e0a"),
    "freeze": (920, 927, "aab5ab71b2901a033a1b913f34e0e1d24cc6d32250998a89db6a1c2af4ae2977"),
    "optimizer": (952, 987, "3c64dcfa092c8edba21054607be8ee3622216f238e7b5d5602f7f9fb1a65fff3"),
    "learning_rate": (1250, 1253, "c26a09b795df00d010dc882011245f95845ce1ddf0982befa90ddf30370b1620"),
    "gate": (1493, 1501, "0dc6141de3f6c33f222454d31f19f5716e9090c9f4309b05e98cddb83848284b"),
})


@dataclass(frozen=True, slots=True)
class SourceContractError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    latent_on: bool = False
    freeze_backbone: bool = False
    weight_decay: float = 0.1
    latent_lr_mult: float = 1.0
    lr: float = 0.01


@dataclass(frozen=True, slots=True)
class GateConfig:
    step: int = 0
    steps: int = 100
    n_layers: int = 4
    stage_a_frac: float = 0.2
    stage_b_frac: float = 0.6


@runtime_checkable
class HistoricalFunctions(Protocol):
    # Signatures intentionally match the original functions, including arity.
    def sample_corruption(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        block_size: int, span_prob: float, generator: torch.Generator,
        gen_prob: float = 0.0, docgen_prob: float = 0.0,
        doc_spans: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def masked_diffusion_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
        bucket: torch.Tensor, block_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]: ...

    def grad_frozen(
        self, name: str, frac: float, n_layers: int, stage_a: float, stage_b: float,
    ) -> bool: ...
