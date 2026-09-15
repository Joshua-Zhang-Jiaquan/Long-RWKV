"""GPU model, checkpoint, parameter, cache, and forward checks."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from torch import Tensor, device as TorchDevice

    from ..full_canvas import FullCanvasModel

from .contracts import (
    CausalCacheEvidence,
    ModelGeometry,
    ParameterEvidence,
    QualificationRuntimeError,
)
from .controls import ProbeConfig


MODEL_LOOP_RANGE: Final = (16, 32)
MODEL_LOOP_REPS: Final = 1
CHECKPOINT_STATE_TENSORS: Final = 1_955
CACHE_ATOL: Final = 0.05


def construct_model(
    config: ProbeConfig,
    torch_module,
    device,
    *,
    model_root: Path | None = None,
    loop_range: tuple[int, int] | None = MODEL_LOOP_RANGE,
    loop_reps: int = MODEL_LOOP_REPS,
    gradient_checkpointing: bool = False,
):
    """Build the bidirectional masked-diffusion model.

    Defaults reproduce the qualification model exactly; the calibration passes an arm's geometry
    instead of editing those defaults, so an arm can never silently change the qualified path.
    """
    if str(config.staged_root) not in sys.path:
        sys.path.insert(0, str(config.staged_root))
    model_module = import_module("models.birwkv7_diffusion")
    model_class = model_module.BiRWKV7ForMaskedDiffusion
    model = model_class.from_hf_pretrained(
        model_root if model_root is not None else config.hf_model_root,
        dtype=torch_module.bfloat16,
        gradient_checkpointing=gradient_checkpointing,
        loop_range=loop_range,
        loop_reps=loop_reps,
    )
    return model.to(device).eval()


def load_checkpoint(model, config: ProbeConfig, torch_module, *, checkpoint_dir: Path | None = None) -> int:
    payload = torch_module.load(
        (checkpoint_dir if checkpoint_dir is not None else config.checkpoint_dir) / "model.pt",
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    if not isinstance(payload, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, torch_module.Tensor)
        for key, value in payload.items()
    ):
        raise QualificationRuntimeError("checkpoint_payload_not_flat_mapping")
    state = payload
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise QualificationRuntimeError("checkpoint_state_not_exact")
    return len(state)


def model_geometry(model) -> ModelGeometry:
    config = model.config
    return ModelGeometry(
        num_hidden_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
    )


def parameter_evidence(model) -> ParameterEvidence:
    categories = {
        "forward_attention": [0, 0],
        "backward_attention": [0, 0],
        "fusion": [0, 0],
        "shared": [0, 0],
        "loop": [0, 0],
    }
    parameters = tuple(model.named_parameters())
    for name, parameter in parameters:
        if ".attn_fwd." in name:
            category = "forward_attention"
        elif ".attn_bwd." in name:
            category = "backward_attention"
        elif ".fuse_proj." in name or name.endswith(".fuse_bias"):
            category = "fusion"
        elif name.startswith("loop."):
            category = "loop"
        else:
            category = "shared"
        categories[category][0] += 1
        categories[category][1] += parameter.numel()
    return ParameterEvidence.model_validate({
        "total_tensors": len(parameters),
        "total_numel": sum(parameter.numel() for _, parameter in parameters),
        "forward_attention_tensors": categories["forward_attention"][0],
        "forward_attention_numel": categories["forward_attention"][1],
        "backward_attention_tensors": categories["backward_attention"][0],
        "backward_attention_numel": categories["backward_attention"][1],
        "fusion_tensors": categories["fusion"][0],
        "fusion_numel": categories["fusion"][1],
        "shared_tensors": categories["shared"][0],
        "shared_numel": categories["shared"][1],
        "loop_tensors": categories["loop"][0],
        "loop_numel": categories["loop"][1],
    })


def causal_cache_evidence(model, torch_module, device) -> CausalCacheEvidence:
    from fla.models.utils import Cache

    attention = model.layers[0].attn_fwd
    hidden = torch_module.randn(
        (1, 48, 2_560), device=device, dtype=torch_module.bfloat16
    )
    with torch_module.inference_mode():
        full, _, _, _ = attention(hidden)
        cache = Cache()
        attention(hidden[:, :32], past_key_values=cache, use_cache=True)
        suffix, _, _, _ = attention(
            hidden[:, 32:], past_key_values=cache, use_cache=False
        )
    delta = (full[:, 32:] - suffix).abs()
    if not bool(torch_module.isfinite(delta).all().item()):
        raise QualificationRuntimeError("fla_causal_cache_delta_non_finite")
    max_abs_delta = float(delta.max().item())
    torch_module.testing.assert_close(
        full[:, 32:], suffix, atol=CACHE_ATOL, rtol=0.0
    )
    return CausalCacheEvidence(
        scope="standalone_forward_rwkv7_attention",
        sequence_length=48,
        prefix_length=32,
        max_abs_delta=max_abs_delta,
        atol=CACHE_ATOL,
        rtol=0.0,
    )


class MaskedForwardTorch(Protocol):
    """Only the existing masked-forward Torch operations, without eager imports."""

    # Preserve Torch's independent bounds, shape and device arguments.
    def randint(
        self, low: int, high: int, size: tuple[int, int], *, device: TorchDevice,
    ) -> Tensor: ...

    def inference_mode(self) -> AbstractContextManager[None]: ...

    def isfinite(self, input: Tensor) -> Tensor: ...


def masked_forward(
    model: FullCanvasModel[Tensor], torch_module: MaskedForwardTorch, device: TorchDevice,
) -> tuple[tuple[int, int, int], bool]:
    from ..full_canvas import FullCanvasAdapter

    ids = torch_module.randint(0, 65_535, (1, 48), device=device)
    ids[:, 8:12] = 65_535
    with torch_module.inference_mode():
        adapter = FullCanvasAdapter(model)
        try:
            request = adapter.open(ids)
            logits = adapter.forward(request)
        finally:
            adapter.reset()
    shape = tuple(int(size) for size in logits.shape)
    if shape != (1, 48, 65_536):
        raise QualificationRuntimeError(f"masked_logits_shape_mismatch:{shape}")
    finite = bool(torch_module.isfinite(logits).all().item())
    if not finite:
        raise QualificationRuntimeError("masked_logits_non_finite")
    return (shape[0], shape[1], shape[2]), finite
