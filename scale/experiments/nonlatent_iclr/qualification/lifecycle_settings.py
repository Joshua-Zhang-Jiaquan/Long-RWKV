"""Read-only execution settings observations; never changes numerical policy."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Literal, Protocol

from pydantic import Field
import torch

from ..full_canvas import FullCanvasModel
from .lifecycle_evidence import StrictEvidence


class LoadedLifecycleModel(FullCanvasModel[torch.Tensor], Protocol):
    @property
    def training(self) -> bool: ...

    def parameters(self) -> Iterator[torch.Tensor]: ...


class ModelSettings(StrictEvidence):
    training: bool
    inference_mode: bool
    grad_enabled: bool
    torch_version: str
    parameter_dtypes: tuple[str, ...] = Field(min_length=1)
    parameter_devices: tuple[str, ...] = Field(min_length=1)
    cuda_autocast_enabled: bool
    cuda_autocast_dtype: str
    float32_matmul_precision: Literal["highest", "high", "medium"]
    matmul_allow_tf32: bool
    fp16_reduced_precision_reduction: bool
    bf16_reduced_precision_reduction: bool
    cudnn_allow_tf32: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    deterministic_warn_only: bool
    cublas_workspace_config: str | None


def observe_settings(model: LoadedLifecycleModel) -> ModelSettings:
    parameter_metadata = {(str(parameter.dtype), str(parameter.device)) for parameter in model.parameters()}
    return ModelSettings.model_validate({
        "training": model.training, "inference_mode": torch.is_inference_mode_enabled(),
        "grad_enabled": torch.is_grad_enabled(), "torch_version": str(torch.__version__),
        "parameter_dtypes": tuple(sorted({dtype for dtype, _ in parameter_metadata})),
        "parameter_devices": tuple(sorted({device for _, device in parameter_metadata})),
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    })
