"""GPU-job-only orchestration for bounded hardware and model checks."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import subprocess
import time
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .controller_receipt import (
    ControllerReceiptExpectation,
    await_controller_receipt,
)
from .contracts import (
    CHECKPOINT_SHA256,
    CheckName,
    FailureCheck,
    MemoryEvidence,
    QualificationRuntimeError,
    RankResult,
    SuccessEvidence,
)
from .controls import ProbeConfig
from .manifest import load_preflight_receipt, sha256_file
from .model_checks import (
    CHECKPOINT_STATE_TENSORS,
    causal_cache_evidence,
    construct_model,
    load_checkpoint,
    masked_forward,
    model_geometry,
    parameter_evidence,
)
from .runtime_identity import initialize_hardware


CHECKPOINT_SIZE_BYTES: Final = 16_367_167_378


@dataclass(frozen=True, slots=True)
class RuntimeCheckFailure(RuntimeError):
    failed_check: FailureCheck
    completed_checks: tuple[CheckName, ...]
    detail: str

    def __str__(self) -> str:
        return self.detail


class _CheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    step: Literal[4_750]
    tokens_seen: float = Field(ge=4_980_736_000.0, le=4_980_736_000.0)


def run_runtime_checks(config: ProbeConfig, rank: int) -> RankResult:
    started_utc = time.time()
    started_monotonic = time.monotonic()
    completed: list[CheckName] = []
    active_check: FailureCheck = "source_identity"
    dist_module = None
    try:
        if sha256_file(config.manifest_path) != config.manifest_sha256:
            raise QualificationRuntimeError("manifest_digest_changed_after_preflight")
        preflight_receipt = load_preflight_receipt(
            config.preflight_receipt,
            nonce=config.nonce,
            manifest_sha256=config.manifest_sha256,
        )
        checkpoint_metadata = _load_checkpoint_metadata(config)
        package_versions = _package_versions(preflight_receipt.package_versions)
        controller_receipt = await_controller_receipt(
            ControllerReceiptExpectation(
                path=config.controller_receipt,
                run_id=config.run_id,
                nonce=config.nonce,
                source_manifest_sha256=config.manifest_sha256,
                timeout_seconds=config.controller_receipt_timeout_seconds,
                poll_interval_seconds=0.1,
            )
        )
        if controller_receipt.requested_image != preflight_receipt.expected_image:
            raise QualificationRuntimeError("controller_preflight_image_mismatch")
        controller_binding = controller_receipt.to_binding_evidence()
        import torch
        import torch.distributed as dist

        dist_module = dist
        active_check = "hardware"
        local_rank, device, inventory, physical_bytes = initialize_hardware(
            config, rank, torch, dist
        )
        completed.append("hardware")
        completed.append("source_identity")

        active_check = "checkpointload"
        torch.manual_seed(20_260_912 + rank)
        torch.cuda.manual_seed_all(20_260_912 + rank)
        torch.cuda.reset_peak_memory_stats(device)
        model = construct_model(config, torch, device)
        checkpoint_state_tensors = load_checkpoint(model, config, torch)
        if checkpoint_state_tensors != CHECKPOINT_STATE_TENSORS:
            raise QualificationRuntimeError("checkpoint_state_tensor_count_mismatch")
        allocated_after_load = torch.cuda.memory_allocated(device)
        completed.append("checkpointload")

        active_check = "parameteridentity"
        geometry = model_geometry(model)
        parameters = parameter_evidence(model)
        completed.append("parameteridentity")

        active_check = "realFLAcausalcache"
        causal_cache = causal_cache_evidence(model, torch, device)
        completed.append("realFLAcausalcache")

        active_check = "actualmaskedforward"
        masked_shape, masked_finite = masked_forward(model, torch, device)
        completed.append("actualmaskedforward")

        active_check = "runtime_boundary"
        from .lifecycle_binding import LifecycleBinding, LifecycleInvocation, checkpoint_from_manifest
        from .lifecycle_runtime import qualify_lifecycle
        from .semantics_runtime import qualify_semantics

        lifecycle_binding = LifecycleBinding.model_validate({
            "controller": controller_binding, "run_id": config.run_id, "nonce": config.nonce,
            "source_manifest_sha256": config.manifest_sha256,
            "checkpoint": checkpoint_from_manifest(config), "checkpoint_step": checkpoint_metadata.step,
            "checkpoint_state_tensors": checkpoint_state_tensors, "geometry": geometry,
            "rank": rank, "local_rank": local_rank, "world_size": dist.get_world_size(), "device": str(device),
        })
        qualify_lifecycle(model, LifecycleInvocation(config.output_dir, lifecycle_binding, device))
        qualify_semantics(model, LifecycleInvocation(config.output_dir, lifecycle_binding, device))

        active_check = "resource_limits"
        torch.cuda.synchronize(device)
        memory = MemoryEvidence(
            allocated_after_load_bytes=allocated_after_load,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            physical_bytes=physical_bytes,
        )
        if max(
            memory.allocated_after_load_bytes,
            memory.peak_allocated_bytes,
            memory.peak_reserved_bytes,
        ) > int(memory.physical_bytes * config.memory_fraction):
            raise QualificationRuntimeError("memory_ceiling_exceeded")
        completed.append("resource_limits")
        dist.barrier()
        torch.cuda.synchronize(device)

        evidence = SuccessEvidence.model_validate(dict(
            kind="success",
            start_utc=started_utc,
            end_utc=time.time(),
            elapsed_monotonic_seconds=time.monotonic() - started_monotonic,
            rank=rank,
            local_rank=local_rank,
            world_size=dist.get_world_size(),
            gpu_inventory=inventory,
            controller_binding=controller_binding,
            python_version=platform.python_version(),
            torch_version=str(torch.__version__),
            fla_version=package_versions["flash-linear-attention"],
            fla_core_version=package_versions["fla-core"],
            transformers_version=package_versions["transformers"],
            safetensors_version=package_versions["safetensors"],
            triton_version=package_versions["pytorch-triton"],
            source_manifest_sha256=preflight_receipt.manifest_sha256,
            verified_source_files=preflight_receipt.verified_source_files,
            checkpoint_step=checkpoint_metadata.step,
            checkpoint_sha256=CHECKPOINT_SHA256,
            checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
            checkpoint_state_tensors=checkpoint_state_tensors,
            geometry=geometry,
            parameters=parameters,
            fla_causal_cache=causal_cache,
            masked_logits_all_finite=masked_finite,
            masked_logits_shape=masked_shape,
            memory=memory,
            fullmodel_cachedprefix="NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED",
            loop_cachedprefix="NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES",
        ))
        return RankResult.passing(rank=rank, nonce=config.nonce, evidence=evidence)
    except RuntimeCheckFailure:
        raise
    except (
        AssertionError,
        ImportError,
        KeyError,
        MemoryError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        raise RuntimeCheckFailure(
            failed_check=active_check,
            completed_checks=tuple(completed),
            detail=f"{active_check}:{type(error).__name__}:{error}",
        ) from error
    finally:
        if dist_module is not None and dist_module.is_initialized():
            try:
                dist_module.destroy_process_group()
            except (OSError, RuntimeError) as error:
                raise RuntimeCheckFailure(
                    failed_check="distributed_teardown",
                    completed_checks=tuple(completed),
                    detail=f"distributed_teardown:{type(error).__name__}:{error}",
                ) from error


def _load_checkpoint_metadata(config: ProbeConfig) -> _CheckpointMetadata:
    try:
        return _CheckpointMetadata.model_validate_json(
            (config.checkpoint_dir / "meta.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise QualificationRuntimeError("checkpoint_metadata_invalid") from error


def _package_versions(values: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for value in values:
        distribution, separator, version = value.partition("==")
        if separator != "==" or not distribution or not version:
            raise QualificationRuntimeError("preflight_package_identity_malformed")
        versions[distribution] = version
    required = {
        "flash-linear-attention",
        "fla-core",
        "transformers",
        "torch",
        "safetensors",
        "pytorch-triton",
    }
    if not required.issubset(versions):
        raise QualificationRuntimeError("preflight_package_identity_incomplete")
    return versions
