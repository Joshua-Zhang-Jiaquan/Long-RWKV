"""Synthetic CUDA metadata seams executed only on CPU; never runtime evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from pathlib import Path
from typing import assert_never

import pytest
import torch
from pydantic import JsonValue

from scale.experiments.nonlatent_iclr.qualification import lifecycle_runtime, runtime_checks, semantics_runtime
from scale.experiments.nonlatent_iclr.qualification.controller_receipt import ControllerReceipt
from scale.experiments.nonlatent_iclr.qualification.contracts import SuccessEvidence
from scale.experiments.nonlatent_iclr.qualification.lifecycle_evidence import LifecycleConfig, LifecycleEvidence, REQUIRED_STEPS
from scale.experiments.nonlatent_iclr.qualification.manifest import PreflightReceipt, REQUIRED_ROLES
from scale.experiments.nonlatent_iclr.qualification.runtime_config import ProbeConfig
from scale.experiments.nonlatent_iclr.qualification.lifecycle_settings import ModelSettings, observe_settings
from scale.tests.nonlatent_iclr.test_qualification_controls import _success_evidence
from scale.tests.nonlatent_iclr.test_qualification_controller_receipt import _receipt_payload
from scale.tests.nonlatent_iclr.test_qualification_runtime_contracts import _probe_config
from scale.tests.nonlatent_iclr.test_qualification_lifecycle import RecordingModel


class CPUModel(RecordingModel):
    training: bool = False

    def parameters(self) -> Iterator[torch.Tensor]:
        yield torch.zeros(1)


def synthetic_lifecycle() -> LifecycleEvidence:
    steps: list[dict[str, JsonValue]] = []
    for scenario, before, after, kind, code in REQUIRED_STEPS:
        step: dict[str, JsonValue] = {"scenario": scenario, "calls_before": before, "calls_after": after, "kind": kind}
        match kind:
            case "comparison":
                step.update(code=code, expected_shape=[1, 48, 65536], actual_shape=[1, 48, 65536],
                            reference_shape=[1, 48, 65536], actual_dtype="torch.float32", reference_dtype="torch.float32",
                            all_finite=True, exact_equal=True, max_abs_error=0.0)
            case "rejection":
                step.update(expected_code=code, observed_code=code)
            case "injection":
                step.update(code=code, synchronizations_before=10, synchronizations_after=11)
            case unreachable:
                assert_never(unreachable)
        steps.append(step)
    return LifecycleEvidence.model_validate_json(dumps({
        "device": "cuda:0", "vocab_size": 65536, "steps": tuple(steps),
        "inputs": tuple({"name": name, "sha256": str(index) * 64}
                        for index, name in enumerate(("a", "b", "edited_a"))),
        "model_calls": 12, "synchronizations": 12, "completed": True, "failure_code": None,
    }))


@dataclass(frozen=True, slots=True)
class RuntimeFixture:
    config: ProbeConfig
    model: CPUModel
    events: list[str]
    synchronize_devices: list[torch.device]
    success: SuccessEvidence


@pytest.fixture
def runtime_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RuntimeFixture:
    config = _probe_config(tmp_path)
    success = SuccessEvidence.model_validate_json(dumps(_success_evidence(0)))
    manifest = dumps({
        "schema_version": 1, "payload_id": "qualification-20260912-01",
        "expected_image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
        "packages": [{"distribution": "torch", "version": "CPU_TEST_FIXTURE"}],
        "files": [{"path": str(config.checkpoint_dir / "model.pt" if role == "checkpoint_model" else tmp_path / role),
                   "role": role, "sha256": success.checkpoint_sha256 if role == "checkpoint_model" else "0" * 64,
                   "size_bytes": success.checkpoint_size_bytes if role == "checkpoint_model" else 1}
                  for role in sorted(REQUIRED_ROLES)],
    }).encode()
    config.manifest_path.write_bytes(manifest)
    config = ProbeConfig.model_validate({**config.model_dump(), "manifest_sha256": sha256(manifest).hexdigest()})
    receipt = ControllerReceipt.model_validate({**_receipt_payload(), "source_manifest_sha256": config.manifest_sha256})
    events: list[str] = []
    synchronized: list[torch.device] = []
    model = CPUModel()
    preflight = PreflightReceipt(
        nonce=config.nonce, manifest_sha256=config.manifest_sha256, verified_source_files=8,
        package_versions=tuple(f"{name}==CPU_TEST_FIXTURE" for name in
                               ("flash-linear-attention", "fla-core", "transformers", "torch", "safetensors", "pytorch-triton")),
        expected_image=receipt.requested_image,
    )
    monkeypatch.setattr(runtime_checks, "load_preflight_receipt", lambda *args, **kwargs: preflight)
    monkeypatch.setattr(runtime_checks, "_load_checkpoint_metadata", lambda config: runtime_checks._CheckpointMetadata.model_validate(
        {"step": 4750, "tokens_seen": 4980736000.0}))
    monkeypatch.setattr(runtime_checks, "await_controller_receipt", lambda expectation: receipt)
    monkeypatch.setattr(runtime_checks, "initialize_hardware", lambda *args:
                        (0, torch.device("cuda:0"), success.gpu_inventory, 85_000_000_000))
    monkeypatch.setattr(runtime_checks, "construct_model", lambda *args: events.append("loaded_fixture") or model)
    monkeypatch.setattr(runtime_checks, "load_checkpoint", lambda *args: 1955)
    monkeypatch.setattr(runtime_checks, "model_geometry", lambda model: success.geometry)
    monkeypatch.setattr(runtime_checks, "parameter_evidence", lambda model: success.parameters)
    monkeypatch.setattr(runtime_checks, "causal_cache_evidence", lambda *args: success.fla_causal_cache)
    monkeypatch.setattr(runtime_checks, "masked_forward", lambda *args:
                        events.append("masked") or ((1, 48, 65536), True))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset_peak"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: events.append("peak") or 200)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 300)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(device))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: events.append("teardown"))

    def fixture_probe(loaded: CPUModel, probe_config: LifecycleConfig, synchronize: Callable[[], None]) -> LifecycleEvidence:
        assert loaded is model and probe_config == LifecycleConfig(vocab_size=65536, device="cuda:0")
        events.append("lifecycle")
        synchronize()
        return synthetic_lifecycle()

    monkeypatch.setattr(lifecycle_runtime, "probe_lifecycle", fixture_probe, raising=False)

    def fixture_settings(loaded: CPUModel) -> ModelSettings:
        assert loaded is model
        return ModelSettings.model_validate({**observe_settings(loaded).model_dump(), "parameter_devices": ("cuda:0",)})

    monkeypatch.setattr(lifecycle_runtime, "observe_settings", fixture_settings)
    # Legacy lifecycle tests isolate the new semantics boundary; dedicated tests exercise it.
    monkeypatch.setattr(semantics_runtime, "qualify_semantics", lambda model, invocation: None)
    return RuntimeFixture(config, model, events, synchronized, success)
