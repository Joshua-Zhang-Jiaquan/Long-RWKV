from __future__ import annotations

from json import dumps

import pytest
import torch
from pydantic import ValidationError

from scale.experiments.nonlatent_iclr.qualification import lifecycle_runtime, probe, runtime_checks
from scale.experiments.nonlatent_iclr.qualification.contracts import RankResult
from scale.experiments.nonlatent_iclr.qualification.lifecycle_binding import LifecycleBinding, checkpoint_from_manifest
from scale.experiments.nonlatent_iclr.qualification.lifecycle_settings import observe_settings
from scale.experiments.nonlatent_iclr.qualification.lifecycle_sidecar import LifecycleSidecar
from scale.tests.nonlatent_iclr.lifecycle_runtime_fixtures import (
    CPUModel, RuntimeFixture, runtime_fixture as runtime_fixture, synthetic_lifecycle,
)


def test_sidecar_binds_rank_and_checkpoint_when_runtime_passes(runtime_fixture: RuntimeFixture) -> None:
    # Given: an explicitly synthetic CPU orchestration fixture.
    fixture = runtime_fixture
    # When: the current runtime passes through the real sidecar publisher.
    result = runtime_checks.run_runtime_checks(fixture.config, 0)
    # Then: independent sidecar identity reconciles with config and core rank evidence.
    sidecar = LifecycleSidecar.model_validate_json((fixture.config.output_dir / "lifecycle-0.json").read_bytes())
    assert sidecar.passed and sidecar.runtime_promoted is False
    assert sidecar.binding.nonce == result.nonce == fixture.config.nonce
    assert sidecar.binding.run_id == fixture.config.run_id
    assert sidecar.binding.source_manifest_sha256 == fixture.config.manifest_sha256
    assert sidecar.binding.checkpoint == checkpoint_from_manifest(fixture.config)
    assert sidecar.binding.rank == sidecar.binding.local_rank == 0
    assert sidecar.binding.device == "cuda:0"
    assert result.evidence.kind == "success"
    assert sidecar.binding.controller == result.evidence.controller_binding


@pytest.mark.parametrize("field,value", [
    ("nonce", "f" * 32), ("run_id", "qualification-20260912-01-foreign12"),
    ("source_manifest_sha256", "f" * 64), ("rank", 1), ("local_rank", 1),
    ("device", "cuda:1"), ("device", "cpu"), ("world_size", 7),
])
def test_binding_rejects_foreign_identity_when_reparsed(
    runtime_fixture: RuntimeFixture, field: str, value: str | int,
) -> None:
    # Given: a valid published fixture binding with one independently changed identity.
    runtime_checks.run_runtime_checks(runtime_fixture.config, 0)
    sidecar = LifecycleSidecar.model_validate_json((runtime_fixture.config.output_dir / "lifecycle-0.json").read_bytes())
    payload = sidecar.binding.model_dump(mode="json")
    payload[field] = value
    # When/Then: JSON parsing rejects disagreement with controller or rank/device identity.
    with pytest.raises(ValidationError):
        LifecycleBinding.model_validate_json(dumps(payload))


@pytest.mark.parametrize("field,value", [("sha256", "f" * 64), ("size_bytes", 1), ("role", "payload")])
def test_binding_rejects_checkpoint_substitution_when_reparsed(
    runtime_fixture: RuntimeFixture, field: str, value: str | int,
) -> None:
    # Given: a bound checkpoint descriptor altered after a successful fixture call.
    runtime_checks.run_runtime_checks(runtime_fixture.config, 0)
    sidecar = LifecycleSidecar.model_validate_json((runtime_fixture.config.output_dir / "lifecycle-0.json").read_bytes())
    payload = sidecar.binding.model_dump(mode="json")
    payload["checkpoint"][field] = value
    # When/Then: unrelated bytes or a different role cannot replace the checkpoint identity.
    with pytest.raises(ValidationError):
        LifecycleBinding.model_validate_json(dumps(payload))


def test_settings_observation_is_real_and_read_only_when_cpu_model_supplied() -> None:
    # Given: an actual CPU-only tensor callable, not synthetic CUDA metadata.
    model = CPUModel()
    training = model.training
    deterministic = torch.are_deterministic_algorithms_enabled()
    precision = torch.get_float32_matmul_precision()
    # When: settings are observed inside the same inference context used by qualification.
    with torch.inference_mode():
        settings = observe_settings(model)
    # Then: CPU residency and actual flags are recorded without changing policy or model mode.
    assert settings.parameter_devices == ("cpu",)
    assert settings.parameter_dtypes == ("torch.float32",)
    assert settings.training == training and settings.inference_mode and not settings.grad_enabled
    assert settings.deterministic_algorithms == deterministic == torch.are_deterministic_algorithms_enabled()
    assert settings.float32_matmul_precision == precision == torch.get_float32_matmul_precision()
    assert model.training == training


def test_training_precondition_is_preserved_when_model_is_not_eval(runtime_fixture: RuntimeFixture) -> None:
    # Given: the loaded CPU fixture reports training mode.
    runtime_fixture.model.training = True
    # When: the real caller rejects rather than silently calling eval().
    with pytest.raises(runtime_checks.RuntimeCheckFailure, match="lifecycle"):
        runtime_checks.run_runtime_checks(runtime_fixture.config, 0)
    # Then: the precondition observation survives and no probe call occurred.
    sidecar = LifecycleSidecar.model_validate_json((runtime_fixture.config.output_dir / "lifecycle-0.json").read_bytes())
    assert sidecar.before is not None and sidecar.before.training
    assert sidecar.observations is None and not sidecar.passed
    assert runtime_fixture.model.training and "lifecycle" not in runtime_fixture.events


def test_terminal_failure_preserves_partial_sidecar_when_probe_rejects(
    runtime_fixture: RuntimeFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a partial observation at the probe seam and real terminal publication.
    partial = synthetic_lifecycle().model_dump()
    partial.update(completed=False, failure_code="nonfinite", steps=(), model_calls=2, synchronizations=2)
    from scale.experiments.nonlatent_iclr.qualification.lifecycle_evidence import LifecycleEvidence
    observation = LifecycleEvidence.model_validate(partial)
    monkeypatch.setattr(lifecycle_runtime, "probe_lifecycle", lambda *args: observation)
    monkeypatch.setenv("RANK", "0")
    config = runtime_fixture.config
    # When: the real entrypoint catches the runtime-boundary failure.
    exit_code = probe.main(("--output-dir", str(config.output_dir), "--nonce", config.nonce,
                           "--manifest", str(config.manifest_path), "--manifest-sha256", config.manifest_sha256,
                           "--checkpoint-dir", str(config.checkpoint_dir), "--staged-root", str(config.staged_root),
                           "--model-dir", str(config.hf_model_root),
                           "--run-id", config.run_id, "--controller-receipt", str(config.controller_receipt)))
    # Then: both partial sidecar and failed rank exist; no passing terminal record escapes.
    result = RankResult.model_validate_json((config.output_dir / "rank-0.json").read_bytes())
    sidecar = LifecycleSidecar.model_validate_json((config.output_dir / "lifecycle-0.json").read_bytes())
    assert exit_code == 1 and result.status == "FAILED"
    assert sidecar.observations == observation and not sidecar.passed
    assert result.evidence.kind == "failure" and result.evidence.failed_check == "runtime_boundary"
