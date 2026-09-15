"""Explicit synthetic CPU orchestration; outputs are temporary test fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from scale.experiments.nonlatent_iclr.qualification import runtime_checks, semantics_runtime
from scale.experiments.nonlatent_iclr.qualification.lifecycle_settings import observe_settings, ModelSettings
from scale.experiments.nonlatent_iclr.qualification.lifecycle_sidecar import LifecycleSidecar
from scale.experiments.nonlatent_iclr.qualification.semantics_optimizer import probe_membership
from scale.experiments.nonlatent_iclr.qualification.semantics_probe import probe_loss_and_gradients
from scale.experiments.nonlatent_iclr.qualification.semantics_runtime import qualify_semantics
from scale.experiments.nonlatent_iclr.qualification.semantics_sidecar import SemanticsSidecar
from scale.tests.nonlatent_iclr.lifecycle_runtime_fixtures import (
    CPUModel, RuntimeFixture, runtime_fixture as runtime_fixture,
)
from scale.tests.nonlatent_iclr.test_model_semantics import TinyModel


@pytest.fixture
def semantics_fixture(runtime_fixture: RuntimeFixture, monkeypatch: pytest.MonkeyPatch) -> RuntimeFixture:
    tiny = TinyModel()
    membership = probe_membership(tiny)
    mask, gradients = probe_loss_and_gradients(tiny, torch.device("cpu"), 32)
    monkeypatch.setattr(semantics_runtime, "qualify_semantics", qualify_semantics)

    def settings(model: CPUModel) -> ModelSettings:
        assert model is runtime_fixture.model
        return observe_settings(model).model_copy(update={"parameter_devices": ("cuda:0",)})

    monkeypatch.setattr(semantics_runtime, "observe_settings", settings)
    monkeypatch.setattr(semantics_runtime, "probe_membership", lambda model: membership)
    monkeypatch.setattr(semantics_runtime, "probe_loss_and_gradients", lambda model, device, layers:
                        (mask.model_copy(update={"device": "cuda:0"}), gradients))
    return runtime_fixture


def read_sidecar(directory: Path) -> SemanticsSidecar:
    return SemanticsSidecar.model_validate_json((directory / "model-semantics-0.json").read_bytes())


def test_semantics_binds_when_runtime_passes(semantics_fixture: RuntimeFixture) -> None:
    # Given: metadata-only CUDA seams over CPU observations.
    fixture = semantics_fixture
    # When: the real caller executes both publication boundaries.
    result = runtime_checks.run_runtime_checks(fixture.config, 0)
    # Then: sidecars share every identity and the core schema remains unchanged.
    sidecar = read_sidecar(fixture.config.output_dir)
    lifecycle = LifecycleSidecar.model_validate_json((fixture.config.output_dir / "lifecycle-0.json").read_bytes())
    assert sidecar.passed and sidecar.binding == lifecycle.binding
    assert sidecar.binding.nonce == result.nonce
    assert sidecar.binding.source_manifest_sha256 == fixture.config.manifest_sha256
    assert result.evidence.kind == "success"


@pytest.mark.parametrize("failure", ["group", "loss", "exception"])
def test_semantics_publishes_failure_when_runtime_rejects(
    semantics_fixture: RuntimeFixture, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    # Given: a bad grouping, wrong loss, or failed forward at explicit CPU seams.
    fixture = semantics_fixture
    membership = probe_membership(TinyModel())
    if failure == "group":
        monkeypatch.setattr(semantics_runtime, "probe_membership", lambda model:
                            (membership[0].model_copy(update={"each_trainable_once": False}), membership[1]))
    else:
        def failing_probe(model: CPUModel, device: torch.device, layers: int):
            if failure == "exception":
                raise RuntimeError("synthetic_forward_failure")
            mask, gradients = probe_loss_and_gradients(TinyModel(), torch.device("cpu"), layers)
            return mask.model_copy(update={"device": "cuda:0", "empty_loss": 1.0}), gradients
        monkeypatch.setattr(semantics_runtime, "probe_loss_and_gradients", failing_probe)
    # When: runtime attempts qualification.
    with pytest.raises(runtime_checks.RuntimeCheckFailure) as rejected:
        runtime_checks.run_runtime_checks(fixture.config, 0)
    # Then: the failed rank cannot pass and partial evidence already exists.
    sidecar = read_sidecar(fixture.config.output_dir)
    assert not sidecar.passed and sidecar.membership
    assert rejected.value.failed_check == "runtime_boundary"
    assert "peak" not in fixture.events and fixture.events[-1] == "teardown"


@pytest.mark.parametrize("field,value", [
    ("nonce", "f" * 32), ("run_id", "qualification-20260912-01-foreign12"),
    ("source_manifest_sha256", "f" * 64), ("rank", 1), ("device", "cuda:1"),
])
def test_semantics_rejects_when_identity_is_substituted(
    semantics_fixture: RuntimeFixture, field: str, value: str | int,
) -> None:
    # Given: an independently bound temporary sidecar.
    runtime_checks.run_runtime_checks(semantics_fixture.config, 0)
    sidecar = read_sidecar(semantics_fixture.config.output_dir)
    payload = sidecar.model_dump()
    payload["binding"][field] = value
    # When/Then: reparsing rejects controller or rank/device disagreement.
    with pytest.raises(ValidationError):
        SemanticsSidecar.model_validate(payload)
