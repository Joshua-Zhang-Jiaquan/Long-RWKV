from __future__ import annotations

from typing import Literal, assert_never

import pytest

from scale.experiments.nonlatent_iclr.qualification import lifecycle_runtime, runtime_checks
from scale.experiments.nonlatent_iclr.qualification.lifecycle_evidence import LifecycleEvidence
from scale.experiments.nonlatent_iclr.qualification.lifecycle_sidecar import LifecycleSidecar
from scale.experiments.nonlatent_iclr.qualification.lifecycle_settings import LoadedLifecycleModel, ModelSettings
from scale.tests.nonlatent_iclr.lifecycle_runtime_fixtures import (
    RuntimeFixture, runtime_fixture as runtime_fixture, synthetic_lifecycle,
)


def test_runtime_routes_lifecycle_before_resources_when_model_loaded(runtime_fixture: RuntimeFixture) -> None:
    # Given: CPU seam doubles; synthetic CUDA coordinates are not GPU observations.
    fixture = runtime_fixture
    # When: the actual editable runtime caller runs its existing orchestration.
    result = runtime_checks.run_runtime_checks(fixture.config, 0)
    # Then: the already-loaded instance crosses the lifecycle seam before peak sampling.
    assert fixture.events.index("masked") < fixture.events.index("lifecycle") < fixture.events.index("peak")
    assert fixture.events.count("reset_peak") == 1
    assert result.status == "PASSED"
    assert (fixture.config.output_dir / "lifecycle-0.json").is_file()
    assert len(fixture.synchronize_devices) == 3
    assert all(str(device) == "cuda:0" for device in fixture.synchronize_devices)


@pytest.mark.parametrize("mutation", ["partial", "contradictory", "foreign_device"])
def test_runtime_preserves_sidecar_before_failure_when_lifecycle_fails(
    runtime_fixture: RuntimeFixture, monkeypatch: pytest.MonkeyPatch,
    mutation: Literal["partial", "contradictory", "foreign_device"],
) -> None:
    # Given: partial or contradictory fixture observations returned by the probe seam.
    payload = synthetic_lifecycle().model_dump()
    match mutation:
        case "partial":
            payload.update(completed=False, failure_code="value_mismatch", model_calls=2, synchronizations=2)
            payload["steps"] = payload["steps"][:1]
        case "contradictory":
            payload["steps"] = ()
        case "foreign_device":
            payload["device"] = "cuda:1"
        case unreachable:
            assert_never(unreachable)
    observation = LifecycleEvidence.model_validate(payload)
    monkeypatch.setattr(lifecycle_runtime, "probe_lifecycle", lambda *args: observation, raising=False)
    # When: the actual runtime caller receives nonpassing lifecycle evidence.
    with pytest.raises(runtime_checks.RuntimeCheckFailure, match="lifecycle"):
        runtime_checks.run_runtime_checks(runtime_fixture.config, 0)
    # Then: observations survive, final resources/success are not reached, teardown still runs.
    assert (runtime_fixture.config.output_dir / "lifecycle-0.json").is_file()
    sidecar = LifecycleSidecar.model_validate_json((runtime_fixture.config.output_dir / "lifecycle-0.json").read_bytes())
    assert sidecar.observations == observation and not sidecar.passed
    assert "peak" not in runtime_fixture.events


def test_runtime_preserves_completed_observations_when_settings_change(
    runtime_fixture: RuntimeFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a post-probe settings observation that differs from its entry state.
    original = lifecycle_runtime.observe_settings
    snapshots: list[ModelSettings] = []

    def changing_settings(model: LoadedLifecycleModel) -> ModelSettings:
        snapshot = original(model)
        if snapshots:
            snapshot = ModelSettings.model_validate({**snapshot.model_dump(), "training": True})
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(lifecycle_runtime, "observe_settings", changing_settings)
    # When: the caller compares actual before/after observations.
    with pytest.raises(runtime_checks.RuntimeCheckFailure, match="lifecycle"):
        runtime_checks.run_runtime_checks(runtime_fixture.config, 0)
    # Then: completed probe data survives but changed execution conditions prevent success.
    sidecar = LifecycleSidecar.model_validate_json((runtime_fixture.config.output_dir / "lifecycle-0.json").read_bytes())
    assert sidecar.observations is not None and sidecar.observations.passed
    assert sidecar.before == snapshots[0] and sidecar.after == snapshots[1] and not sidecar.passed


def test_runtime_reparses_observations_when_typed_instance_bypasses_validation(
    runtime_fixture: RuntimeFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an invalid instance that was constructed without Pydantic validation.
    invalid = synthetic_lifecycle().model_copy(update={"runtime_promoted": True})
    monkeypatch.setattr(lifecycle_runtime, "probe_lifecycle", lambda *args: invalid)
    # When: the caller reparses serialized observations instead of trusting the instance.
    with pytest.raises(runtime_checks.RuntimeCheckFailure, match="lifecycle"):
        runtime_checks.run_runtime_checks(runtime_fixture.config, 0)
    # Then: rejected bytes remain available without converting them into valid evidence.
    sidecar = LifecycleSidecar.model_validate_json((runtime_fixture.config.output_dir / "lifecycle-0.json").read_bytes())
    assert sidecar.observations is None and sidecar.rejected_observation_json is not None
    assert sidecar.issue is not None and sidecar.issue.code == "invalid_observation"
    assert not sidecar.passed


def test_runtime_preserves_foreign_temporary_when_publication_is_blocked(runtime_fixture: RuntimeFixture) -> None:
    # Given: a temporary file not owned by this invocation.
    config = runtime_fixture.config
    config.output_dir.mkdir()
    temporary = config.output_dir / f".lifecycle-0.{config.nonce}.tmp"
    temporary.write_text("foreign temporary fixture", encoding="utf-8")
    # When: exclusive creation fails before temporary ownership is acquired.
    with pytest.raises(runtime_checks.RuntimeCheckFailure):
        runtime_checks.run_runtime_checks(config, 0)
    # Then: cleanup cannot delete another writer's file or yield success.
    assert temporary.read_text(encoding="utf-8") == "foreign temporary fixture"
    assert "peak" not in runtime_fixture.events
    assert runtime_fixture.events[-1] == "teardown"


def test_runtime_cannot_pass_when_sidecar_destination_exists(runtime_fixture: RuntimeFixture) -> None:
    # Given: an existing sidecar that must never be overwritten or reused.
    output = runtime_fixture.config.output_dir
    output.mkdir()
    destination = output / "lifecycle-0.json"
    destination.write_text("preserved foreign fixture", encoding="utf-8")
    # When: lifecycle publication attempts an exclusive write.
    with pytest.raises(runtime_checks.RuntimeCheckFailure):
        runtime_checks.run_runtime_checks(runtime_fixture.config, 0)
    # Then: existing bytes survive and no rank success or resource finalization occurs.
    assert destination.read_text(encoding="utf-8") == "preserved foreign fixture"
    assert "peak" not in runtime_fixture.events
