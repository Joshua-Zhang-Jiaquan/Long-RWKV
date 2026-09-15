from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.qualification import probe
from scale.experiments.nonlatent_iclr.qualification.controls import ProbeConfig


@pytest.mark.parametrize(
    "runtime_error",
    (AssertionError("assert_close_failed"), TypeError("api_signature_failed")),
)
def test_probe_boundary_publishes_every_ordinary_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runtime_error: Exception,
) -> None:
    # Given: a valid rank identity and an ordinary check/API exception from the GPU runner.
    monkeypatch.setenv("RANK", "0")

    def fail_runtime(_config: ProbeConfig, _rank: int) -> None:
        raise runtime_error

    monkeypatch.setattr(probe, "_run", fail_runtime)
    arguments = (
        "--output-dir", str(tmp_path),
        "--nonce", "a" * 32,
        "--manifest", str(tmp_path / "runtime-manifest.json"),
        "--manifest-sha256", "b" * 64,
        "--run-id", "qualification-20260912-01-deadbeef",
        "--controller-receipt", str(tmp_path / "controller-receipt.json"),
    )

    # When: the real process boundary executes without importing CUDA libraries.
    exit_code = probe.main(arguments)
    captured = capsys.readouterr()
    payload = json.loads((tmp_path / "rank-0.json").read_text(encoding="utf-8"))

    # Then: it preserves traceback detail and publishes a failed rank atomically.
    assert exit_code == 1
    assert payload["status"] == "FAILED"
    assert payload["evidence"]["kind"] == "failure"
    assert payload["evidence"]["exception_type"] == type(runtime_error).__name__
    assert type(runtime_error).__name__ in captured.err


def test_hardware_match_accepts_driver_visible_h100_name_without_marketing_token() -> None:
    # Given: the exact driver-visible authorized product name observed in this pool.
    observed_name = "NVIDIA H100 80GB HBM3"

    # When / Then: identity relies on H100 plus measured 80 GB capacity, not an SXM substring.
    assert probe.is_authorized_h100(observed_name, 81_559)
    assert not probe.is_authorized_h100("NVIDIA A100-SXM4-80GB", 81_559)
    assert not probe.is_authorized_h100(observed_name, 40_000)


def test_probe_pins_the_checkpoint_training_loop_architecture() -> None:
    # Given / When / Then: model construction uses the checkpoint's recorded loop contract.
    assert probe.MODEL_LOOP_RANGE == (16, 32)
    assert probe.MODEL_LOOP_REPS == 1
