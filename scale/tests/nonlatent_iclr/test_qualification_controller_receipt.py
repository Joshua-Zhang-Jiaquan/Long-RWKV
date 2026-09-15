from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
import time

from scale.experiments.nonlatent_iclr.qualification.controller_receipt import (
    ControllerReceipt,
    ControllerReceiptError,
    ControllerReceiptExpectation,
    await_controller_receipt,
)


type JsonScalar = str | int | bool | None

_JOB_ID = "job-12345678-1234-4abc-8def-1234567890ab"
_RUN_ID = "qualification-20260912-01-deadbeef"
_NONCE = "a" * 32
_MANIFEST_SHA256 = "b" * 64


def _receipt_payload() -> dict[str, JsonScalar]:
    return {
        "schema_version": 1,
        "job_id": _JOB_ID,
        "job_identity_origin": "controller_qz_createjob_receipt",
        "submitted_request_sha256": "c" * 64,
        "authorization_id": "nonlatent-h100-qualification-20260912-01",
        "project_id": "project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
        "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
        "logic_compute_group_id": "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
        "spec_id": "7166bd2e-6cbe-4bd9-be38-762d11003e7f",
        "run_id": _RUN_ID,
        "nonce": _NONCE,
        "source_manifest_sha256": _MANIFEST_SHA256,
        "requested_image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
        "scheduler_image_id": "image-7330d118-df9d-4e2e-82b1-c2543e831eb4",
        "observed_image_digest": None,
        "authorized_gpu_type": "NVIDIA_H100_SXM_80G",
        "requested_nodes": 1,
        "requested_gpus_per_node": 8,
        "requested_gpus": 8,
        "maximum_runtime_seconds": 1_800,
        "maximum_gpu_hours": 4,
        "maximum_job_submissions": 1,
        "maximum_automatic_retries": 0,
    }


def _expectation(path: Path, *, timeout_seconds: float = 0.0) -> ControllerReceiptExpectation:
    return ControllerReceiptExpectation(
        path=path,
        run_id=_RUN_ID,
        nonce=_NONCE,
        source_manifest_sha256=_MANIFEST_SHA256,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=0.01,
    )


def _write_receipt(path: Path, payload: dict[str, JsonScalar]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _error_code(
    expectation: ControllerReceiptExpectation,
    environment: dict[str, str] | None = None,
) -> str:
    try:
        await_controller_receipt(expectation, environment=environment or {})
    except ControllerReceiptError as error:
        return error.code
    raise AssertionError("invalid controller receipt was accepted")


def test_controller_receipt_accepts_honest_unknown_digest(tmp_path: Path) -> None:
    # Given: a controller-published CreateJob receipt with no observable image digest.
    receipt_path = tmp_path / "controller-receipt.json"
    _write_receipt(receipt_path, _receipt_payload())

    # When: runtime validates the immutable authorization and launch bindings.
    receipt = await_controller_receipt(_expectation(receipt_path), environment={})

    # Then: the real job id is trusted from the receipt without inventing a digest.
    assert receipt.job_id == _JOB_ID
    assert receipt.job_identity_origin == "controller_qz_createjob_receipt"
    assert receipt.observed_image_digest is None
    fabricated = _receipt_payload()
    fabricated["observed_image_digest"] = "f" * 64
    _write_receipt(receipt_path, fabricated)
    assert _error_code(_expectation(receipt_path)) == "controller_receipt_malformed"


def test_controller_receipt_rejects_native_job_id_mismatch(tmp_path: Path) -> None:
    # Given: a valid receipt whose job id disagrees with a scheduler-native variable.
    receipt_path = tmp_path / "controller-receipt.json"
    _write_receipt(receipt_path, _receipt_payload())

    # When: the runtime compares both independent job identities.
    code = _error_code(
        _expectation(receipt_path),
        {"QZ_JOB_ID": "job-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
    )

    # Then: the mismatch fails closed before any CUDA work.
    assert code == "native_job_identity_mismatch"


def test_controller_receipt_rejects_malformed_or_extra_fields(tmp_path: Path) -> None:
    # Given: a receipt carrying an unreviewed field despite otherwise valid JSON.
    receipt_path = tmp_path / "controller-receipt.json"
    payload = _receipt_payload()
    payload["unreviewed_claim"] = True
    _write_receipt(receipt_path, payload)

    # When/Then: strict parsing rejects the controller artifact.
    assert _error_code(_expectation(receipt_path)) == "controller_receipt_malformed"


def test_controller_receipt_rejects_inconsistent_runtime_binding(
    tmp_path: Path,
) -> None:
    # Given: a well-shaped receipt bound to a different nonce than this launch.
    receipt_path = tmp_path / "controller-receipt.json"
    payload = _receipt_payload()
    payload["nonce"] = "d" * 32
    _write_receipt(receipt_path, payload)

    # When/Then: schema validity cannot override the expected runtime binding.
    assert _error_code(_expectation(receipt_path)) == "controller_receipt_nonce_mismatch"


def test_controller_receipt_rejects_changed_authorized_limits(tmp_path: Path) -> None:
    # Given: a controller artifact that requests an automatic retry.
    receipt_path = tmp_path / "controller-receipt.json"
    payload = _receipt_payload()
    payload["maximum_automatic_retries"] = 1
    _write_receipt(receipt_path, payload)

    # When/Then: the exact no-retry authorization remains a schema invariant.
    assert _error_code(_expectation(receipt_path)) == "controller_receipt_malformed"


def test_controller_receipt_times_out_when_missing(tmp_path: Path) -> None:
    # Given: the expected shared-GPFS receipt path has not been published.
    expectation = _expectation(tmp_path / "missing.json")

    # When/Then: bounded waiting fails closed without touching CUDA.
    assert _error_code(expectation) == "controller_receipt_timeout"


def test_controller_receipt_waits_for_atomic_publication(tmp_path: Path) -> None:
    # Given: a controller that publishes through an atomic rename after launch begins.
    receipt_path = tmp_path / "controller-receipt.json"

    def publish_later() -> None:
        time.sleep(0.02)
        temporary = tmp_path / ".controller-receipt.tmp"
        _write_receipt(temporary, _receipt_payload())
        temporary.replace(receipt_path)

    publisher = Thread(target=publish_later)
    publisher.start()

    # When: runtime waits within the explicit sub-minute receipt budget.
    receipt = await_controller_receipt(
        _expectation(receipt_path, timeout_seconds=0.5), environment={}
    )
    publisher.join(timeout=1.0)

    # Then: only the final atomically published receipt is consumed.
    assert receipt == ControllerReceipt.model_validate(_receipt_payload())
    assert not publisher.is_alive()
