from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Final

from pydantic import TypeAdapter

import controller as subject


ROOT: Final = Path(
    "/inspire/hdd/project/multimodal-diffusion-language-model/"
    "zhangjiaquan-253108540222/DiffRWKV-RELAY"
)
SUBMISSION_DIR: Final = ROOT / (
    ".omo/evidence/nonlatent-rnn-dlm-iclr-research/"
    "qualification-20260912-01/submission"
)
CONTROLLER: Final = SUBMISSION_DIR / "controller.py"
AUTHORIZATION: Final = (
    ROOT / ".omo/authorizations/nonlatent-h100-qualification-20260912-01.json"
)
AUTHORIZATION_ID: Final = "nonlatent-h100-qualification-20260912-01"
MANIFEST_SHA256: Final = (
    "92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c"
)
RUN_ID: Final = "qualification-20260912-01-00000000-0000-4000-8000-000000000000"
RECEIPT_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/controller-receipts"
)
STATE_ADAPTER: Final = TypeAdapter(dict[str, str | int])


def _invoke_reserve(state_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CONTROLLER),
            "reserve",
            str(AUTHORIZATION),
            str(state_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _known_state_bytes(state_path: Path, authorization_id: str) -> bytes:
    state = {
        "schema_version": 1,
        "authorization_id": authorization_id,
        "authorization_sha256": sha256(AUTHORIZATION.read_bytes()).hexdigest(),
        "run_id": RUN_ID,
        "nonce": "0" * 32,
        "job_name": RUN_ID,
        "request_path": str(state_path.parent / "exact_job_spec.json"),
        "controller_receipt_path": str(RECEIPT_ROOT / f"{RUN_ID}.json"),
        "source_manifest_sha256": MANIFEST_SHA256,
        "status": "NOT_SUBMITTED",
        "created_at_utc": "2026-09-12T00:00:00Z",
    }
    return json.dumps(state, separators=(",", ":"), sort_keys=True).encode()


def test_reserve_creates_bound_state_when_absent(tmp_path: Path) -> None:
    # Given
    state_path = tmp_path / "controller-state.json"

    # When
    completed = _invoke_reserve(state_path)

    # Then
    assert completed.returncode == 0, completed.stderr
    state = STATE_ADAPTER.validate_json(state_path.read_bytes())
    assert state["authorization_id"] == AUTHORIZATION_ID
    assert state["authorization_sha256"] == sha256(
        AUTHORIZATION.read_bytes()
    ).hexdigest()
    assert state["status"] == "NOT_SUBMITTED"
    assert re.fullmatch(
        r"qualification-20260912-01-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        str(state["run_id"]),
    )
    assert re.fullmatch(r"[0-9a-f]{32}", str(state["nonce"]))
    assert state["job_name"] == state["run_id"]


def test_reserve_reuses_identity_when_state_exists(tmp_path: Path) -> None:
    # Given
    state_path = tmp_path / "controller-state.json"
    original = _known_state_bytes(state_path, AUTHORIZATION_ID)
    state_path.write_bytes(original)

    # When
    completed = _invoke_reserve(state_path)

    # Then
    assert completed.returncode == 0, completed.stderr
    assert state_path.read_bytes() == original


def test_reserve_refuses_mismatched_existing_state(tmp_path: Path) -> None:
    # Given
    state_path = tmp_path / "controller-state.json"
    original = _known_state_bytes(state_path, "other-authorization")
    state_path.write_bytes(original)

    # When
    completed = _invoke_reserve(state_path)

    # Then
    assert completed.returncode != 0
    assert "existing_state_ambiguous" in completed.stderr
    assert state_path.read_bytes() == original


def test_reserve_refuses_malformed_existing_state(tmp_path: Path) -> None:
    # Given
    state_path = tmp_path / "controller-state.json"
    original = b"{}"
    state_path.write_bytes(original)

    # When
    completed = _invoke_reserve(state_path)

    # Then
    assert completed.returncode != 0
    assert "existing_state_ambiguous" in completed.stderr
    assert state_path.read_bytes() == original


def test_load_state_accepts_absolute_path_for_reserved_relative_request() -> None:
    # Given
    state_path = SUBMISSION_DIR / "controller-state.json"

    # When
    state = subject.load_state(state_path)

    # Then
    assert state.run_id == (
        "qualification-20260912-01-7bef6cf8-fda4-46b9-84ff-d7ca4fe5b892"
    )
