from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

import authority_models
import controller_models as models
import request_contract as subject


GLOBAL_RELEASE_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-06-global-ffaa464d-adapter-v6"
)
GLOBAL_LAUNCHER: Final = GLOBAL_RELEASE_ROOT / (
    "scale/experiments/nonlatent_iclr/qualification/run_qualification.sh"
)
GLOBAL_MANIFEST: Final = GLOBAL_RELEASE_ROOT / (
    "scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json"
)
EXPECTED_LAUNCH_COMMAND: Final = f"/usr/bin/bash {GLOBAL_LAUNCHER}"
EXPECTED_MANIFEST_SHA256: Final = (
    "692d1cc88d6b2c6335736d47492aa933bf57b9a23a4d0dcc9e5e3eeda165cf32"
)


def test_request_contract_binds_worker_visible_global_launcher() -> None:
    # Given / When
    command_schema = subject.CreateJobRequest.model_json_schema()["properties"][
        "command"
    ]

    # Then
    assert GLOBAL_LAUNCHER.is_file()
    assert subject.LAUNCH_COMMAND == EXPECTED_LAUNCH_COMMAND
    assert command_schema["const"] == EXPECTED_LAUNCH_COMMAND
    assert "/inspire/hdd/project/" not in subject.LAUNCH_COMMAND


def test_manifest_contract_binds_reviewed_global_release() -> None:
    # Given / When
    actual_digest = sha256(GLOBAL_MANIFEST.read_bytes()).hexdigest()
    permit_schema = authority_models.Permit.model_json_schema()["properties"][
        "source_manifest_sha256"
    ]

    # Then
    assert actual_digest == EXPECTED_MANIFEST_SHA256
    assert models.MANIFEST_SHA256 == EXPECTED_MANIFEST_SHA256
    assert permit_schema["const"] == EXPECTED_MANIFEST_SHA256
