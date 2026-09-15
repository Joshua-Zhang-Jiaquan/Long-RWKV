from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path

from pydantic import JsonValue
import pytest

from scale.experiments.nonlatent_iclr.qualification import controls
from scale.experiments.nonlatent_iclr.qualification.controls import QualificationInputError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    tmp_path: Path, package_version: str | None = None
) -> tuple[Path, str, Path, int]:
    contents = {
        "payload.py": "payload-v1\n",
        "staged.py": "staged-v1\n",
        "package_api.py": "api-v1\n",
        "config.json": "{}\n",
        "model-00001-of-00002.safetensors": "shard-one\n",
        "model-00002-of-00002.safetensors": "shard-two\n",
        "meta.json": '{"step":4750}\n',
        "model.pt": "checkpoint\n",
    }
    for name, content in contents.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    roles = {
        "payload.py": "payload",
        "staged.py": "staged_source",
        "package_api.py": "package_source",
        "config.json": "hf_config",
        "model.safetensors.index.json": "hf_index",
        "model-00001-of-00002.safetensors": "hf_shard",
        "model-00002-of-00002.safetensors": "hf_shard",
        "meta.json": "checkpoint_meta",
        "model.pt": "checkpoint_model",
    }
    files: list[JsonValue] = []
    for name, role in roles.items():
        path = tmp_path / name
        files.append(
            {
                "path": str(path),
                "role": role,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    actual_version = metadata.version("pydantic")
    manifest_payload: JsonValue = {
        "expected_image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
        "files": files,
        "packages": [
            {
                "distribution": "pydantic",
                "version": package_version or actual_version,
            }
        ],
        "payload_id": "qualification-20260912-01",
        "schema_version": 1,
    }
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    return manifest_path, _sha256(manifest_path), tmp_path / "payload.py", len(files)


def test_runtime_manifest_verifies_real_files_and_package_identity(tmp_path: Path) -> None:
    # Given: a signed manifest covering every required role and the installed package version.
    manifest_path, digest, _, file_count = _write_manifest(tmp_path)

    # When: the CPU-only source boundary streams and verifies it.
    verification = controls.verify_runtime_manifest(
        manifest_path, expected_sha256=digest
    )

    # Then: the receipt binds the signed manifest and every listed byte source.
    assert verification.manifest_sha256 == digest
    assert verification.verified_file_count == file_count
    assert verification.package_versions == (f"pydantic=={metadata.version('pydantic')}",)


def test_runtime_manifest_rejects_a_same_path_byte_mutation(tmp_path: Path) -> None:
    # Given: a valid signed manifest whose payload file changes after signing.
    manifest_path, digest, payload_path, _ = _write_manifest(tmp_path)
    payload_path.write_text("payload-v2\n", encoding="utf-8")

    # When / Then: byte verification fails even though the path and size are unchanged.
    with pytest.raises(QualificationInputError, match="manifest_file_sha256_mismatch:payload.py"):
        controls.verify_runtime_manifest(manifest_path, expected_sha256=digest)


def test_runtime_manifest_rejects_an_uninstalled_package_identity(tmp_path: Path) -> None:
    # Given: an internally signed manifest naming the wrong package version.
    manifest_path, digest, _, _ = _write_manifest(tmp_path, package_version="0.0-invalid")

    # When / Then: package identity fails before any model or CUDA import.
    with pytest.raises(QualificationInputError, match="package_version_mismatch:pydantic"):
        controls.verify_runtime_manifest(manifest_path, expected_sha256=digest)
