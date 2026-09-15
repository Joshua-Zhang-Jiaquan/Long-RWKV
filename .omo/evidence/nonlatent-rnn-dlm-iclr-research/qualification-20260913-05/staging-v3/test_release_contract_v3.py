from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict


V2_RELEASE: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-05-global-ffaa464d-v2"
)
V3_RELEASE: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-05-global-ffaa464d-closure-v3"
)
QUALIFICATION: Final = Path("scale/experiments/nonlatent_iclr/qualification")
MANIFEST: Final = QUALIFICATION / "runtime_manifest.json"
INVENTORY: Final = QUALIFICATION / "model_source_inventory.json"
LAUNCHER: Final = QUALIFICATION / "run_qualification.sh"
MODEL_SOURCE_ROOT: Final = V3_RELEASE / "model_source/scale"
ORIGINAL_STAGED_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "qz_stage_traj4096_v7/scale"
)
CHECKPOINT_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "m2_baseline_triangle/m4loop_endpoint_ckpt"
)
HF_ROOT: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "models/RWKV7-Goose-World3-2.9B-HF"
)


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    role: Literal[
        "payload",
        "staged_source",
        "package_source",
        "hf_config",
        "hf_index",
        "hf_shard",
        "checkpoint_meta",
        "checkpoint_model",
    ]
    sha256: str
    size_bytes: int


class PackageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution: str
    version: str


class RuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    payload_id: str
    expected_image: str
    files: tuple[ManifestFile, ...]
    packages: tuple[PackageIdentity, ...]


def _manifest(root: Path) -> RuntimeManifest:
    return RuntimeManifest.model_validate_json(
        (root / MANIFEST).read_text(encoding="utf-8")
    )


def _environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{V3_RELEASE}:{MODEL_SOURCE_ROOT}",
        }
    )
    return environment


def test_manifest_owns_complete_models_package_and_preserves_external_pins() -> None:
    # Given: the reviewed v2 manifest and its closure-owned successor.
    prior = _manifest(V2_RELEASE)
    current = _manifest(V3_RELEASE)
    roles = Counter(entry.role for entry in current.files)
    current_external = tuple(
        entry for entry in current.files if entry.role not in {"payload", "staged_source"}
    )
    prior_external = tuple(
        entry for entry in prior.files if entry.role not in {"payload", "staged_source"}
    )

    # When: payload and staged-source ownership boundaries are classified.
    payload_paths = tuple(entry.path for entry in current.files if entry.role == "payload")
    staged_paths = tuple(
        entry.path for entry in current.files if entry.role == "staged_source"
    )

    # Then: 72 owned model files replace five mutable bindings and all other pins survive.
    assert len(current.files) == 109
    assert roles == {
        "payload": 23,
        "staged_source": 72,
        "package_source": 8,
        "hf_config": 1,
        "hf_index": 1,
        "hf_shard": 2,
        "checkpoint_meta": 1,
        "checkpoint_model": 1,
    }
    assert all(path.is_relative_to(V3_RELEASE) for path in payload_paths)
    assert all(path.is_relative_to(MODEL_SOURCE_ROOT / "models") for path in staged_paths)
    assert current_external == prior_external
    assert current.packages == prior.packages
    assert current.expected_image == prior.expected_image
    assert V3_RELEASE / INVENTORY in payload_paths
    assert ORIGINAL_STAGED_ROOT.as_posix() not in (V3_RELEASE / MANIFEST).read_text(
        encoding="utf-8"
    )


def test_launcher_and_runtime_config_select_only_owned_model_source(
    tmp_path: Path,
) -> None:
    # Given: the v3 launcher and an isolated configuration import boundary.
    launcher_source = (V3_RELEASE / LAUNCHER).read_text(encoding="utf-8")
    command = (
        "import json,sys; "
        "from scale.experiments.nonlatent_iclr.qualification import runtime_config; "
        "print(json.dumps({"
        "'staged_root':str(runtime_config.STAGED_ROOT),"
        "'checkpoint':str(runtime_config.CHECKPOINT_DIR),"
        "'module':str(runtime_config.__file__),"
        "'loaded':[name for name in ('torch','fla','transformers','models') if name in sys.modules]}))"
    )

    # When: configuration is imported and launcher help is executed without CUDA.
    imported = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    help_result = subprocess.run(
        ["/usr/bin/bash", str(V3_RELEASE / LAUNCHER), "--help"],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    observed = json.loads(imported.stdout)

    # Then: every configured model-source route is owned while endpoint binding is unchanged.
    assert imported.returncode == 0
    assert help_result.returncode == 0
    assert observed == {
        "staged_root": str(MODEL_SOURCE_ROOT),
        "checkpoint": str(CHECKPOINT_ROOT),
        "module": str(V3_RELEASE / QUALIFICATION / "runtime_config.py"),
        "loaded": [],
    }
    assert f"readonly DEFAULT_STAGED_ROOT={MODEL_SOURCE_ROOT}" in launcher_source
    assert ORIGINAL_STAGED_ROOT.as_posix() not in launcher_source


def test_nonlatent_model_constructor_contract_is_byte_identical_to_v2() -> None:
    # Given: the reviewed and closure-owned runtime model constructors.
    prior_model_checks = (V2_RELEASE / QUALIFICATION / "model_checks.py").read_bytes()

    # When: the v3 constructor bytes are selected.
    current_model_checks = (V3_RELEASE / QUALIFICATION / "model_checks.py").read_bytes()

    # Then: the full model remains BiRWKV7 nonlatent despite legacy package imports.
    assert current_model_checks == prior_model_checks
    assert b'model_class = model_module.BiRWKV7ForMaskedDiffusion' in current_model_checks
    assert b'StateInjectionDiTRELAY' not in current_model_checks


def test_stale_owned_model_helper_hash_fails_before_large_blob_hashing(
    tmp_path: Path,
) -> None:
    # Given: a same-size one-byte mutation of a first-position owned helper entry.
    manifest = _manifest(V3_RELEASE)
    target = next(
        entry
        for entry in manifest.files
        if entry.path.name == "state_hijacking_dit_torch_types.py"
    )
    payload = target.path.read_bytes()
    corrupted = tmp_path / target.path.name
    corrupted.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
    replacement = target.model_copy(update={"path": corrupted})
    modified = manifest.model_copy(
        update={
            "files": (
                replacement,
                *(entry for entry in manifest.files if entry.path != target.path),
            )
        }
    )
    modified_path = tmp_path / "runtime_manifest.json"
    modified_path.write_text(modified.model_dump_json(), encoding="utf-8")
    digest = hashlib.sha256(modified_path.read_bytes()).hexdigest()

    # When: the real global-only CPU validator checks the mutated source first.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scale.experiments.nonlatent_iclr.qualification.cli",
            "validate",
            "--output-dir",
            str(tmp_path / "output"),
            "--nonce",
            "d" * 32,
            "--run-id",
            "qualification-20260912-01-v3stale1",
            "--controller-receipt",
            str(tmp_path / "controller.json"),
            "--controller-receipt-timeout-seconds",
            "0",
            "--staged-root",
            str(MODEL_SOURCE_ROOT),
            "--checkpoint-dir",
            str(CHECKPOINT_ROOT),
            "--model-dir",
            str(HF_ROOT),
            "--manifest",
            str(modified_path),
            "--manifest-sha256",
            digest,
        ],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: strict source identity rejects the helper without loading model weights.
    assert result.returncode == 1
    assert json.loads(result.stdout) == {
        "detail": (
            "source_identity:manifest_file_sha256_mismatch:"
            "state_hijacking_dit_torch_types.py"
        ),
        "status": "FAILED",
    }
