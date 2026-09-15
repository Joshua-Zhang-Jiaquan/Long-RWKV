from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[5]
OLD_RELEASE = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-05-global-92fbf9af"
)
RELEASE = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-05-global-ffaa464d-v2"
)
QUALIFICATION = Path("scale/experiments/nonlatent_iclr/qualification")
MANIFEST = QUALIFICATION / "runtime_manifest.json"
ALIAS = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "m2_baseline_triangle/m4loop_endpoint_ckpt"
)
HF_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "models/RWKV7-Goose-World3-2.9B-HF"
)
EXPECTED_CHECKPOINT_HASHES = {
    "meta.json": "321ce0465b2cedb13c870c1ad4ec2c7c36d14cd37670e9caa018b9e63954e7ef",
    "model.pt": "ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07",
}
EXPECTED_CHECKPOINT_SIZES = {"meta.json": 43, "model.pt": 16_367_167_378}

type ManifestRole = Literal[
    "payload",
    "staged_source",
    "package_source",
    "hf_config",
    "hf_index",
    "hf_shard",
    "checkpoint_meta",
    "checkpoint_model",
]


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    role: ManifestRole
    sha256: str
    size_bytes: int


class PackageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution: str
    version: str


class RuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    payload_id: str
    expected_image: str
    files: tuple[ManifestFile, ...]
    packages: tuple[PackageIdentity, ...]


def _manifest(root: Path) -> RuntimeManifest:
    return RuntimeManifest.model_validate_json(
        (root / MANIFEST).read_text(encoding="utf-8")
    )


def _release_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(RELEASE),
        }
    )
    return environment


def test_manifest_extends_complete_prior_payload_with_environment_helper() -> None:
    # Given: the complete prior worker release and its versioned alias successor.
    old = _manifest(OLD_RELEASE)
    new = _manifest(RELEASE)
    old_relatives = {
        entry.path.relative_to(OLD_RELEASE)
        for entry in old.files
        if entry.role == "payload"
    }

    # When: the new payload inventory is interpreted below its own release root.
    new_relatives = {
        entry.path.relative_to(RELEASE)
        for entry in new.files
        if entry.role == "payload"
    }

    # Then: only the stdlib diagnostic helper extends the proven 40-file closure.
    assert new_relatives == old_relatives | {QUALIFICATION / "runtime_environment.py"}
    assert len(new.files) == 41


def test_manifest_aliases_exact_reviewed_checkpoint_bytes() -> None:
    # Given: the two checkpoint identities in the versioned manifest.
    manifest = _manifest(RELEASE)
    entries = {
        entry.path.name: entry
        for entry in manifest.files
        if entry.role in {"checkpoint_meta", "checkpoint_model"}
    }

    # When: paths, hashes, and sizes are projected by checkpoint filename.
    observed = {
        name: (entry.path.parent, entry.sha256, entry.size_bytes)
        for name, entry in entries.items()
    }

    # Then: the neutral copy changes only the parent path, never byte identity.
    assert observed == {
        name: (ALIAS, EXPECTED_CHECKPOINT_HASHES[name], EXPECTED_CHECKPOINT_SIZES[name])
        for name in ("meta.json", "model.pt")
    }


def test_noncheckpoint_external_manifest_records_are_preserved() -> None:
    # Given: old and new manifests with payload and checkpoint paths excluded.
    old = _manifest(OLD_RELEASE)
    new = _manifest(RELEASE)
    excluded = {"payload", "checkpoint_meta", "checkpoint_model"}

    # When: unchanged external records are selected in manifest order.
    old_external = tuple(entry for entry in old.files if entry.role not in excluded)
    new_external = tuple(entry for entry in new.files if entry.role not in excluded)

    # Then: staged source, package source, HF records, image, and versions are exact.
    assert new_external == old_external
    assert new.packages == old.packages
    assert new.expected_image == old.expected_image


def test_manifest_contains_no_project_payload_fallback() -> None:
    # Given: all paths in the versioned runtime manifest.
    manifest = _manifest(RELEASE)
    project_prefix = str(PROJECT_ROOT) + "/"

    # When: payload ownership and absent-project references are inspected.
    payload_paths = tuple(entry.path for entry in manifest.files if entry.role == "payload")

    # Then: payloads are release-owned and no manifest path names the project checkout.
    assert all(path.is_relative_to(RELEASE) for path in payload_paths)
    assert all(not str(entry.path).startswith(project_prefix) for entry in manifest.files)


def test_constructor_configuration_matches_alias_metadata_and_hf_geometry() -> None:
    # Given: the actual staged constructor call, checkpoint metadata, and HF config.
    source = (RELEASE / QUALIFICATION / "model_checks.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    constructor_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_hf_pretrained"
    )
    metadata = json.loads((ALIAS / "meta.json").read_text(encoding="utf-8"))
    hf_config = json.loads((HF_ROOT / "config.json").read_text(encoding="utf-8"))

    # When: machine-consumed constructor keywords and bound dimensions are projected.
    keyword_names = {keyword.arg for keyword in constructor_calls[0].keywords}
    geometry = tuple(
        hf_config[name]
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "vocab_size",
            "head_dim",
            "intermediate_size",
        )
    )

    # Then: loop/dtype inputs, step identity, token count, and geometry remain reviewed values.
    assert len(constructor_calls) == 1
    assert keyword_names == {"dtype", "loop_range", "loop_reps"}
    assert metadata == {"step": 4750, "tokens_seen": 4_980_736_000.0}
    assert geometry == (32, 2_560, 65_536, 64, 10_240)


def test_global_only_config_and_model_contract_import_without_runtime_libraries(
    tmp_path: Path,
) -> None:
    # Given: an isolated interpreter whose sole application path is the new release.
    command = (
        "import json,sys; "
        "from scale.experiments.nonlatent_iclr.qualification import model_checks,runtime_config; "
        "print(json.dumps({"
        "\"checkpoint\":str(runtime_config.CHECKPOINT_DIR),"
        "\"loop_range\":model_checks.MODEL_LOOP_RANGE,"
        "\"loop_reps\":model_checks.MODEL_LOOP_REPS,"
        "\"modules\":[str(model_checks.__file__),str(runtime_config.__file__)],"
        "\"runtime_imports\":[name for name in (\"torch\",\"fla\",\"transformers\") if name in sys.modules],"
        "\"sys_path\":sys.path}))"
    )

    # When: configuration and constructor contracts load without model construction.
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    observed = json.loads(result.stdout)

    # Then: all code/config comes from global roots and no runtime library is imported.
    assert result.returncode == 0
    assert observed["checkpoint"] == str(ALIAS)
    assert observed["loop_range"] == [16, 32]
    assert observed["loop_reps"] == 1
    assert all(str(path).startswith(str(RELEASE)) for path in observed["modules"])
    assert observed["runtime_imports"] == []
    assert str(PROJECT_ROOT) not in observed["sys_path"]


def test_global_only_cli_rejects_bad_input_without_cuda(tmp_path: Path) -> None:
    # Given: the real release-only CLI and a complete request with an invalid nonce.
    arguments = [
        sys.executable,
        "-m",
        "scale.experiments.nonlatent_iclr.qualification.cli",
        "validate",
        "--output-dir",
        str(tmp_path / "unused"),
        "--nonce",
        "bad",
    ]

    # When: the CPU boundary runs outside the project checkout.
    result = subprocess.run(
        arguments,
        cwd=tmp_path,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: the release emits its structured validation failure without CUDA imports.
    assert result.returncode == 2
    assert json.loads(result.stdout) == {
        "detail": "nonce_must_be_32_lower_hex_characters",
        "status": "INVALID_ARGUMENT",
    }
