from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from closure_audit import (
    all_model_python_sources,
    resolve_local_closure,
    sha256_file,
)


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
MANIFEST_RELATIVE: Final = Path(
    "scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json"
)
ENTRYPOINTS: Final = ("models", "models.birwkv7_diffusion")
INVENTORY_RELATIVE: Final = Path(
    "scale/experiments/nonlatent_iclr/qualification/model_source_inventory.json"
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


class RuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    files: tuple[ManifestFile, ...]


class SourceMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    module: str
    classification: Literal["runtime_reachable", "copied_inactive_package_member"]
    original_path: Path
    original_size_bytes: int
    original_sha256: str
    owned_path: Path
    owned_size_bytes: int
    owned_sha256: str


class ModelSourceInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    original_root: Path
    owned_root: Path
    entrypoints: tuple[str, ...]
    reviewer_static_candidate_count: Literal[64]
    direct_local_dependencies_outside_models: tuple[str, ...]
    unavailable_local_dependencies: tuple[str, ...]
    entries: tuple[SourceMapping, ...]


def _release() -> Path:
    configured = os.environ.get("QUALIFICATION_CLOSURE_RELEASE")
    return V3_RELEASE if configured is None else Path(configured)


def _manifest(release: Path) -> RuntimeManifest:
    return RuntimeManifest.model_validate_json(
        (release / MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )


def _source_root(manifest: RuntimeManifest) -> Path:
    initializers = tuple(
        entry.path
        for entry in manifest.files
        if entry.role == "staged_source"
        and entry.path.as_posix().endswith("models/__init__.py")
    )
    assert len(initializers) == 1
    return initializers[0].parents[1]


def test_state_hijacking_dit_parent_import_is_manifested() -> None:
    # Given: the staged models package parent and entrypoint closure.
    manifest = _manifest(_release())
    staged_sources = tuple(
        entry.path for entry in manifest.files if entry.role == "staged_source"
    )
    source_root = _source_root(manifest)

    # When: local imports are resolved statically from the package parent and entrypoint.
    closure = resolve_local_closure(source_root, ENTRYPOINTS)
    closure_paths = frozenset(module.path for module in closure)

    # Then: the first reviewer-identified parent import is source-bound by the manifest.
    state_hijacking_dit = source_root / "models/state_hijacking_dit.py"
    assert state_hijacking_dit in closure_paths
    assert state_hijacking_dit in staged_sources, (
        f"unmanifested_runtime_source:{state_hijacking_dit}"
    )


def test_static_closure_equals_manifested_runtime_reachable_inventory() -> None:
    # Given: the immutable owned-source inventory and its manifest.
    release = _release()
    manifest = _manifest(release)
    inventory = ModelSourceInventory.model_validate_json(
        (release / INVENTORY_RELATIVE).read_text(encoding="utf-8")
    )

    # When: the two runtime roots are resolved across every local import.
    closure = resolve_local_closure(inventory.owned_root, ENTRYPOINTS)
    closure_modules = tuple(module.module for module in closure)
    classified_modules = tuple(
        sorted(
            entry.module
            for entry in inventory.entries
            if entry.classification == "runtime_reachable"
        )
    )

    # Then: all 64 reviewed candidates are exactly classified and manifest-bound.
    assert inventory.entrypoints == ENTRYPOINTS
    assert len(closure) == inventory.reviewer_static_candidate_count
    assert closure_modules == classified_modules
    assert frozenset(module.path for module in closure).issubset(
        entry.path for entry in manifest.files if entry.role == "staged_source"
    )


def test_complete_models_package_mapping_is_owned_and_byte_identical() -> None:
    # Given: all copied package members and their original-to-owned mappings.
    release = _release()
    manifest = _manifest(release)
    inventory = ModelSourceInventory.model_validate_json(
        (release / INVENTORY_RELATIVE).read_text(encoding="utf-8")
    )
    mapped_owned = frozenset(entry.owned_path for entry in inventory.entries)
    manifested = frozenset(
        entry.path for entry in manifest.files if entry.role == "staged_source"
    )

    # When: actual package members and both sides of every mapping are hashed.
    actual_owned = frozenset(all_model_python_sources(inventory.owned_root))
    mapping_results = tuple(
        (
            entry.original_path.is_file(),
            entry.owned_path.is_file(),
            entry.original_path.is_symlink(),
            entry.owned_path.is_symlink(),
            entry.original_path.stat().st_size,
            entry.owned_path.stat().st_size,
            sha256_file(entry.original_path),
            sha256_file(entry.owned_path),
            entry.original_size_bytes,
            entry.owned_size_bytes,
            entry.original_sha256,
            entry.owned_sha256,
        )
        for entry in inventory.entries
    )

    # Then: all 72 bounded package files are owned, manifested, and exact copies.
    assert len(inventory.entries) == 72
    assert sum(
        entry.classification == "copied_inactive_package_member"
        for entry in inventory.entries
    ) == 8
    assert mapped_owned == actual_owned == manifested
    assert all(
        original_exists
        and owned_exists
        and not original_symlink
        and not owned_symlink
        and original_size == owned_size == expected_original_size == expected_owned_size
        and original_sha256
        == owned_sha256
        == expected_original_sha256
        == expected_owned_sha256
        for (
            original_exists,
            owned_exists,
            original_symlink,
            owned_symlink,
            original_size,
            owned_size,
            original_sha256,
            owned_sha256,
            expected_original_size,
            expected_owned_size,
            expected_original_sha256,
            expected_owned_sha256,
        ) in mapping_results
    )
    assert inventory.direct_local_dependencies_outside_models == ()
    assert inventory.unavailable_local_dependencies == ()


def test_module_origins_resolve_to_owned_tree_without_execution(tmp_path: Path) -> None:
    # Given: an isolated stdlib interpreter and the owned models search root.
    source_root = V3_RELEASE / "model_source/scale"
    command = "\n".join(
        (
            "import json,sys",
            "from importlib.machinery import PathFinder",
            f"root={str(source_root)!r}",
            "package=PathFinder.find_spec('models',[root])",
            "assert package is not None",
            "locations=package.submodule_search_locations",
            "assert locations is not None",
            "module=PathFinder.find_spec('models.birwkv7_diffusion',locations)",
            "assert module is not None",
            "print(json.dumps({'package':package.origin,'module':module.origin,'loaded':[name for name in ('torch','fla','transformers','models') if name in sys.modules]}))",
        )
    )

    # When: package and entrypoint specs are located without importing either module.
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env={
            "CUDA_VISIBLE_DEVICES": "",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(source_root),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    observed = json.loads(result.stdout)

    # Then: both origins are owned and no runtime or CUDA-bearing package executed.
    assert result.returncode == 0
    assert Path(observed["package"]).is_relative_to(source_root)
    assert Path(observed["module"]).is_relative_to(source_root)
    assert observed["loaded"] == []
