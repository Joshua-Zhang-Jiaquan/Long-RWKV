from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict


V5_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-05-global-ffaa464d-bytecode-v5"
)
V6_ROOT = Path(
    os.environ.get(
        "QUALIFICATION_RELEASE_UNDER_TEST",
        "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
        "nonlatent_iclr_qualification/payloads/"
        "qualification-20260913-06-global-ffaa464d-adapter-v6",
    )
)
QUALIFICATION = Path("scale/experiments/nonlatent_iclr/qualification")
FULL_CANVAS = Path("scale/experiments/nonlatent_iclr/full_canvas.py")
FULL_CANVAS_SHA256 = "1e5400e4283744d6c29ce533f0138b2b3e2b7bcd78b1a64b668a79f97c6b3e22"
V5_MANIFEST_SHA256 = "9271a2036e84c3fa29442a9f7627c9889d024106b8286321d82da26df856e51d"
V5_MODEL_CHECKS_SHA256 = "2883ecbb48dadd093180513a51858541db5fcc8efef712bd55fa88e1374ee1a4"
V5_LAUNCHER_SHA256 = "112ab2595a9db913a40c3976a96998ccf54f9fb7a9680da1ed841cf871dbd52b"
V5_GUARD_SHA256 = "b5bcff58019e09a2acee0bf0ea6b4839371f91c3e25c32982d56f47f5c3b8a09"
EXTERNAL_ROLES = frozenset(
    {
        "package_source",
        "hf_config",
        "hf_index",
        "hf_shard",
        "checkpoint_meta",
        "checkpoint_model",
    }
)


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    path: Path
    role: str
    sha256: str
    size_bytes: int


class RuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    files: tuple[ManifestFile, ...]


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    module: str
    original_sha256: str
    original_size_bytes: int
    owned_path: Path
    owned_sha256: str
    owned_size_bytes: int


class SourceInventory(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    owned_root: Path
    entries: tuple[SourceEntry, ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(root: Path) -> RuntimeManifest:
    return RuntimeManifest.model_validate_json(
        (root / QUALIFICATION / "runtime_manifest.json").read_text(encoding="utf-8")
    )


def source_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def test_candidate_delta_is_only_approved_adapter_packaging() -> None:
    # Given: frozen hardened v5 and its separately staged successor.
    before = source_files(V5_ROOT)
    after = source_files(V6_ROOT)

    # When: content identities are compared independently of absolute ownership.
    added = set(after) - set(before)
    removed = set(before) - set(after)
    changed = {path for path in before.keys() & after.keys() if before[path] != after[path]}

    # Then: only the approved source, caller, and path-bound metadata differ.
    assert added == {str(FULL_CANVAS)}
    assert removed == set()
    assert changed == {
        str(QUALIFICATION / "model_checks.py"),
        str(QUALIFICATION / "model_source_inventory.json"),
        str(QUALIFICATION / "run_qualification.sh"),
        str(QUALIFICATION / "runtime_config.py"),
        str(QUALIFICATION / "runtime_manifest.json"),
    }


def test_candidate_manifest_binds_adapter_and_preserves_v5_identities() -> None:
    # Given: the v5 identity baseline and candidate manifest/inventory.
    previous = load_manifest(V5_ROOT)
    candidate = load_manifest(V6_ROOT)
    inventory_path = V6_ROOT / QUALIFICATION / "model_source_inventory.json"
    inventory = SourceInventory.model_validate_json(
        inventory_path.read_text(encoding="utf-8")
    )
    manifested = {entry.path: entry for entry in candidate.files}

    # When: role and external/staged identity projections are compared.
    roles = Counter(entry.role for entry in candidate.files)
    external = lambda manifest: {
        (entry.path, entry.role): (entry.sha256, entry.size_bytes)
        for entry in manifest.files
        if entry.role in EXTERNAL_ROLES
    }
    staged = lambda manifest: sorted(
        (entry.sha256, entry.size_bytes)
        for entry in manifest.files
        if entry.role == "staged_source"
    )

    # Then: full_canvas is a fresh parent payload and every prior identity remains.
    assert len(candidate.files) == 111
    assert roles == {
        "payload": 25,
        "staged_source": 72,
        "package_source": 8,
        "hf_config": 1,
        "hf_index": 1,
        "hf_shard": 2,
        "checkpoint_meta": 1,
        "checkpoint_model": 1,
    }
    adapter = V6_ROOT / FULL_CANVAS
    assert adapter.is_file()
    assert sha256_file(adapter) == FULL_CANVAS_SHA256
    assert manifested[adapter].role == "payload"
    assert manifested[adapter].sha256 == FULL_CANVAS_SHA256
    assert manifested[adapter].size_bytes == adapter.stat().st_size
    assert external(candidate) == external(previous)
    assert staged(candidate) == staged(previous)
    assert inventory.owned_root == V6_ROOT / "model_source/scale"
    assert len(inventory.entries) == 72
    assert all(
        entry.original_sha256 == entry.owned_sha256
        and entry.original_size_bytes == entry.owned_size_bytes
        and entry.owned_path.is_relative_to(inventory.owned_root)
        for entry in inventory.entries
    )


def test_frozen_v5_anchors_and_candidate_bytecode_guard_remain_exact() -> None:
    # Given/When: immutable predecessor and inherited guard hashes are recalculated.
    anchors = {
        "manifest": sha256_file(V5_ROOT / QUALIFICATION / "runtime_manifest.json"),
        "model_checks": sha256_file(V5_ROOT / QUALIFICATION / "model_checks.py"),
        "launcher": sha256_file(V5_ROOT / QUALIFICATION / "run_qualification.sh"),
        "guard": sha256_file(V5_ROOT / QUALIFICATION / "bytecode_guard.sh"),
    }
    candidate_bytecode = tuple(
        path
        for path in V6_ROOT.rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    )

    # Then: v5 stays pinned and v6 inherits the exact bytecode implementation cleanly.
    assert anchors == {
        "manifest": V5_MANIFEST_SHA256,
        "model_checks": V5_MODEL_CHECKS_SHA256,
        "launcher": V5_LAUNCHER_SHA256,
        "guard": V5_GUARD_SHA256,
    }
    assert sha256_file(V6_ROOT / QUALIFICATION / "bytecode_guard.sh") == V5_GUARD_SHA256
    assert candidate_bytecode == ()
