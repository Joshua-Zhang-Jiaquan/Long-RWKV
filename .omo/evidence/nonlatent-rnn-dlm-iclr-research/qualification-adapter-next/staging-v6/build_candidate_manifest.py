#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2,<3"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv or project install needed):
#      uv run build_candidate_manifest.py
# 3. In the pinned qualification environment used here:
#      PYTHONDONTWRITEBYTECODE=1 /usr/bin/python build_candidate_manifest.py
# ──────────────────

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


V5_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-05-global-ffaa464d-bytecode-v5"
)
V6_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-06-global-ffaa464d-adapter-v6"
)
QUALIFICATION = Path("scale/experiments/nonlatent_iclr/qualification")
FULL_CANVAS = Path("scale/experiments/nonlatent_iclr/full_canvas.py")
FULL_CANVAS_SHA256 = "1e5400e4283744d6c29ce533f0138b2b3e2b7bcd78b1a64b668a79f97c6b3e22"


class CandidateManifestError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: Path
    role: str
    sha256: str
    size_bytes: int


class PackagePin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    distribution: str
    version: str


class ManifestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    payload_id: str
    expected_image: str
    packages: tuple[PackagePin, ...]
    files: tuple[ManifestEntry, ...]


class InventoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    module: str
    original_path: Path
    original_size_bytes: int
    original_sha256: str
    owned_path: Path
    owned_size_bytes: int
    owned_sha256: str
    classification: Literal["runtime_reachable", "copied_inactive_package_member"]


class InventoryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    original_root: Path
    owned_root: Path
    entrypoints: tuple[str, ...]
    reviewer_static_candidate_count: int
    direct_local_dependencies_outside_models: tuple[str, ...]
    unavailable_local_dependencies: tuple[str, ...]
    entries: tuple[InventoryEntry, ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebind_owned_path(path: Path) -> Path:
    if path.is_relative_to(V5_ROOT):
        return V6_ROOT / path.relative_to(V5_ROOT)
    return path


def write_model(path: Path, model: BaseModel) -> None:
    serialized = json.dumps(model.model_dump(mode="json"), indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")


def refresh_entry(entry: ManifestEntry) -> ManifestEntry:
    path = rebind_owned_path(entry.path)
    if path.is_relative_to(V6_ROOT):
        return entry.model_copy(
            update={
                "path": path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entry


def stage_inventory() -> None:
    source = V5_ROOT / QUALIFICATION / "model_source_inventory.json"
    destination = V6_ROOT / QUALIFICATION / "model_source_inventory.json"
    inventory = InventoryDocument.model_validate_json(source.read_text(encoding="utf-8"))
    entries = tuple(
        entry.model_copy(update={"owned_path": rebind_owned_path(entry.owned_path)})
        for entry in inventory.entries
    )
    staged = inventory.model_copy(
        update={"owned_root": V6_ROOT / "model_source/scale", "entries": entries}
    )
    write_model(destination, staged)


def stage_manifest() -> None:
    source = V5_ROOT / QUALIFICATION / "runtime_manifest.json"
    destination = V6_ROOT / QUALIFICATION / "runtime_manifest.json"
    manifest = ManifestDocument.model_validate_json(source.read_text(encoding="utf-8"))
    parent_init = V5_ROOT / "scale/experiments/nonlatent_iclr/__init__.py"
    adapter_path = V6_ROOT / FULL_CANVAS
    adapter = ManifestEntry(
        path=adapter_path,
        role="payload",
        sha256=sha256_file(adapter_path),
        size_bytes=adapter_path.stat().st_size,
    )
    files: list[ManifestEntry] = []
    for entry in manifest.files:
        files.append(refresh_entry(entry))
        if entry.path == parent_init:
            files.append(adapter)
    staged = manifest.model_copy(update={"files": tuple(files)})
    paths = tuple(entry.path for entry in staged.files)
    if len(staged.files) != 111 or len(paths) != len(set(paths)):
        raise CandidateManifestError("candidate_manifest_cardinality_invalid")
    if adapter.sha256 != FULL_CANVAS_SHA256:
        raise CandidateManifestError("approved_full_canvas_identity_mismatch")
    write_model(destination, staged)


def main() -> None:
    stage_inventory()
    stage_manifest()
    manifest = V6_ROOT / QUALIFICATION / "runtime_manifest.json"
    inventory = V6_ROOT / QUALIFICATION / "model_source_inventory.json"
    print(f"runtime_manifest_sha256={sha256_file(manifest)}")
    print(f"model_source_inventory_sha256={sha256_file(inventory)}")


if __name__ == "__main__":
    main()
