from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict


V3_RELEASE: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/"
    "payloads/qualification-20260913-05-global-ffaa464d-closure-v3"
)
V4_RELEASE: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/"
    "payloads/qualification-20260913-05-global-ffaa464d-origin-v4"
)
RELEASE_ROOT: Final = Path(
    os.environ.get("QUALIFICATION_RELEASE_UNDER_TEST", str(V4_RELEASE))
)
QUALIFICATION_DIR: Final = (
    RELEASE_ROOT / "scale/experiments/nonlatent_iclr/qualification"
)
MODEL_ROOT: Final = RELEASE_ROOT / "model_source/scale"
MANIFEST_PATH: Final = QUALIFICATION_DIR / "runtime_manifest.json"
INVENTORY_PATH: Final = QUALIFICATION_DIR / "model_source_inventory.json"
EXTERNAL_ROLES: Final = frozenset(
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

    original_path: Path
    original_size_bytes: int
    original_sha256: str
    owned_path: Path
    owned_size_bytes: int
    owned_sha256: str


class SourceInventory(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    owned_root: Path
    entries: tuple[SourceEntry, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(root: Path) -> RuntimeManifest:
    path = root / "scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json"
    return RuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def test_manifest_is_fresh_for_origin_fix_and_source_inventory() -> None:
    # Given: the release manifest and its canonical complete source inventory.
    manifest = _load_manifest(RELEASE_ROOT)
    inventory = SourceInventory.model_validate_json(
        INVENTORY_PATH.read_text(encoding="utf-8")
    )
    manifested = {entry.path: entry for entry in manifest.files}
    changed_payloads = (
        QUALIFICATION_DIR / "run_qualification.sh",
        QUALIFICATION_DIR / "model_checks.py",
        QUALIFICATION_DIR / "runtime_config.py",
        INVENTORY_PATH,
    )

    # When: changed payloads and every owned model source are matched to the manifest.
    staged_sources = frozenset(
        entry.path for entry in manifest.files if entry.role == "staged_source"
    )
    mapped_sources = frozenset(entry.owned_path for entry in inventory.entries)

    # Then: metadata and source bytes are fresh, complete, and v4-owned.
    assert len(manifest.files) == 109
    assert inventory.owned_root == MODEL_ROOT
    assert len(inventory.entries) == 72
    assert staged_sources == mapped_sources
    for path in changed_payloads:
        record = manifested[path]
        assert record.role == "payload"
        assert record.size_bytes == path.stat().st_size
        assert record.sha256 == _sha256(path)
    for source in inventory.entries:
        assert source.owned_path.is_relative_to(MODEL_ROOT)
        assert source.original_size_bytes == source.owned_size_bytes
        assert source.original_sha256 == source.owned_sha256
        assert source.owned_size_bytes == source.owned_path.stat().st_size
        assert source.owned_sha256 == _sha256(source.owned_path)


def test_v4_preserves_all_package_hf_and_checkpoint_identities() -> None:
    # Given: v3's independently reviewed external identities and the candidate release.
    previous = _load_manifest(V3_RELEASE)
    candidate = _load_manifest(RELEASE_ROOT)

    # When: entries outside owned payload/model-source paths are projected by identity.
    previous_external = {
        (entry.path, entry.role): (entry.sha256, entry.size_bytes)
        for entry in previous.files
        if entry.role in EXTERNAL_ROLES
    }
    candidate_external = {
        (entry.path, entry.role): (entry.sha256, entry.size_bytes)
        for entry in candidate.files
        if entry.role in EXTERNAL_ROLES
    }

    # Then: packages, HF weights, and neutral checkpoint bytes remain exactly pinned.
    assert candidate_external == previous_external


def test_runtime_contract_remains_bounded_and_nonlatent() -> None:
    # Given: the candidate launcher and model construction boundary.
    launcher = (QUALIFICATION_DIR / "run_qualification.sh").read_text(
        encoding="utf-8"
    )
    model_checks = (QUALIFICATION_DIR / "model_checks.py").read_text(
        encoding="utf-8"
    )

    # When/Then: the origin fix leaves topology, deadlines, verdict, and model recipe intact.
    assert "readonly EXPECTED_WORLD_SIZE=8" in launcher
    assert "readonly WORKFLOW_TIMEOUT_SECONDS=1650" in launcher
    assert "readonly KILL_GRACE_SECONDS=30" in launcher
    assert '--nproc-per-node="${EXPECTED_WORLD_SIZE}" --max-restarts=0' in launcher
    assert "terminal_status=PARTIAL_QUALIFICATION" in launcher
    assert "terminal_detail=corevalid/fullmodelcacheunqualified" in launcher
    assert "MODEL_LOOP_RANGE: Final = (16, 32)" in model_checks
    assert "MODEL_LOOP_REPS: Final = 1" in model_checks
    assert "model_class.from_hf_pretrained(" in model_checks
    assert "dtype=torch_module.bfloat16" in model_checks
    assert "loop_range=MODEL_LOOP_RANGE" in model_checks
    assert "loop_reps=MODEL_LOOP_REPS" in model_checks
    assert "StateInjection" not in model_checks
