from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from bytecode_test_support import RELEASE_ROOT, V4_RELEASE, guard_path, owned_bytecode


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
QUALIFICATION_DIR = RELEASE_ROOT / "scale/experiments/nonlatent_iclr/qualification"
MODEL_ROOT = RELEASE_ROOT / "model_source/scale"
MANIFEST_PATH = QUALIFICATION_DIR / "runtime_manifest.json"
INVENTORY_PATH = QUALIFICATION_DIR / "model_source_inventory.json"


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

    original_size_bytes: int
    original_sha256: str
    owned_path: Path
    owned_size_bytes: int
    owned_sha256: str


class SourceInventory(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    owned_root: Path
    entries: tuple[SourceEntry, ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(release: Path) -> RuntimeManifest:
    path = release / "scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json"
    return RuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def test_manifest_binds_source_clean_v5_and_preserves_all_identity_roles() -> None:
    # Given: the v4 identity baseline and the source-clean candidate metadata.
    previous = load_manifest(V4_RELEASE)
    candidate = load_manifest(RELEASE_ROOT)
    inventory = SourceInventory.model_validate_json(
        INVENTORY_PATH.read_text(encoding="utf-8")
    )
    manifested = {entry.path: entry for entry in candidate.files}
    changed_payloads = (
        QUALIFICATION_DIR / "run_qualification.sh",
        QUALIFICATION_DIR / "bytecode_guard.sh",
        QUALIFICATION_DIR / "runtime_config.py",
        INVENTORY_PATH,
    )

    # When: roles, owned source mappings, and external identities are projected.
    roles = Counter(entry.role for entry in candidate.files)
    staged_sources = frozenset(
        entry.path for entry in candidate.files if entry.role == "staged_source"
    )
    external_before = {
        (entry.path, entry.role): (entry.sha256, entry.size_bytes)
        for entry in previous.files
        if entry.role in EXTERNAL_ROLES
    }
    external_after = {
        (entry.path, entry.role): (entry.sha256, entry.size_bytes)
        for entry in candidate.files
        if entry.role in EXTERNAL_ROLES
    }

    # Then: one helper extends the 109 identities while every prior role remains.
    assert len(candidate.files) == 110
    assert roles == {
        "payload": 24,
        "staged_source": 72,
        "package_source": 8,
        "hf_config": 1,
        "hf_index": 1,
        "hf_shard": 2,
        "checkpoint_meta": 1,
        "checkpoint_model": 1,
    }
    assert inventory.owned_root == MODEL_ROOT
    assert len(inventory.entries) == 72
    assert staged_sources == frozenset(entry.owned_path for entry in inventory.entries)
    assert external_after == external_before
    assert owned_bytecode(RELEASE_ROOT) == ()
    for path in changed_payloads:
        record = manifested[path]
        assert record.role == "payload"
        assert record.size_bytes == path.stat().st_size
        assert record.sha256 == sha256_file(path)
    for source in inventory.entries:
        assert source.owned_path.is_relative_to(MODEL_ROOT)
        assert source.original_size_bytes == source.owned_size_bytes
        assert source.original_sha256 == source.owned_sha256
        assert source.owned_size_bytes == source.owned_path.stat().st_size
        assert source.owned_sha256 == sha256_file(source.owned_path)


def test_v5_preserves_runtime_contract_and_origin_controls() -> None:
    # Given: the candidate launcher, guard, and accepted v4 model boundary.
    launcher = (QUALIFICATION_DIR / "run_qualification.sh").read_text(
        encoding="utf-8"
    )
    guard = guard_path(RELEASE_ROOT).read_text(encoding="utf-8")
    model_checks = (QUALIFICATION_DIR / "model_checks.py").read_text(
        encoding="utf-8"
    )

    # When/Then: bytecode isolation leaves topology, deadlines, verdict, and model intact.
    assert "readonly EXPECTED_WORLD_SIZE=8" in launcher
    assert "readonly WORKFLOW_TIMEOUT_SECONDS=1650" in launcher
    assert "readonly KILL_GRACE_SECONDS=30" in launcher
    assert '--nproc-per-node="${EXPECTED_WORLD_SIZE}" --max-restarts=0' in launcher
    assert "terminal_status=PARTIAL_QUALIFICATION" in launcher
    assert "terminal_detail=corevalid/fullmodelcacheunqualified" in launcher
    assert "BYTECODE_PRECHECK_CLEAN" in guard
    assert "BYTECODE_PRECHECK_REJECTED" in guard
    assert "MODEL_LOOP_RANGE: Final = (16, 32)" in model_checks
    assert "MODEL_LOOP_REPS: Final = 1" in model_checks
    assert "model_class.from_hf_pretrained(" in model_checks
    assert "dtype=torch_module.bfloat16" in model_checks
    assert "loop_range=MODEL_LOOP_RANGE" in model_checks
    assert "loop_reps=MODEL_LOOP_REPS" in model_checks
    assert "foreign_preloaded_model_module" in model_checks
    assert "unmanifested_loaded_model_module" in model_checks
    assert "StateInjection" not in model_checks
