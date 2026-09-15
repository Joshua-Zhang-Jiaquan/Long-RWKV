from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


V6_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-06-global-ffaa464d-adapter-v6"
)
V7_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260914-07-global-ffaa464d-lifecycle-v7"
)
RUNTIME_REVIEW = Path(
    ".omo/evidence/nonlatent-rnn-dlm-iclr-research/"
    "qualification-20260913-06/runtime-review"
)
QUALIFICATION = Path("scale/experiments/nonlatent_iclr/qualification")
LIFECYCLE_HASHES = {
    "lifecycle.py": "c4b305313b1ea3a0dfb738e06341e46ddb7a032ad769b2850628a288d2d712aa",
    "lifecycle_checks.py": "a83addc0a6ef8961830a01f874992e221ad06e635bbfda0a3e59ff574929dd4c",
    "lifecycle_evidence.py": "c2ed4371f6582cdd8f4900b46db3c3c03bf8034dc4c21872971e0ab720e39088",
    "lifecycle_binding.py": "8b31a422ca36234b55de0be0bd73345964310a6d67429c47a6e9952d4341877c",
    "lifecycle_settings.py": "df922a709cbd55fa3cb2ad74bc815731e89b818bcd106b7a0ba22d2c6b07026e",
    "lifecycle_sidecar.py": "6de17a53cf9c025cd06632289ade8512e2b35bf5f9c41807d8ac0e10cd8e7a4c",
    "lifecycle_runtime.py": "83f8a35d14b20da5ee795b63e8f779dd06721ec86d651152a8bab3fba951d783",
}
V6_ANCHORS = {
    "runtime_checks.py": "c34753e6ed0450f8697a9fda761cc3a3571b3c3c51e4d1d159459a4b98cb6dde",
    "model_checks.py": "6ddb94ddffc01d0fa416dcd7b7bfc0cc9dde201a864ccaf84e338201ab7ec023",
    "bytecode_guard.sh": "b5bcff58019e09a2acee0bf0ea6b4839371f91c3e25c32982d56f47f5c3b8a09",
    "runtime_manifest.json": "692d1cc88d6b2c6335736d47492aa933bf57b9a23a4d0dcc9e5e3eeda165cf32",
}
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def load_manifest(root: Path) -> RuntimeManifest:
    payload = (root / QUALIFICATION / "runtime_manifest.json").read_text(
        encoding="utf-8"
    )
    return RuntimeManifest.model_validate_json(payload)


def external_identities(manifest: RuntimeManifest) -> dict[tuple[Path, str], tuple[str, int]]:
    return {
        (entry.path, entry.role): (
            entry.sha256,
            entry.size_bytes,
        )
        for entry in manifest.files
        if entry.role in EXTERNAL_ROLES
    }


def test_candidate_delta_is_exact_lifecycle_successor() -> None:
    # Given: immutable adapter-v6 and its fresh source-clean successor copy.
    before = source_files(V6_ROOT)
    after = source_files(V7_ROOT)

    # When: file identities are compared independently of release-root modes.
    added = set(after) - set(before)
    removed = set(before) - set(after)
    changed = {path for path in before.keys() & after.keys() if before[path] != after[path]}

    # Then: only approved lifecycle code, caller, and path-bound metadata differ.
    assert added == {str(QUALIFICATION / name) for name in LIFECYCLE_HASHES}
    assert removed == set()
    assert changed == {
        str(QUALIFICATION / "runtime_checks.py"),
        str(QUALIFICATION / "model_source_inventory.json"),
        str(QUALIFICATION / "run_qualification.sh"),
        str(QUALIFICATION / "runtime_config.py"),
        str(QUALIFICATION / "runtime_manifest.json"),
    }


def test_candidate_manifest_binds_approved_modules_and_parent_layout() -> None:
    # Given: the exact approved lifecycle identities and successor manifest.
    manifest = load_manifest(V7_ROOT)
    entries = {entry.path: entry for entry in manifest.files}
    roles = Counter(entry.role for entry in manifest.files)

    # When/Then: all seven modules and package parents are release-owned payloads.
    assert len(manifest.files) == 118
    assert roles["payload"] == 32
    for name, expected_hash in LIFECYCLE_HASHES.items():
        path = V7_ROOT / QUALIFICATION / name
        assert sha256_file(path) == expected_hash
        assert entries[path].role == "payload"
        assert (entries[path].sha256, entries[path].size_bytes) == (
            expected_hash,
            path.stat().st_size,
        )
    for relative in (
        Path("scale/__init__.py"),
        Path("scale/experiments/__init__.py"),
        Path("scale/experiments/nonlatent_iclr/__init__.py"),
        QUALIFICATION / "__init__.py",
    ):
        assert entries[V7_ROOT / relative].role == "payload"


def test_v6_guards_external_identities_and_core_schemas_remain_exact() -> None:
    # Given: pinned v6/review anchors and both manifests.
    v6_manifest = load_manifest(V6_ROOT)
    v7_manifest = load_manifest(V7_ROOT)

    # When/Then: predecessor anchors, release guards, external bytes, and core schemas stay exact.
    assert {
        name: sha256_file(V6_ROOT / QUALIFICATION / name)
        for name in V6_ANCHORS
    } == V6_ANCHORS
    assert sha256_file(RUNTIME_REVIEW / "verdict.json") == (
        "492d71e497cd2b8f91b5507af134ff333c318d97f9f0724b55b36052940dd681"
    )
    assert sha256_file(RUNTIME_REVIEW / "validation.json") == (
        "c320d46f1ab34a8aec4257e0a88780ba624b10dda89d74cccd43398a2ecc1724"
    )
    assert external_identities(v7_manifest) == external_identities(v6_manifest)
    for name in ("model_checks.py", "bytecode_guard.sh", "contracts.py", "controls.py"):
        assert sha256_file(V7_ROOT / QUALIFICATION / name) == sha256_file(
            V6_ROOT / QUALIFICATION / name
        )


def test_candidate_is_source_clean_and_inventory_is_rebound() -> None:
    # Given/When: candidate bytecode and inventory ownership are inspected.
    bytecode = tuple(
        path
        for path in V7_ROOT.rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    )
    inventory = json.loads(
        (V7_ROOT / QUALIFICATION / "model_source_inventory.json").read_text(
            encoding="utf-8"
        )
    )

    # Then: no owned bytecode exists and all 72 source entries belong to v7.
    assert bytecode == ()
    assert inventory["owned_root"] == str(V7_ROOT / "model_source/scale")
    assert len(inventory["entries"]) == 72
    assert all(
        str(entry["owned_path"]).startswith(f"{inventory['owned_root']}/")
        for entry in inventory["entries"]
    )
