"""CPU-only verification of externally pinned qualification input bytes."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Final, Literal, Self
import hashlib

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contracts import HexDigest, Nonce, QualificationInputError


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

EXPECTED_IMAGE: Final = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
REQUIRED_ROLES: Final[frozenset[ManifestRole]] = frozenset(
    {
        "payload",
        "staged_source",
        "package_source",
        "hf_config",
        "hf_index",
        "hf_shard",
        "checkpoint_meta",
        "checkpoint_model",
    }
)
HASH_CHUNK_BYTES: Final = 8 * 1024 * 1024


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: Path
    role: ManifestRole
    sha256: HexDigest
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def require_absolute_path(self) -> Self:
        if not self.path.is_absolute():
            raise QualificationInputError("manifest_file_path_not_absolute")
        return self


class PackageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    distribution: str = Field(min_length=1)
    version: str = Field(min_length=1)


class RuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    payload_id: Literal["qualification-20260912-01"]
    expected_image: Literal[EXPECTED_IMAGE]
    files: tuple[ManifestFile, ...] = Field(min_length=8)
    packages: tuple[PackageIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_unique_inventory(self) -> Self:
        paths = tuple(entry.path for entry in self.files)
        if len(paths) != len(set(paths)):
            raise QualificationInputError("manifest_file_paths_not_unique")
        roles = frozenset(entry.role for entry in self.files)
        if not REQUIRED_ROLES.issubset(roles):
            raise QualificationInputError("manifest_required_roles_missing")
        distributions = tuple(package.distribution for package in self.packages)
        if len(distributions) != len(set(distributions)):
            raise QualificationInputError("manifest_package_names_not_unique")
        return self


class ManifestVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    manifest_sha256: HexDigest
    verified_file_count: int = Field(ge=8)
    package_versions: tuple[str, ...]
    expected_image: Literal[EXPECTED_IMAGE]


class PreflightReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    status: Literal["PREFLIGHT_VERIFIED"] = "PREFLIGHT_VERIFIED"
    detail: Literal["source_manifest_and_controls_verified"] = (
        "source_manifest_and_controls_verified"
    )
    nonce: Nonce
    manifest_sha256: HexDigest
    verified_source_files: int = Field(ge=8)
    package_versions: tuple[str, ...]
    expected_image: Literal[EXPECTED_IMAGE]

    @classmethod
    def from_verification(
        cls, *, nonce: str, verification: ManifestVerification
    ) -> PreflightReceipt:
        return cls(
            nonce=nonce,
            manifest_sha256=verification.manifest_sha256,
            verified_source_files=verification.verified_file_count,
            package_versions=verification.package_versions,
            expected_image=verification.expected_image,
        )


class _HFIndex(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    weight_map: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise QualificationInputError(f"manifest_file_unreadable:{path.name}") from error
    return digest.hexdigest()


def verify_runtime_manifest(
    manifest_path: Path, *, expected_sha256: str
) -> ManifestVerification:
    if manifest_path.is_symlink():
        raise QualificationInputError("manifest_path_is_symlink")
    if sha256_file(manifest_path) != expected_sha256:
        raise QualificationInputError("manifest_digest_mismatch")
    try:
        manifest = RuntimeManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise QualificationInputError("manifest_schema_invalid") from error
    for entry in manifest.files:
        _verify_file(entry)
    _verify_hf_shard_inventory(manifest)
    package_versions = tuple(_verify_package(package) for package in manifest.packages)
    return ManifestVerification(
        manifest_sha256=expected_sha256,
        verified_file_count=len(manifest.files),
        package_versions=package_versions,
        expected_image=manifest.expected_image,
    )


def load_preflight_receipt(
    path: Path, *, nonce: str, manifest_sha256: str
) -> PreflightReceipt:
    try:
        receipt = PreflightReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise QualificationInputError("preflight_receipt_invalid") from error
    if receipt.nonce != nonce:
        raise QualificationInputError("preflight_receipt_nonce_mismatch")
    if receipt.manifest_sha256 != manifest_sha256:
        raise QualificationInputError("preflight_receipt_manifest_mismatch")
    return receipt


def _verify_file(entry: ManifestFile) -> None:
    if entry.path.is_symlink() or not entry.path.is_file():
        raise QualificationInputError(f"manifest_file_unavailable:{entry.path.name}")
    before = entry.path.stat()
    if before.st_size != entry.size_bytes:
        raise QualificationInputError(f"manifest_file_size_mismatch:{entry.path.name}")
    observed_sha256 = sha256_file(entry.path)
    after = entry.path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise QualificationInputError(f"manifest_file_changed_during_hash:{entry.path.name}")
    if observed_sha256 != entry.sha256:
        raise QualificationInputError(f"manifest_file_sha256_mismatch:{entry.path.name}")


def _verify_hf_shard_inventory(manifest: RuntimeManifest) -> None:
    index_entries = tuple(entry for entry in manifest.files if entry.role == "hf_index")
    shard_entries = tuple(entry for entry in manifest.files if entry.role == "hf_shard")
    if len(index_entries) != 1:
        raise QualificationInputError("manifest_hf_index_count_not_1")
    try:
        index = _HFIndex.model_validate_json(index_entries[0].path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise QualificationInputError("manifest_hf_index_invalid") from error
    indexed_names = frozenset(index.weight_map.values())
    manifested_names = frozenset(entry.path.name for entry in shard_entries)
    if indexed_names != manifested_names:
        raise QualificationInputError("manifest_hf_shards_do_not_match_index")


def _verify_package(package: PackageIdentity) -> str:
    try:
        observed_version = metadata.version(package.distribution)
    except metadata.PackageNotFoundError as error:
        raise QualificationInputError(
            f"package_not_installed:{package.distribution}"
        ) from error
    if observed_version != package.version:
        raise QualificationInputError(
            f"package_version_mismatch:{package.distribution}"
        )
    return f"{package.distribution}=={observed_version}"
