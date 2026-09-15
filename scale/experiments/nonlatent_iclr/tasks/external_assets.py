"""Fail-closed inventory of non-generated Task4 requirements."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .qualification_adapters import qualify


@dataclass(frozen=True, slots=True)
class ExternalRequirement:
    """A declared local asset with an acceptance rule, never a fabricated fixture."""

    requirement: str
    relative_path: str
    acceptance: str


@dataclass(frozen=True, slots=True)
class AssetStatus:
    """Observed state of one declared external requirement."""

    requirement: str
    ready: bool
    reason: str


REQUIREMENTS = (
    ExternalRequirement("ruler_development_suite", "ruler/development_manifest.json", "pinned version, license, and source hash"),
    ExternalRequirement("longbench_development_suite", "longbench/development_manifest.json", "pinned version, license, and source hash"),
    ExternalRequirement("rights_cleared_repository_tasks", "repository_tasks/manifest.json", "200 task records with rights, family, and source hash"),
    ExternalRequirement("isolated_repository_evaluator", "repository_evaluator/isolation_manifest.json", "real externally managed VM/container evaluator declaration"),
    ExternalRequirement("rwkv_tokenizer", "tokenizer/rwkv_tokenizer_manifest.json", "local RWKV tokenizer identity and token-length procedure"),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def inventory_external_assets(asset_root: Path) -> tuple[AssetStatus, ...]:
    """Inspect declared paths without fetching, executing, or assuming licenses."""
    return tuple(_inspect(asset_root, requirement) for requirement in REQUIREMENTS)



def _inspect(asset_root: Path, requirement: ExternalRequirement) -> AssetStatus:
    path = asset_root / requirement.relative_path
    if not path.is_file():
        return AssetStatus(requirement.requirement, False, f"BLOCKED: missing declared local asset {path}")
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return AssetStatus(requirement.requirement, False, f"BLOCKED: unreadable manifest {path}")
    if not isinstance(raw, dict):
        return AssetStatus(requirement.requirement, False, f"BLOCKED: manifest is not an object {path}")
    parsed = {str(key): value for key, value in cast("dict[object, object]", raw).items()}
    if not _has_provenance(parsed):
        return AssetStatus(requirement.requirement, False, f"BLOCKED: incomplete or empty provenance manifest {path}")
    match requirement.requirement:
        case "ruler_development_suite" | "longbench_development_suite":
            failure = _bound_asset_failure(path, parsed)
        case "rights_cleared_repository_tasks":
            failure = _repository_task_failure(path, parsed)
        case "isolated_repository_evaluator":
            failure = _evaluator_failure(path, parsed)
        case "rwkv_tokenizer":
            failure = _tokenizer_failure(path, parsed)
        case other:
            return AssetStatus(other, False, "BLOCKED: undeclared external requirement")
    if failure is not None:
        return AssetStatus(requirement.requirement, False, f"BLOCKED: {failure}")
    qualified, detail = qualify(requirement.requirement, path, parsed)
    if qualified:
        return AssetStatus(requirement.requirement, True, f"QUALIFIED: {detail}")
    return AssetStatus(requirement.requirement, False, f"BLOCKED_{detail}")


def _has_provenance(manifest: dict[str, object]) -> bool:
    digest = manifest.get("sha256")
    return all(_nonempty_text(manifest.get(key)) for key in ("version", "license", "sha256")) and isinstance(digest, str) and _SHA256.fullmatch(digest) is not None


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _bound_asset_failure(path: Path, manifest: dict[str, object]) -> str | None:
    asset_path = manifest.get("asset_path")
    if not isinstance(asset_path, str) or not asset_path:
        return "missing local asset path"
    asset = path.parent / asset_path
    if not asset.is_file():
        return "declared local asset is absent"
    digest = sha256(asset.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        return "declared hash is not bound to local asset bytes"
    return None


def _repository_task_failure(path: Path, manifest: dict[str, object]) -> str | None:
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        return "requires exactly 200 repository task records"
    records = cast("list[object]", raw_records)
    if len(records) != 200:
        return "requires exactly 200 repository task records"
    identities: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            return "repository task record is not an object"
        record = cast("dict[object, object]", item)
        required = ("id", "license", "provenance", "repository_family", "split", "source_path", "source_sha256")
        if not all(_nonempty_text(record.get(key)) for key in required):
            return "repository task record lacks rights or split provenance"
        identity = str(record.get("id"))
        if identity in identities:
            return "repository task identities are not unique"
        identities.add(identity)
        source = path.parent / str(record.get("source_path"))
        if not source.is_file() or sha256(source.read_bytes()).hexdigest() != record.get("source_sha256"):
            return "repository task source hash is not bound to local bytes"
    return None


def _evaluator_failure(path: Path, manifest: dict[str, object]) -> str | None:
    evidence_path = manifest.get("isolation_evidence_path")
    if not isinstance(evidence_path, str) or not evidence_path:
        return "missing bound isolation evidence"
    bound: dict[str, object] = {"asset_path": evidence_path, "sha256": manifest.get("sha256")}
    return _bound_asset_failure(path, bound)


def _tokenizer_failure(path: Path, manifest: dict[str, object]) -> str | None:
    procedure = manifest.get("qualification_procedure")
    results_path = manifest.get("results_path")
    if not isinstance(procedure, str) or not procedure or not isinstance(results_path, str) or not results_path:
        return "missing explicit token-length qualification procedure or results"
    bound: dict[str, object] = {"asset_path": results_path, "sha256": manifest.get("sha256")}
    return _bound_asset_failure(path, bound)
