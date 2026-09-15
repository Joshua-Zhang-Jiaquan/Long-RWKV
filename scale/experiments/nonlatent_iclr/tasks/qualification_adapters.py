"""Trusted, replay-based qualification adapters for declared Task4 external assets.

Each adapter recomputes the evidence it is asked to trust: it re-hashes local bytes,
compares the declared pin against the frozen ``pinned_sources`` table, and replays
provenance through the accepted modules. A declaration that is merely self-consistent
is not evidence, so ``BLOCKED_<CODE>`` reasons name exactly what failed.

This module must never import the network-capable acquisition module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final, cast

from .pinned_sources import (
    ALLOWED_HOSTS, ALLOWED_LICENSES, PinnedSourceError, SUITE_DIRECTORIES, source_for,
)
from .provenance import repository_split_map
from .repository_evaluator import probe_isolation, stable_projection
from .tokenizer_provenance import verify_qualification

AdapterResult = tuple[bool, str]
Adapter = Callable[[str, Path, dict[str, object]], AdapterResult]

TOKENIZER_SIDECAR: Final = "rwkv_tokenizer_qualification.json"
REPOSITORY_TASK_COUNT: Final = 200
EVALUATOR_SCOPES: Final = frozenset({"process_sandbox_not_externally_managed", "externally_managed_vm"})


def _text(value: object) -> str | None:
    """Return a non-empty string value or None, so declarations are never assumed."""
    return value if isinstance(value, str) and value.strip() else None


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> object | None:
    try:
        return cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_object(path: Path) -> dict[str, object] | None:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        return None
    return {str(key): value for key, value in cast("dict[object, object]", raw).items()}


def _entries(value: object) -> list[object]:
    return cast("list[object]", value) if isinstance(value, list) else []


def qualify_pinned_suite(requirement: str, path: Path, manifest: dict[str, object]) -> AdapterResult:
    """Replay a commit-pinned benchmark suite against the frozen pin table."""
    try:
        pinned = source_for(requirement)
    except PinnedSourceError:
        return False, "UNSUPPORTED: no frozen pin exists for this suite"
    asset_path = _text(manifest.get("asset_path"))
    bundle = _load_object(path.parent / asset_path) if asset_path else None
    if bundle is None:
        return False, "UNREADABLE: bound bundle is absent or not a JSON object"
    upstream = bundle.get("upstream")
    if not isinstance(upstream, dict):
        return False, "SHAPE: bundle declares no upstream block"
    declared = cast("dict[object, object]", upstream)
    declared_commit = _text(declared.get("commit"))
    if declared_commit != pinned.commit:
        return False, f"PIN: commit {declared_commit!r} is not the frozen commit {pinned.commit}"
    declared_license = _text(declared.get("license"))
    if declared_license != pinned.license or declared_license not in ALLOWED_LICENSES:
        return False, f"LICENSE: {declared_license!r} is not the frozen allowlisted license {pinned.license}"
    if _text(declared.get("host")) not in ALLOWED_HOSTS:
        return False, f"HOST: {declared.get('host')!r} is not an allowlisted host"
    if _text(declared.get("repo")) != pinned.repo:
        return False, f"PIN: repo {declared.get('repo')!r} is not the frozen repo {pinned.repo}"
    declared_files = _entries(bundle.get("files"))
    if not declared_files:
        return False, "SHAPE: bundle declares no files"
    frozen = {entry.dest: entry.sha256 for entry in pinned.files}
    seen: set[str] = set()
    for item in declared_files:
        if not isinstance(item, dict):
            return False, "SHAPE: file entry is not an object"
        entry = cast("dict[object, object]", item)
        dest = _text(entry.get("dest"))
        digest = _text(entry.get("sha256"))
        if dest is None or digest is None:
            return False, "SHAPE: file entry lacks a destination or hash"
        seen.add(dest)
        target = path.parent / dest
        if not target.is_file():
            return False, f"MISSING: {dest} is declared but absent locally"
        if _digest(target) != digest:
            return False, f"BINDING: {dest} does not match the declared hash"
    if seen != set(frozen):
        return False, f"PIN: bundle file set {sorted(seen)} is not the frozen file set"
    generators = [name for name in _entries(bundle.get("generators")) if isinstance(name, str)]
    if not generators:
        return False, "GENERATOR: bundle pins no data generator"
    for generator in generators:
        if generator not in frozen:
            return False, f"GENERATOR: {generator!r} is not a pinned file"
    return True, f"pinned {pinned.repo}@{pinned.commit[:12]} {declared_license}, {len(seen)} files and {len(generators)} generator(s) replayed"


def qualify_tokenizer(_requirement: str, path: Path, _manifest: dict[str, object]) -> AdapterResult:
    """Replay tokenizer provenance, and the representative length matrix when declared."""
    sidecar = _load_object(path.parent / TOKENIZER_SIDECAR)
    if sidecar is None:
        return False, f"REPLAY: {TOKENIZER_SIDECAR} is absent, so no provenance can be replayed"
    model_root = _text(sidecar.get("model_root"))
    evidence_root = _text(sidecar.get("evidence_root"))
    receipt_path = _text(sidecar.get("receipt_path"))
    if model_root is None or evidence_root is None or receipt_path is None:
        return False, "SHAPE: sidecar must declare model_root, evidence_root and receipt_path"
    if not verify_qualification(Path(model_root), Path(evidence_root), Path(receipt_path)):
        return False, "REPLAY: tokenizer provenance did not replay against the local model"
    artifact = _text(sidecar.get("length_artifact_path"))
    if artifact:
        # Imported lazily: length_qualification reaches task_registry, which reaches back here.
        from .length_qualification import QualificationInputs, verify_length_artifact

        inputs = QualificationInputs(Path(model_root), Path(evidence_root), Path(receipt_path))
        if not verify_length_artifact(Path(artifact), inputs):
            return False, "REPLAY: representative length artifact did not replay"
        return True, "vocabulary, provenance and representative token lengths replayed"
    return True, "vocabulary and provenance replayed; representative token lengths not yet bound"


def qualify_repository_tasks(_requirement: str, path: Path, manifest: dict[str, object]) -> AdapterResult:
    """Validate rights, split inheritance, hash binding, and replay a task sample."""
    records = _entries(manifest.get("records"))
    if len(records) != REPOSITORY_TASK_COUNT:
        return False, f"SHAPE: exactly {REPOSITORY_TASK_COUNT} repository task records are required"
    families: dict[str, str] = {}
    parsed: list[dict[object, object]] = []
    for item in records:
        if not isinstance(item, dict):
            return False, "SHAPE: repository task record is not an object"
        record = cast("dict[object, object]", item)
        parsed.append(record)
        license_id = _text(record.get("license"))
        if license_id not in ALLOWED_LICENSES:
            return False, f"LICENSE: {license_id!r} is not allowlisted"
        family = _text(record.get("repository_family"))
        split = _text(record.get("split"))
        if family is None or split is None:
            return False, "SHAPE: record lacks a repository family or split"
        existing = families.setdefault(family, split)
        if existing != split:
            return False, f"SPLIT: family {family} straddles {existing!r} and {split!r}"
        for key in ("source_path", "evaluator_path"):
            relative = _text(record.get(key))
            digest = _text(record.get(f"{key.removesuffix('_path')}_sha256"))
            if relative is None or digest is None:
                return False, f"SHAPE: record lacks a bound {key}"
            target = path.parent / relative
            if not target.is_file() or _digest(target) != digest:
                return False, f"BINDING: {relative} is not bound to its declared hash"
    covered = set(families.values())
    if "validation" not in covered or "confirmation" not in covered:
        return False, (
            "SPLIT: no held-out repository family exists, so the corpus cannot support "
            f"held-out evaluation; {len(families)} families cover only {sorted(covered)}"
        )
    weights: dict[str, int] = {}
    for record in parsed:
        family = _text(record.get("repository_family"))
        if family is not None:
            weights[family] = weights.get(family, 0) + 1
    expected_splits = repository_split_map(weights)
    for record in parsed:
        family = str(record.get("repository_family"))
        if record.get("split") != expected_splits.get(family):
            return False, f"SPLIT: family {family} does not match the replayed split assignment"
    from .repository_evaluator import verify_task

    for index, record in enumerate(parsed):
        if index % 20:
            continue
        source = (path.parent / str(record.get("source_path"))).read_text(encoding="utf-8")
        check = (path.parent / str(record.get("evaluator_path"))).read_text(encoding="utf-8")
        entry = _text(record.get("entry_point"))
        qualified, reason = verify_task(source, check, entry_point=entry)
        if not qualified:
            return False, f"REPLAY: sampled task {record.get('id')!r} did not discriminate: {reason}"
    held_out = sum(1 for record in parsed if record.get("split") != "train")
    return True, f"{len(parsed)} records replayed, {held_out} held out across {sorted(covered)}"


def qualify_evaluator(_requirement: str, path: Path, manifest: dict[str, object]) -> AdapterResult:
    """Replay the live isolation probe against the evidence the declaration binds."""
    scope = _text(manifest.get("scope"))
    if scope not in EVALUATOR_SCOPES:
        return False, f"SCOPE: {scope!r} is not a declared isolation scope"
    if scope == "externally_managed_vm":
        return False, "SCOPE: this host provides no externally managed VM or container"
    backend = _text(manifest.get("isolation_backend"))
    if backend is None:
        return False, "SHAPE: isolation backend is not declared"
    evidence_path = _text(manifest.get("isolation_evidence_path"))
    if evidence_path is None:
        return False, "SHAPE: isolation evidence path is not declared"
    bound = _load_object(path.parent / evidence_path)
    if bound is None:
        return False, "UNREADABLE: bound isolation evidence is absent or not a JSON object"
    live = stable_projection(probe_isolation())
    declared = stable_projection(bound)
    if live != declared:
        changed = sorted(name for name in live if live[name] != declared.get(name))
        return False, f"REPLAY: live isolation probe no longer matches the bound evidence on {changed}"
    return True, f"isolation replayed for scope {scope} via {backend}"


ADAPTERS: Mapping[str, Adapter] = {
    "ruler_development_suite": qualify_pinned_suite,
    "longbench_development_suite": qualify_pinned_suite,
    "rwkv_tokenizer": qualify_tokenizer,
    "rights_cleared_repository_tasks": qualify_repository_tasks,
    "isolated_repository_evaluator": qualify_evaluator,
}


def qualify(requirement: str, path: Path, manifest: dict[str, object]) -> AdapterResult:
    """Dispatch to the adapter for a declared requirement; unknown requirements fail closed."""
    adapter = ADAPTERS.get(requirement)
    if adapter is None:
        return False, "UNSUPPORTED: no adapter is registered for this requirement"
    return adapter(requirement, path, manifest)


__all__ = [
    "ADAPTERS", "Adapter", "AdapterResult", "EVALUATOR_SCOPES", "REPOSITORY_TASK_COUNT",
    "SUITE_DIRECTORIES", "TOKENIZER_SIDECAR", "qualify", "qualify_evaluator",
    "qualify_pinned_suite", "qualify_repository_tasks", "qualify_tokenizer",
]
