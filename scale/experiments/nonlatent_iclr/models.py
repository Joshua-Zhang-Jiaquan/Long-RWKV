from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast

from . import Artifact, View
from .schema import load_json_object, validate


class ClaimState(StrEnum):
    OBSERVED = "observed"
    CLAIMED = "claimed"
    CORRECTED = "corrected"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class SourceRoots:
    snapshot: str
    external: str
    live: str
    staged: str


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    source: str
    state: ClaimState
    detail: str
    records: tuple[str, ...]
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Ledger:
    schema_version: int
    attempt: str
    source_roots: SourceRoots
    artifacts: tuple[Artifact, ...]
    claims: tuple[Claim, ...]
    views: tuple[View, ...]
    accounting_note: str


@dataclass(frozen=True, slots=True)
class AuditResult:
    status: str
    detail: str
    blocked_claims: int = 0


def json_text(value: Ledger | AuditResult) -> str:
    return json.dumps(asdict(value), indent=2, sort_keys=True)


def ledger_bytes(ledger: Ledger) -> bytes:
    return (json_text(ledger) + "\n").encode()


def result_json(result: AuditResult) -> str:
    return json_text(result)


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value)


def _mapping_items(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in cast(list[object], value))


def _artifact(item: Mapping[str, object]) -> Artifact:
    return Artifact(
        source=cast(str, item["source"]),
        origin=cast(str, item["origin"]),
        path=cast(str, item["path"]),
        kind=cast(str, item["kind"]),
        identity=cast(str, item["identity"]),
        sha256=cast(str | None, item["sha256"]),
        bytes=cast(int | None, item["bytes"]),
        file_count=cast(int, item["file_count"]),
        reason=cast(str | None, item["reason"]),
    )


def _claim(item: Mapping[str, object]) -> Claim:
    return Claim(
        claim_id=cast(str, item["claim_id"]),
        source=cast(str, item["source"]),
        state=ClaimState(cast(str, item["state"])),
        detail=cast(str, item["detail"]),
        records=tuple(cast(list[str], item["records"])),
        fields=tuple(cast(list[str], item["fields"])),
    )


def _view(item: Mapping[str, object]) -> View:
    return View(
        source=cast(str, item["source"]),
        state=cast(str, item["state"]),
        detail=cast(str, item["detail"]),
    )


def parse_ledger(payload: str) -> Ledger:
    raw = load_json_object(payload)
    error = validate(raw)
    if error is not None:
        raise ValueError(error)
    roots = _mapping(raw["source_roots"])
    return Ledger(
        schema_version=cast(int, raw["schema_version"]),
        attempt=cast(str, raw["attempt"]),
        source_roots=SourceRoots(
            snapshot=cast(str, roots["snapshot"]),
            external=cast(str, roots["external"]),
            live=cast(str, roots["live"]),
            staged=cast(str, roots["staged"]),
        ),
        artifacts=tuple(_artifact(item) for item in _mapping_items(raw["artifacts"])),
        claims=tuple(_claim(item) for item in _mapping_items(raw["claims"])),
        views=tuple(_view(item) for item in _mapping_items(raw["views"])),
        accounting_note=cast(str, raw["accounting_note"]),
    )
