from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from .contract import ARTIFACT_CONTRACT, CLAIM_CONTRACTS, SCHEMA_VERSION

HEX = re.compile(r"[0-9a-f]{64}\Z")
ATTEMPT = re.compile(r"[0-9a-f]{32}\Z")
SOURCES = frozenset({"R1", "R2", "R3", "R4", "R5", "R6"})
ORIGINS = frozenset({"snapshot", "external"})
KINDS = frozenset({"document", "raw_record", "code", "manifest", "metadata"})
IDENTITIES = frozenset({"sha256", "missing", "unverified"})
STATES = frozenset({"observed", "claimed", "corrected", "unresolved"})
VIEW_NAMES = frozenset({"snapshot", "live", "staged"})
ROOT_NAMES = frozenset({"snapshot", "external", "live", "staged"})
ENVELOPE = frozenset({
    "schema_version", "attempt", "source_roots", "artifacts", "claims", "views",
    "accounting_note",
})


def _object(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    return cast(Mapping[str, object], value)


def _list(value: object) -> list[object] | None:
    return cast(list[object], value) if isinstance(value, list) else None


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _safe_relative(value: object) -> bool:
    if not _nonempty_string(value):
        return False
    text = cast(str, value)
    path = PurePosixPath(text)
    return (
        not path.is_absolute()
        and path != PurePosixPath(".")
        and ".." not in path.parts
        and "\\" not in text
        and "\x00" not in text
    )


def _string_list(value: object, *, paths: bool = False) -> list[str] | None:
    values = _list(value)
    if values is None or not values:
        return None
    if not all(_safe_relative(item) if paths else _nonempty_string(item) for item in values):
        return None
    strings = cast(list[str], values)
    return strings if len(strings) == len(set(strings)) else None


def _validate_roots(value: object) -> str | None:
    roots = _object(value)
    if roots is None or set(roots) != ROOT_NAMES:
        return "invalid source roots"
    values = tuple(roots[name] for name in sorted(ROOT_NAMES))
    if not all(isinstance(root, str) and Path(root).is_absolute() and "\x00" not in root for root in values):
        return "invalid source roots"
    if len(set(cast(tuple[str, ...], values))) != len(values):
        return "invalid source roots"
    return None


def _validate_artifact(value: object) -> str | None:
    item = _object(value)
    keys = {"source", "origin", "path", "kind", "identity", "sha256", "bytes", "file_count", "reason"}
    if item is None or set(item) != keys:
        return "invalid artifact"
    source, origin, kind, identity = (item[name] for name in ("source", "origin", "kind", "identity"))
    if source not in SOURCES or origin not in ORIGINS or kind not in KINDS or identity not in IDENTITIES:
        return "invalid artifact scalar"
    if not _safe_relative(item["path"]):
        return "invalid artifact path"
    size, count, digest, reason = item["bytes"], item["file_count"], item["sha256"], item["reason"]
    if size is not None and (type(size) is not int or cast(int, size) < 0):
        return "invalid bytes"
    if type(count) is not int or cast(int, count) < 0:
        return "invalid file_count"
    if reason is not None and not _nonempty_string(reason):
        return "invalid reason"
    if identity == "sha256":
        if not isinstance(digest, str) or HEX.fullmatch(digest) is None:
            return "invalid sha256"
        if size is None or cast(int, count) == 0:
            return "invalid hashed identity"
    elif digest is not None:
        return "unexpected sha256"
    elif identity == "missing" and (size is not None or cast(int, count) != 0 or reason is None):
        return "invalid missing identity"
    elif identity == "unverified" and reason is None:
        return "invalid unverified identity"
    return None


def _validate_claim(value: object) -> str | None:
    claim = _object(value)
    keys = {"claim_id", "source", "state", "detail", "records", "fields"}
    if claim is None or set(claim) != keys:
        return "invalid claim"
    if not _nonempty_string(claim["claim_id"]):
        return "invalid claim scalar"
    if claim["source"] not in SOURCES or claim["state"] not in STATES:
        return "invalid claim scalar"
    if not _nonempty_string(claim["detail"]):
        return "invalid claim scalar"
    if _string_list(claim["records"], paths=True) is None or _string_list(claim["fields"]) is None:
        return "invalid claim provenance"
    return None


def _validate_views(value: object) -> str | None:
    views = _list(value)
    if views is None or len(views) != len(VIEW_NAMES):
        return "invalid views coverage"
    names: list[str] = []
    for value_item in views:
        item = _object(value_item)
        if item is None or set(item) != {"source", "state", "detail"}:
            return "invalid view"
        if item["source"] not in VIEW_NAMES or item["state"] not in STATES or not _nonempty_string(item["detail"]):
            return "invalid view scalar"
        names.append(cast(str, item["source"]))
    return None if frozenset(names) == VIEW_NAMES and len(names) == len(set(names)) else "invalid views coverage"


def validate(raw: object) -> str | None:
    ledger = _object(raw)
    if ledger is None or set(ledger) != ENVELOPE:
        return "invalid envelope"
    if type(ledger["schema_version"]) is not int or ledger["schema_version"] != SCHEMA_VERSION:
        return "invalid schema_version"
    attempt = ledger["attempt"]
    if not isinstance(attempt, str) or ATTEMPT.fullmatch(attempt) is None:
        return "invalid attempt"
    root_error = _validate_roots(ledger["source_roots"])
    if root_error is not None:
        return root_error
    if not _nonempty_string(ledger["accounting_note"]):
        return "invalid accounting_note"
    artifacts = _list(ledger["artifacts"])
    if artifacts is None:
        return "invalid artifacts"
    seen: set[tuple[str, str]] = set()
    inventory: set[tuple[str, str, str, str]] = set()
    for value in artifacts:
        error = _validate_artifact(value)
        if error is not None:
            return error
        item = cast(Mapping[str, object], value)
        key = cast(str, item["origin"]), cast(str, item["path"])
        if key in seen:
            return "duplicate artifact"
        seen.add(key)
        inventory.add(
            (
                cast(str, item["source"]),
                key[0],
                key[1],
                cast(str, item["kind"]),
            )
        )
    if inventory != ARTIFACT_CONTRACT or len(artifacts) != len(ARTIFACT_CONTRACT):
        return "invalid required inventory"
    claims = _list(ledger["claims"])
    if claims is None or not claims:
        return "missing claims"
    actual_claims: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for claim_value in claims:
        error = _validate_claim(claim_value)
        if error is not None:
            return error
        claim = cast(Mapping[str, object], claim_value)
        if claim["state"] not in {"claimed", "unresolved"}:
            return "invalid claim classification"
        actual_claims.add(
            (
                cast(str, claim["claim_id"]),
                cast(str, claim["source"]),
                tuple(cast(list[str], claim["records"])),
                tuple(cast(list[str], claim["fields"])),
            )
        )
    required_claims = {
        (item.claim_id, item.source, item.records, item.fields)
        for item in CLAIM_CONTRACTS
    }
    if actual_claims != required_claims or len(claims) != len(required_claims):
        return "invalid claim coverage"
    return _validate_views(ledger["views"])


def validate_receipt(raw: object, *, expected_attempt: str) -> str | None:
    receipt = _object(raw)
    if receipt is None or set(receipt) != {"schema_version", "attempt", "ledger_sha256", "status"}:
        return "invalid receipt envelope"
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        return "invalid receipt schema_version"
    attempt = receipt["attempt"]
    if not isinstance(attempt, str) or ATTEMPT.fullmatch(attempt) is None or attempt != expected_attempt:
        return "invalid receipt attempt"
    digest = receipt["ledger_sha256"]
    if not isinstance(digest, str) or HEX.fullmatch(digest) is None:
        return "invalid receipt sha256"
    if type(receipt["status"]) is not str or receipt["status"] != "AUDIT_COMPLETE":
        return "invalid receipt status"
    return None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def load_json_object(payload: str) -> Mapping[str, object]:
    value = cast(object, json.loads(payload, object_pairs_hook=_unique_object))
    result = _object(value)
    if result is None:
        raise ValueError("JSON root must be an object")
    return result
