"""Item 2 (data_plan.md): canonical record schema and validator.

A canonical record is architecture-independent (data_plan.md §4). This module
defines the schema for the four record types, the forbidden derived-view
field vocabulary, content-addressed ids, lineage links, and a fail-closed
validator.

Record envelope:
  {
    "schema_version": 1,
    "record_type": "document" | "repository" | "episode" | "task",
    "id": "<sha256(record_type + content_hash + source_locator)[:24]>",
    "content_hash": "<blake2b-128 hex of original bytes>",
    "source_locator": "<repo url / dataset name + file + line>",
    "payload": {... original text/code bytes, language, ...},
    "lineage": {"version": 1, "parent_id": null, "correction_reason": null},
    "rights": {"license": str|None, "provenance": str, "acquired_at": iso8601,
               "teacher": null | {"model": str, "prompt_hash": str}},
    "fingerprints": {"exact": "<hex>", "minhash": [80 ints]},
    "split_assignment": null | {"bucket": "train"|"val"|"test",
                                "family": str|None, "rule": str},
    "relations": {... issue/patch/test/env refs; ordered events for episodes ...}
  }

Forbidden (derived-view vocabulary — validator rejects at ANY depth):
  mask, corruption, fim, state_carry, packing, block_t, prompt_end,
  loss_mask, condition_mask, generated_mask, noise_schedule, sep_token_id
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = 1
RECORD_TYPES = ("document", "repository", "episode", "task")
SPLIT_BUCKETS = ("train", "val", "test")

FORBIDDEN_FIELDS = frozenset({
    "mask", "corruption", "fim", "state_carry", "packing", "block_t",
    "prompt_end", "loss_mask", "condition_mask", "generated_mask",
    "noise_schedule", "sep_token_id", "corruption_seed",
})

_REQUIRED_TOP = ("schema_version", "record_type", "id", "content_hash",
                 "source_locator", "payload", "lineage", "rights", "fingerprints")

_ISO8601 = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?(Z|[+-]\d{2}:?\d{2})?$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def content_hash(data: bytes) -> str:
    """blake2b-128 hex of the original bytes — the canonical content hash."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def record_id(record_type: str, chash: str, source_locator: str) -> str:
    """Content-addressed id: sha256(type + hash + locator)[:24]."""
    digest = hashlib.sha256(
        f"{record_type}|{chash}|{source_locator}".encode("utf-8", "replace")
    ).hexdigest()
    return digest[:24]


def _walk_forbidden(obj: Any, path: str = "") -> list[str]:
    """Find forbidden keys at any depth; returns dotted paths."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if str(k).lower() in FORBIDDEN_FIELDS:
                hits.append(here)
            hits.extend(_walk_forbidden(v, here))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_walk_forbidden(v, f"{path}[{i}]"))
    return hits


def validate(record: dict) -> tuple[bool, list[str]]:
    """Fail-closed validation. Returns (ok, errors)."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return False, ["record is not a dict"]
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    rt = record.get("record_type")
    if rt not in RECORD_TYPES:
        errors.append(f"record_type must be one of {RECORD_TYPES}")
    for field in _REQUIRED_TOP:
        if field not in record:
            errors.append(f"missing required field: {field}")
    if errors:
        return False, errors

    if not _HEX32.match(str(record.get("content_hash", ""))):
        errors.append("content_hash must be blake2b-128 hex (32 chars)")
    if not isinstance(record.get("source_locator"), str) or not record["source_locator"]:
        errors.append("source_locator must be a non-empty string")
    if not isinstance(record.get("payload"), dict):
        errors.append("payload must be a dict")
    if not isinstance(record.get("id"), str) or len(record["id"]) != 24:
        errors.append("id must be a 24-char string")
    else:
        expected = record_id(rt, record["content_hash"], record["source_locator"])
        if record["id"] != expected:
            errors.append(f"id mismatch: expected {expected}, got {record['id']}")

    lineage = record.get("lineage")
    if not isinstance(lineage, dict) or "version" not in lineage:
        errors.append("lineage must carry a version")
    else:
        if lineage.get("version") < 1:
            errors.append("lineage.version must be >= 1")
        if lineage.get("version") > 1 and not lineage.get("parent_id"):
            errors.append("corrections (version > 1) must carry parent_id")
        if lineage.get("parent_id") and lineage.get("version") == 1:
            errors.append("version 1 records must not carry parent_id")

    rights = record.get("rights")
    if not isinstance(rights, dict):
        errors.append("rights must be a dict")
    else:
        if not _ISO8601.match(str(rights.get("acquired_at", ""))):
            errors.append("rights.acquired_at must be ISO-8601")
        if "provenance" not in rights:
            errors.append("rights.provenance is required")
        teacher = rights.get("teacher")
        if teacher is not None and not (isinstance(teacher, dict) and "model" in teacher):
            errors.append("rights.teacher must be null or {'model': ...}")

    fps = record.get("fingerprints")
    if not isinstance(fps, dict) or "exact" not in fps:
        errors.append("fingerprints.exact is required")
    if isinstance(fps, dict) and not _HEX64.match(str(fps.get("exact", ""))):
        errors.append("fingerprints.exact must be sha256-style hex (64 chars)")

    split = record.get("split_assignment")
    if split is not None:
        if not isinstance(split, dict):
            errors.append("split_assignment must be null or a dict")
        elif split.get("bucket") not in SPLIT_BUCKETS:
            errors.append(f"split_assignment.bucket must be one of {SPLIT_BUCKETS}")

    if rt == "episode":
        events = (record.get("relations") or {}).get("events")
        if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
            errors.append("episode records must carry relations.events as a list of dicts")

    # payload integrity: the declared content_hash must match the payload
    # bytes — this is what detects a mutated payload (the plan's failure-path QA)
    payload = record.get("payload", {})
    if rt == "document" and isinstance(payload.get("text"), str):
        actual = content_hash(payload["text"].encode("utf-8", "replace"))
        if actual != record.get("content_hash"):
            errors.append(
                "payload mutation detected: content_hash does not match payload.text"
            )

    forbidden = _walk_forbidden(record)
    if forbidden:
        errors.append(f"forbidden derived-view fields: {forbidden[:5]}")
    return (not errors), errors


def make_document(text: str, source_locator: str, license_name: str | None,
                  provenance: str, acquired_at: str) -> dict:
    """Build a well-formed document canonical record (helper for adapters)."""
    data = text.encode("utf-8", "replace")
    chash = content_hash(data)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "document",
        "id": record_id("document", chash, source_locator),
        "content_hash": chash,
        "source_locator": source_locator,
        "payload": {"text": text, "language": None},
        "lineage": {"version": 1, "parent_id": None, "correction_reason": None},
        "rights": {"license": license_name, "provenance": provenance,
                   "acquired_at": acquired_at, "teacher": None},
        "fingerprints": {"exact": hashlib.sha256(data).hexdigest()},
        "split_assignment": None,
        "relations": {},
    }


def load_jsonl(path) -> "list[dict]":
    """Load canonical records from a jsonl file (invalid lines raise)."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            ok, errors = validate(rec)
            if not ok:
                raise ValueError(f"{path}:{i + 1}: {errors}")
            out.append(rec)
    return out
