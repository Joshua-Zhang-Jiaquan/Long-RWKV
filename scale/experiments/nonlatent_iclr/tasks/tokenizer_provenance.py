from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .tokenizer_qualification import _results_document
from .tokenizer_qualification import qualification_sources_current
from .tokenizer_qualification import qualify_local

_MANIFEST_KEYS = frozenset({"version", "license", "sha256", "results_path", "qualification_procedure", "encoding_qualified", "matrix_token_lengths_qualified", "global_task4_gate"})
_RECEIPT_KEYS = frozenset({"results_sha256", "manifest_sha256", "source_hashes"})


def verify_qualification(model_root: Path, output_root: Path, receipt_path: Path) -> bool:
    results_path = output_root / "rwkv_tokenizer_results.json"
    manifest_path = output_root / "rwkv_tokenizer_manifest.json"
    try:
        results_bytes = results_path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
        receipt = _load_object(receipt_path)
        manifest = _load_object(manifest_path)
        results = json.loads(results_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(results, dict) or set(manifest) != _MANIFEST_KEYS or set(receipt) != _RECEIPT_KEYS:
        return False
    results_hash = _sha256(results_bytes)
    if manifest["results_path"] != results_path.name or manifest["sha256"] != results_hash:
        return False
    if receipt["results_sha256"] != results_hash or receipt["manifest_sha256"] != _sha256(manifest_bytes):
        return False
    if not isinstance(results.get("source_hashes"), dict):
        return False
    if receipt["source_hashes"] != results["source_hashes"]:
        return False
    if not qualification_sources_current(results["source_hashes"], model_root):
        return False
    expected = qualify_local(model_root)
    return results == _results_document(expected)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
