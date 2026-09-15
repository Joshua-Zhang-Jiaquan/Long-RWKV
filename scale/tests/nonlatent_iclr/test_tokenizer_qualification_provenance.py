from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import os

from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import qualification_sources_current
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import write_qualification
from scale.experiments.nonlatent_iclr.tasks.tokenizer_provenance import verify_qualification

MODEL_ROOT: Final = Path(os.environ.get(
    "NONLATENT_MODEL_DIR",
    Path(__file__).resolve().parents[3] / "external/models/RWKV7-Goose-World3-2.9B-HF"))
EXACT_TASKS = Path("scale/experiments/nonlatent_iclr/tasks/exact_tasks.py")


def _receipt(evidence_root: Path) -> Path:
    return next((evidence_root / "tokenizer-qualification").glob("*/receipt.json"))


def _rehash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replayed_closed_results_bind_manifest_receipt_and_generator(tmp_path: Path) -> None:
    # Given: a fresh local qualification artifact set
    output = tmp_path / "tokenizer"
    evidence = tmp_path / "evidence"
    write_qualification(MODEL_ROOT, output, evidence)

    # When: verifier replays the four known public prompts
    # Then: receipt, result bytes, source hashes, and closed projections are accepted together
    assert verify_qualification(MODEL_ROOT, output, _receipt(evidence)) is True


def test_fabricated_or_stale_provenance_is_rejected(tmp_path: Path) -> None:
    # Given: otherwise bound artifacts and an isolated changed generator source copy
    output = tmp_path / "tokenizer"
    evidence = tmp_path / "evidence"
    result = write_qualification(MODEL_ROOT, output, evidence)
    receipt_path = _receipt(evidence)
    results_path = output / "rwkv_tokenizer_results.json"
    manifest_path = output / "rwkv_tokenizer_manifest.json"
    copied_generator = tmp_path / "exact_tasks.py"
    copied_generator.write_bytes(EXACT_TASKS.read_bytes() + b"\n")

    # When: empty projections are rehashed to look self-consistent and the generator copy changes
    forged = json.loads(results_path.read_text(encoding="utf-8"))
    forged["projections"] = []
    results_path.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = _rehash(results_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["results_sha256"] = manifest["sha256"]
    receipt["manifest_sha256"] = _rehash(manifest_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    # Then: replay rejects fabricated projections and source freshness rejects changed generator bytes
    assert verify_qualification(MODEL_ROOT, output, receipt_path) is False
    assert qualification_sources_current(result.source_hashes, MODEL_ROOT, copied_generator) is False


# The artifacts these tests exercise are not part of the repository: they are
# large, rights-gated, and produced on a cluster. A clone must SKIP rather than
# fail, so the suite reports honestly what it can verify without them. Point
# NONLATENT_MODEL_DIR at a prepared root to run them.
pytestmark = pytest.mark.skipif(
    not MODEL_ROOT.exists(),
    reason=f"external artifact absent: {MODEL_ROOT} -- set the matching "
           f"NONLATENT_* environment variable to a prepared root")
