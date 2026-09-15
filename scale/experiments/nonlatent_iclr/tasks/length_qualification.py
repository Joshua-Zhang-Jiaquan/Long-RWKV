"""RWKV token measurements for unchanged representative exact-task fixtures."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

from ..task_registry import JsonValue
from ..task_registry import build_registry
from .exact_tasks import InfeasibleTaskError, generate_exact_task, validate_exact_gold
from .models import ExactTaskRequest, PublicCondition
from .tokenizer_provenance import verify_qualification
from .tokenizer_qualification import (
    VOCAB_NAME, VOCAB_SHA256, SafeRWKVTokenizer, load_safe_tokenizer,
)

GENERATION_PROTOCOL: Final = (
    "UNRESOLVED: config BOS/EOS=1/2 conflicts with generation EOS/PAD=0 "
    "and tokenizer EOT/PAD=0"
)
EXTERNAL_BLOCKERS: Final = (
    "ruler/development_manifest.json", "longbench/development_manifest.json",
    "repository_tasks/manifest.json", "repository_evaluator/isolation_manifest.json",
)
SUMMARY_NAME: Final = "length_qualification_summary.json"
REPRESENTATIVE_ARTIFACT: Final = "representative_rwkv_lengths.json"


@dataclass(frozen=True, slots=True)
class QualificationInputs:
    model_root: Path
    tokenizer_evidence_root: Path
    tokenizer_receipt: Path


class LengthQualificationError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason: str = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CellMeasurement:
    request: ExactTaskRequest
    family: str
    character_length: int
    rwkv_token_length: int
    decoded_bytes_equal: bool
    max_token_id: int
    reserved_ids_clear: bool
    evidence_character_offset: int
    evidence_byte_offset: int
    evidence_token_offset: int
    evidence_intra_token_byte_offset: int
    prompt_sha256: str
    token_ids_sha256: str
    rwkv_token_qualified: bool = True


@dataclass(frozen=True, slots=True)
class InfeasibleCell:
    request: ExactTaskRequest
    reason: str


@dataclass(frozen=True, slots=True)
class SourceBinding:
    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RepresentativeQualification:
    qualified_cells: tuple[CellMeasurement, ...]
    infeasible_cells: tuple[InfeasibleCell, ...]
    source_hashes: tuple[SourceBinding, ...]
    representative_token_lengths_qualified: bool = True
    matrix_token_lengths_qualified: bool = False
    status: str = "BLOCKED"
    scope: str = "representative_exact_task_public_prompts; seed=101; instance_index=0"
    declared_length_kind: str = "logical_character_fixture"
    encoding_policy: str = "plain UTF-8 bytes; no automatic BOS/EOS/special insertion"
    token_ids_hash_format: str = "concatenated unsigned 32-bit little-endian ids"
    evidence_offset_policy: str = "zero-based full-stream containing token plus intra-token byte offset"
    generation_protocol_status: str = GENERATION_PROTOCOL
    external_suites_status: str = "BLOCKED; not read or qualified"
    external_blockers: tuple[str, ...] = EXTERNAL_BLOCKERS
    vocab_sha256: str = VOCAB_SHA256


def accepted_tokenizer(inputs: QualificationInputs) -> SafeRWKVTokenizer:
    """Require the fixed vocabulary and replay accepted provenance without rewriting it."""
    tokenizer = load_safe_tokenizer(inputs.model_root / VOCAB_NAME, VOCAB_SHA256)
    if not verify_qualification(inputs.model_root, inputs.tokenizer_evidence_root, inputs.tokenizer_receipt):
        raise LengthQualificationError("accepted tokenizer provenance mismatch")
    return tokenizer


def measure_cell(request: ExactTaskRequest, condition: PublicCondition, tokenizer: SafeRWKVTokenizer) -> CellMeasurement:
    """Measure a generated public fixture; the production caller owns tokenizer admission."""
    if condition != generate_exact_task(request).condition:
        raise LengthQualificationError("fixture differs from deterministic generator")
    prompt = condition.public_prompt
    source = prompt.encode("utf-8")
    ids = tokenizer.encode_text(prompt)
    if not ids or not all(1 <= token_id <= 65529 for token_id in ids):
        raise LengthQualificationError("reserved or absent token ids")
    if tokenizer.decode_ids(ids) != source:
        raise LengthQualificationError("decode bytes mismatch")
    opening = "BEGIN EVIDENCE\n"
    character_offset = prompt.index(opening) + len(opening)
    byte_offset = len(prompt[:character_offset].encode("utf-8"))
    consumed = 0
    token_offset = 0
    for token_offset, token_id in enumerate(ids):
        width = len(tokenizer.tokens[token_id])
        if consumed + width > byte_offset:
            break
        consumed += width
    return CellMeasurement(
        request=request, family=request.family, character_length=len(prompt),
        rwkv_token_length=len(ids), decoded_bytes_equal=True, max_token_id=max(ids),
        reserved_ids_clear=True, evidence_character_offset=character_offset,
        evidence_byte_offset=byte_offset, evidence_token_offset=token_offset,
        evidence_intra_token_byte_offset=byte_offset - consumed,
        prompt_sha256=sha256(source).hexdigest(),
        token_ids_sha256=sha256(b"".join(token_id.to_bytes(4, "little") for token_id in ids)).hexdigest(),
    )


def require_token_length(measured: CellMeasurement, expected_tokens: int) -> None:
    """Reject logical-character or unqualified counts at a token-budget assertion boundary."""
    if not (measured.rwkv_token_qualified and measured.decoded_bytes_equal and measured.reserved_ids_clear):
        raise LengthQualificationError("unqualified token measurement")
    if measured.rwkv_token_length != expected_tokens:
        raise LengthQualificationError("RWKV token length mismatch")


def qualify_representatives(inputs: QualificationInputs) -> RepresentativeQualification:
    """Generate one representative per existing cell; never read an external suite."""
    tokenizer = accepted_tokenizer(inputs)
    qualified: list[CellMeasurement] = []
    infeasible: list[InfeasibleCell] = []
    for cell in build_registry().cells:
        request = ExactTaskRequest(cell.family, 101, cell.length, cell.position_fraction, cell.load, cell.distractor, 0)
        try:
            task = generate_exact_task(request)
        except InfeasibleTaskError as error:
            infeasible.append(InfeasibleCell(request, str(error)))
            continue
        if not validate_exact_gold(task):
            raise LengthQualificationError("generated exact gold mismatch")
        qualified.append(measure_cell(request, task.condition, tokenizer))
    root = Path(__file__).resolve().parent
    sources = (
        root / "length_qualification.py", root / "exact_tasks.py", root / "models.py",
        root / "provenance.py", root / "external_assets.py", root.parent / "task_registry.py",
        root / "tokenizer_qualification.py", root / "tokenizer_provenance.py",
        inputs.model_root / VOCAB_NAME,
        inputs.tokenizer_evidence_root / "rwkv_tokenizer_results.json",
        inputs.tokenizer_evidence_root / "rwkv_tokenizer_manifest.json", inputs.tokenizer_receipt,
    )
    bindings = tuple(SourceBinding(path.name, sha256(path.read_bytes()).hexdigest()) for path in sources)
    return RepresentativeQualification(tuple(qualified), tuple(infeasible), bindings)


def qualification_bytes(result: RepresentativeQualification) -> bytes:
    """Canonical public measurements only: no prompts, private answers, or hidden tests."""
    return (json.dumps(asdict(result), indent=2, sort_keys=True) + "\n").encode("utf-8")


def qualification_summary(result: RepresentativeQualification, artifact_path: Path, inputs: QualificationInputs) -> dict[str, JsonValue]:
    """The public block emitted into the registry: counts, policy and replay coordinates."""
    return {
        "artifact": artifact_path.name,
        "artifact_sha256": sha256(artifact_path.read_bytes()).hexdigest(),
        "replay_model_root": str(inputs.model_root),
        "replay_evidence_root": str(inputs.tokenizer_evidence_root),
        "replay_receipt_path": str(inputs.tokenizer_receipt),
        "qualified_representative_cells": len(result.qualified_cells),
        "infeasible_cells": len(result.infeasible_cells),
        "representative_token_lengths_qualified": result.representative_token_lengths_qualified,
        "matrix_token_lengths_qualified": result.matrix_token_lengths_qualified,
        "scope": result.scope,
        "declared_length_kind": result.declared_length_kind,
        "encoding_policy": result.encoding_policy,
        "generation_protocol_status": result.generation_protocol_status,
        "external_suites_status": result.external_suites_status,
        "source_hashes": {binding.name: binding.sha256 for binding in result.source_hashes},
    }


def write_summary(inputs: QualificationInputs, artifact_path: Path, output_path: Path) -> dict[str, JsonValue]:
    """Emit the registry summary only for an artifact that replays from the accepted sources."""
    result = qualify_representatives(inputs)
    if artifact_path.read_bytes() != qualification_bytes(result):
        raise LengthQualificationError("representative artifact did not replay")
    summary = qualification_summary(result, artifact_path, inputs)
    with output_path.open("xb") as stream:
        _ = stream.write((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return summary


def verify_length_artifact(path: Path, inputs: QualificationInputs) -> bool:
    """Fail closed by replaying all cells; untrusted JSON never enters the domain."""
    return path.read_bytes() == qualification_bytes(qualify_representatives(inputs))


def main() -> None:
    """Offline CLI: python -m ...tasks.length_qualification MODEL EVIDENCE RECEIPT OUTPUT."""
    model, evidence, receipt, output = (Path(value) for value in sys.argv[1:])
    result = qualify_representatives(QualificationInputs(model, evidence, receipt))
    with output.open("xb") as stream:
        _ = stream.write(qualification_bytes(result))
    print(f"representative={len(result.qualified_cells)} infeasible={len(result.infeasible_cells)} Task4=BLOCKED")


if __name__ == "__main__":
    main()
