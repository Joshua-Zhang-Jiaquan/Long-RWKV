"""Safe offline qualification for the active RWKV byte-trie tokenizer."""

from __future__ import annotations

import ast
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .exact_tasks import ExactTaskRequest
from .exact_tasks import generate_exact_task

VOCAB_SHA256 = "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
VOCAB_NAME = "rwkv_vocab_v20230424.txt"
MODEL_FILES = ("hf_rwkv_tokenizer.py", VOCAB_NAME, "config.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "tokenizer.json")
TASKS_ROOT = Path(__file__).resolve().parent


class VocabularyFormatError(ValueError):
    """Raised when a hash-bound vocabulary cannot be parsed as byte literals."""


class SafeRWKVTokenizer:  # noqa: MUTABLE_OK
    """Mutable trie builder; encoding itself is read-only after initialization."""

    def __init__(self, tokens: dict[int, bytes]) -> None:
        self.tokens = tokens
        self.children: list[dict[int, int]] = [{}]
        self.terminals: list[int | None] = [None]
        for token_id, token in tokens.items():
            self._add(token, token_id)

    def _add(self, token: bytes, token_id: int) -> None:
        node = 0
        for byte in token:
            child = self.children[node].get(byte)
            if child is None:
                child = len(self.children)
                self.children[node][byte] = child
                self.children.append({})
                self.terminals.append(None)
            node = child
        if self.terminals[node] is not None:
            raise VocabularyFormatError("duplicate byte token")
        self.terminals[node] = token_id

    def encode_text(self, text: str) -> tuple[int, ...]:
        """Encode plain UTF-8 bytes with no BOS/EOS/PAD or special-token insertion."""
        if not isinstance(text, str):
            raise VocabularyFormatError("plain text must be str")
        source = text.encode("utf-8")
        output: list[int] = []
        offset = 0
        while offset < len(source):
            node = 0
            cursor = offset
            best: tuple[int, int] | None = None
            while cursor < len(source):
                child = self.children[node].get(source[cursor])
                if child is None:
                    break
                node = child
                cursor += 1
                token_id = self.terminals[node]
                if token_id is not None:
                    best = (cursor, token_id)
            if best is None:
                raise VocabularyFormatError(f"no byte token at offset {offset}")
            offset, token_id = best
            output.append(token_id)
        return tuple(output)

    def decode_ids(self, token_ids: tuple[int, ...]) -> bytes:
        """Decode only declared vocabulary ids to their original bytes."""
        try:
            return b"".join(self.tokens[token_id] for token_id in token_ids)
        except KeyError as error:
            raise VocabularyFormatError(f"unknown token id: {error.args[0]}") from error


@dataclass(frozen=True, slots=True)
class PromptProjection:
    family: str
    character_length: int
    token_length: int
    evidence_character_offset: int
    evidence_token_offset: int
    max_id: int
    no_reserved_ids: bool
    decoded_bytes_equal: bool


@dataclass(frozen=True, slots=True)
class QualificationResult:
    encoding_qualified: bool
    matrix_token_lengths_qualified: bool
    projections: tuple[PromptProjection, ...]
    source_hashes: dict[str, str]


def load_safe_tokenizer(vocab_path: Path, expected_sha256: str) -> SafeRWKVTokenizer:
    """Verify vocabulary bytes before parsing literal-only lines into a byte trie."""
    actual = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise VocabularyFormatError("vocabulary sha256 mismatch")
    tokens: dict[int, bytes] = {}
    for expected_id, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines(), start=1):
        token_id, token = _parse_vocab_line(line, expected_id)
        tokens[token_id] = token
    if tuple(tokens) != tuple(range(1, len(tokens) + 1)):
        raise VocabularyFormatError("vocabulary ids must be contiguous from 1")
    return SafeRWKVTokenizer(tokens)


def validate_tokenizer_metadata(metadata: dict[str, object]) -> bool:
    """Accept only the reviewed slow RwkvTokenizer metadata, never tokenizer.json BPE."""
    auto_map = metadata.get("auto_map")
    return metadata.get("tokenizer_class") == "RwkvTokenizer" and metadata.get("use_fast") is False and isinstance(auto_map, dict) and auto_map.get("AutoTokenizer") == ["hf_rwkv_tokenizer.RwkvTokenizer", None]


def qualify_local(model_root: Path) -> QualificationResult:
    """Qualify actual local bytes and sample four feasible public task prompts on CPU."""
    source_hashes = _source_hashes(model_root)
    metadata = json.loads((model_root / "tokenizer_config.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not validate_tokenizer_metadata(metadata):
        raise VocabularyFormatError("active tokenizer metadata does not select reviewed RWKV trie")
    tokenizer = load_safe_tokenizer(model_root / VOCAB_NAME, VOCAB_SHA256)
    if len(tokenizer.tokens) != 65529:
        raise VocabularyFormatError("active vocabulary must expose ids 1..65529")
    projections = tuple(_project(tokenizer, family) for family in ("associative_recall", "overwrite_delayed_query", "finite_hmm", "code_dataflow"))
    return QualificationResult(all(item.decoded_bytes_equal and item.no_reserved_ids for item in projections), False, projections, source_hashes)


def qualification_projection(result: QualificationResult) -> tuple[PromptProjection, ...]:
    """Expose only public-prompt token measurements, never private exact-task gold."""
    return result.projections


def qualification_sources_current(source_hashes: Mapping[str, str], model_root: Path, exact_tasks_source: Path | None = None) -> bool:
    """Reject stale vocabulary, metadata, contract, or qualification procedure hashes."""
    return dict(source_hashes) == _source_hashes(model_root, exact_tasks_source)


def write_qualification(model_root: Path, output_root: Path, evidence_root: Path) -> QualificationResult:
    """Write source-bound tokenizer results while leaving global Task4 readiness unchanged."""
    result = qualify_local(model_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "rwkv_tokenizer_results.json"
    manifest_path = output_root / "rwkv_tokenizer_manifest.json"
    _archive_if_present(results_path)
    _archive_if_present(manifest_path)
    results = _results_document(result)
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"version": "safe-rwkv-byte-trie-v1", "license": "local artifact metadata; benchmark rights unresolved", "sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(), "results_path": results_path.name, "qualification_procedure": "ast.literal_eval hash-bound vocabulary + offline byte trie", "encoding_qualified": result.encoding_qualified, "matrix_token_lengths_qualified": False, "global_task4_gate": "BLOCKED: tokenizer artifact available; external qualification adapter integration pending"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    attempt = evidence_root / "tokenizer-qualification" / uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "receipt.json").write_text(json.dumps({"results_sha256": manifest["sha256"], "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "source_hashes": result.source_hashes}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parse_vocab_line(line: str, expected_id: int) -> tuple[int, bytes]:
    try:
        id_text, remainder = line.split(" ", maxsplit=1)
        literal, length_text = remainder.rsplit(" ", maxsplit=1)
        token_id = int(id_text)
        value = ast.literal_eval(literal)
        length = int(length_text)
    except (SyntaxError, ValueError):
        raise VocabularyFormatError("malformed vocabulary line") from None
    if token_id != expected_id or not isinstance(value, (str, bytes)):
        raise VocabularyFormatError("invalid vocabulary id or literal")
    token = value.encode("utf-8") if isinstance(value, str) else value
    if not token or len(token) != length:
        raise VocabularyFormatError("vocabulary byte length mismatch")
    return token_id, token


def _project(tokenizer: SafeRWKVTokenizer, family: str) -> PromptProjection:
    task = generate_exact_task(ExactTaskRequest(family, 101, 8192, 50, 8, "none", 0))
    prompt = task.condition.public_prompt
    ids = tokenizer.encode_text(prompt)
    evidence = "BEGIN EVIDENCE\n"
    character_offset = prompt.index(evidence) + len(evidence)
    token_offset = len(tokenizer.encode_text(prompt[:character_offset]))
    return PromptProjection(family, len(prompt), len(ids), character_offset, token_offset, max(ids), all(1 <= token_id <= 65529 for token_id in ids), tokenizer.decode_ids(ids) == prompt.encode("utf-8"))


def _results_document(result: QualificationResult) -> dict[str, object]:
    return {"encoding_policy": "plain UTF-8 bytes; no automatic BOS/EOS/special insertion", "active_encoder": "hash-bound rwkv_vocab_v20230424.txt byte trie", "inactive_tokenizer_json_policy": "tokenizer.json BPE bytes are hashed for provenance but never loaded", "encoding_qualified": result.encoding_qualified, "matrix_token_lengths_qualified": result.matrix_token_lengths_qualified, "generation_protocol_status": "UNRESOLVED: config BOS/EOS=1/2 conflicts with generation EOS/PAD=0 and tokenizer EOT/PAD=0", "reserved_id_policy": "actual vocabulary ids 1..65529; id 0 is metadata EOT/PAD only; legacy 65530/65531/65535 are not asserted vocabulary ids", "source_hashes": result.source_hashes, "projections": [{"family": item.family, "character_length": item.character_length, "token_length": item.token_length, "evidence_character_offset": item.evidence_character_offset, "evidence_token_offset": item.evidence_token_offset, "max_id": item.max_id, "no_reserved_ids": item.no_reserved_ids, "decoded_bytes_equal": item.decoded_bytes_equal} for item in result.projections]}


def _source_hashes(model_root: Path, exact_tasks_source: Path | None = None) -> dict[str, str]:
    exact_source = exact_tasks_source if exact_tasks_source is not None else TASKS_ROOT / "exact_tasks.py"
    files = tuple((name, model_root / name) for name in MODEL_FILES) + (("token_contract.py", Path(__file__).resolve().parents[3] / "data" / "token_contract.py"), ("exact_tasks.py", exact_source), ("tokenizer_qualification.py", Path(__file__)), ("tokenizer_provenance.py", TASKS_ROOT / "tokenizer_provenance.py"))
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files}


def _archive_if_present(path: Path) -> None:
    if not path.is_file():
        return
    content = path.read_bytes()
    archive = path.with_name(f"{path.stem}.archive-{hashlib.sha256(content).hexdigest()[:16]}{path.suffix}")
    if not archive.exists():
        archive.write_bytes(content)
