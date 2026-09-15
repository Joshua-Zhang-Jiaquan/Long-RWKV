from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
from typing import Final

import pytest

from scale.experiments.nonlatent_iclr.tasks import length_qualification as q
from scale.experiments.nonlatent_iclr.tasks.exact_tasks import generate_exact_task
from scale.experiments.nonlatent_iclr.tasks.models import ExactTaskRequest
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import (
    VOCAB_NAME, VOCAB_SHA256, SafeRWKVTokenizer, VocabularyFormatError,
    load_safe_tokenizer,
)

MODEL: Final = Path(os.environ.get(
    "NONLATENT_MODEL_DIR",
    Path(__file__).resolve().parents[3] / "external/models/RWKV7-Goose-World3-2.9B-HF"))
ASSETS: Final = Path("DAN/nonlatent_iclr/task4_assets/tokenizer")
RECEIPT: Final = Path(".omo/evidence/nonlatent-rnn-dlm-iclr-research/task-04/tokenizer-qualification/8ef417180818408d8bc65efcb4dcd75c/receipt.json")
REQUEST: Final = ExactTaskRequest("associative_recall", 101, 8192, 50, 8, "none", 0)


@pytest.fixture(scope="module")
def tokenizer() -> SafeRWKVTokenizer:
    return load_safe_tokenizer(MODEL / VOCAB_NAME, VOCAB_SHA256)


def test_encoding_when_using_generated_fixture(tokenizer: SafeRWKVTokenizer) -> None:
    # Given: the unchanged seed-101 public fixture and the accepted encoder.
    task = generate_exact_task(REQUEST)
    # When: its actual bytes are measured.
    measured = q.measure_cell(REQUEST, task.condition, tokenizer)
    # Then: the known projection and true full-stream boundary agree.
    assert (measured.character_length, measured.rwkv_token_length) == (8192, 1589)
    assert measured.decoded_bytes_equal and measured.reserved_ids_clear
    assert measured.max_token_id == 54436
    assert (measured.evidence_character_offset, measured.evidence_byte_offset,
            measured.evidence_token_offset) == (4096, 4096, 520)
    assert measured.prompt_sha256 != measured.token_ids_sha256


@pytest.mark.parametrize("reserved_id", [0, 65530, 65531, 65535])
def test_rejects_when_text_encodes_reserved_id(reserved_id: int) -> None:
    # Given: a tampered trie maps the entire real prompt to a reserved id.
    task = generate_exact_task(REQUEST)
    tampered = SafeRWKVTokenizer({reserved_id: task.condition.public_prompt.encode()})
    # When / Then: even byte-exact decoding cannot qualify reserved ids.
    with pytest.raises(q.LengthQualificationError, match="reserved"):
        _ = q.measure_cell(REQUEST, task.condition, tampered)


def test_rejects_when_decoded_bytes_mismatch(tokenizer: SafeRWKVTokenizer) -> None:
    # Given: a real trie whose decode table alone has been corrupted.
    tampered = SafeRWKVTokenizer(dict(tokenizer.tokens))
    task = generate_exact_task(REQUEST)
    first = tampered.encode_text(task.condition.public_prompt)[0]
    tampered.tokens[first] = b"corrupted"
    # When / Then: valid ids do not substitute for byte equality.
    with pytest.raises(q.LengthQualificationError, match="decode"):
        _ = q.measure_cell(REQUEST, task.condition, tampered)


def test_rejects_when_vocab_hash_is_wrong(tmp_path: Path) -> None:
    # Given: a local vocabulary with the wrong bytes.
    _ = (tmp_path / VOCAB_NAME).write_text("1 b'x' 1\n")
    # When / Then: the matrix loader cannot accept another hash-bound vocabulary.
    with pytest.raises(VocabularyFormatError, match="sha256"):
        _ = q.accepted_tokenizer(q.QualificationInputs(tmp_path, ASSETS, RECEIPT))


def test_rejects_when_asserted_length_is_character_length(tokenizer: SafeRWKVTokenizer) -> None:
    # Given: an actual token measurement of a character-sized fixture.
    measured = q.measure_cell(REQUEST, generate_exact_task(REQUEST).condition, tokenizer)
    # When / Then: a nominal 8192-character cell cannot assert 8192 RWKV tokens.
    with pytest.raises(q.LengthQualificationError, match="token length"):
        q.require_token_length(measured, REQUEST.declared_length)


def test_rejects_when_measurement_is_not_qualified(tokenizer: SafeRWKVTokenizer) -> None:
    # Given: an explicitly unqualified measurement with an otherwise correct length.
    measured = q.measure_cell(REQUEST, generate_exact_task(REQUEST).condition, tokenizer)
    unqualified = replace(measured, rwkv_token_qualified=False)
    # When / Then: a count without qualification cannot satisfy a token assertion.
    with pytest.raises(q.LengthQualificationError, match="unqualified"):
        q.require_token_length(unqualified, measured.rwkv_token_length)


def test_rejects_when_fixture_bytes_are_changed(tokenizer: SafeRWKVTokenizer) -> None:
    # Given: a public condition truncated after generation.
    condition = generate_exact_task(REQUEST).condition
    truncated = replace(condition, public_prompt=condition.public_prompt[:-1])
    # When / Then: a changed logical fixture cannot be qualified.
    with pytest.raises(q.LengthQualificationError, match="fixture"):
        _ = q.measure_cell(REQUEST, truncated, tokenizer)


def test_matrix_when_all_representatives_are_generated() -> None:
    # Given: accepted local provenance, not external benchmark inputs.
    inputs = q.QualificationInputs(MODEL, ASSETS, RECEIPT)
    # When: the existing registry's representative fixtures are encoded.
    result = q.qualify_representatives(inputs)
    # Then: feasible cells are measured without promoting the full token matrix.
    assert len(result.qualified_cells) == 689
    assert len(result.infeasible_cells) == 31
    assert result.representative_token_lengths_qualified
    assert not result.matrix_token_lengths_qualified
    assert all(row.rwkv_token_qualified for row in result.qualified_cells)


def test_cli_when_publishing_and_rejecting_tampered_artifact(tmp_path: Path) -> None:
    # Given: a new output path and the real offline command surface.
    output = tmp_path / "lengths.json"
    # When: the CLI publishes measurements.
    process = subprocess.run(
        [sys.executable, "-m", q.__name__, str(MODEL), str(ASSETS), str(RECEIPT), str(output)],
        capture_output=True, text=True, check=False, timeout=90,
    )
    # Then: replay accepts exact evidence, but rejects a non-qualified assertion.
    assert process.returncode == 0, process.stderr
    inputs = q.QualificationInputs(MODEL, ASSETS, RECEIPT)
    assert q.verify_length_artifact(output, inputs)
    original = output.read_bytes()
    tampered = original.replace(b'"rwkv_token_qualified": true', b'"rwkv_token_qualified": false', 1)
    assert tampered != original
    _ = output.write_bytes(tampered)
    assert not q.verify_length_artifact(output, inputs)


def test_cli_when_vocabulary_is_wrong_leaves_no_artifact(tmp_path: Path) -> None:
    # Given: wrong local vocabulary bytes at the real CLI boundary.
    _ = (tmp_path / VOCAB_NAME).write_text("1 b'x' 1\n")
    output = tmp_path / "rejected.json"
    # When: qualification is attempted.
    process = subprocess.run(
        [sys.executable, "-m", q.__name__, str(tmp_path), str(ASSETS), str(RECEIPT), str(output)],
        capture_output=True, text=True, check=False, timeout=10,
    )
    # Then: failure cannot publish a qualification artifact.
    assert process.returncode != 0
    assert not output.exists()


# The artifacts these tests exercise are not part of the repository: they are
# large, rights-gated, and produced on a cluster. A clone must SKIP rather than
# fail, so the suite reports honestly what it can verify without them. Point
# NONLATENT_MODEL_DIR at a prepared root to run them.
pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason=f"external artifact absent: {MODEL} -- set the matching "
           f"NONLATENT_* environment variable to a prepared root")
