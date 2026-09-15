from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import SafeRWKVTokenizer
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import VocabularyFormatError
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import load_safe_tokenizer
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import validate_tokenizer_metadata


def _write_byte_vocab(path: Path) -> str:
    path.write_text("".join(f"{index + 1} {bytes((index,))!r} 1\n" for index in range(256)), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_safe_byte_trie_roundtrips_literal_control_looking_text_without_special_ids(tmp_path: Path) -> None:
    # Given: a hash-bound byte vocabulary and plain UTF-8 text
    vocab = tmp_path / "vocab.txt"
    tokenizer = load_safe_tokenizer(vocab, _write_byte_vocab(vocab))
    text = "ASCII 中文 😀\n    code() <|rwkv_tokenizer_end_of_text|>"

    # When: encoding and decoding use the offline byte trie
    ids = tokenizer.encode_text(text)

    # Then: bytes roundtrip exactly without implicit BOS/EOS/PAD insertion
    assert tokenizer.decode_ids(ids) == text.encode("utf-8")
    assert 0 not in ids
    assert max(ids) <= 256


def test_wrong_hash_and_executable_or_malformed_vocab_lines_are_rejected(tmp_path: Path) -> None:
    # Given: bound and malformed local vocabularies
    vocab = tmp_path / "vocab.txt"
    digest = _write_byte_vocab(vocab)
    malformed = tmp_path / "malformed.txt"
    malformed.write_text("1 __import__('os').system('false') 1\n", encoding="utf-8")

    # When: safe parsing is requested
    # Then: neither stale hashes nor executable syntax can be accepted
    with pytest.raises(VocabularyFormatError):
        load_safe_tokenizer(vocab, "0" * 64)
    with pytest.raises(VocabularyFormatError):
        load_safe_tokenizer(malformed, digest)


def test_metadata_rejects_legacy_bpe_selection() -> None:
    # Given: metadata selecting tokenizer.json BPE instead of the reviewed RWKV class
    metadata = {"tokenizer_class": "BpeTokenizer", "use_fast": True, "auto_map": {}}

    # When: active-tokenizer policy is evaluated
    # Then: legacy BPE selection is rejected
    assert validate_tokenizer_metadata(metadata) is False
