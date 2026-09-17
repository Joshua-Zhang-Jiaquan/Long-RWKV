"""Task 9 prep: token-targeted fixtures must be exact in TOKENS.

The registered matrix declares its lengths in characters, and the measured
consequence is that the declared 65,536 cell is 9,027-16,939 RWKV tokens.  A
long-context claim needs the length axis to mean tokens, so these tests pin the
three properties that make the new fixtures usable and the three that keep them
from disturbing what already exists:

* exact token length at every declared size, and the evidence at the declared
  position fraction;
* the prompt ENDS at the query line, so there is an answer slot (the character
  fixture puts its length filler after the query, which is the defect this
  addresses);
* evidence and gold come from the registered generator unchanged;
* the character fixtures and their pinned length set are untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.tasks import exact_tasks
from scale.experiments.nonlatent_iclr.tasks import models
from scale.experiments.nonlatent_iclr.tasks import token_targeted_tasks as tt
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import (
    VOCAB_SHA256, load_safe_tokenizer)

REPO = Path(__file__).resolve().parents[3]
VOCAB = REPO / "base_models" / "RWKV7-Goose-World3-2.9B-HF" / "rwkv_vocab_v20230424.txt"


@pytest.fixture(scope="module")
def tokenizer():
    if not VOCAB.is_file():
        pytest.skip(f"pinned vocabulary absent: {VOCAB}")
    return load_safe_tokenizer(VOCAB, VOCAB_SHA256)


class _FakeEncode:
    """A byte-level stand-in, for tests that must run without the vocabulary."""

    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(text.encode("utf-8"))


def _request(**overrides) -> tt.TokenTargetedRequest:
    base = dict(family="associative_recall", data_seed=101, token_length=4096,
                position_fraction=50, load=8, distractor="none", instance_index=0)
    base.update(overrides)
    return tt.TokenTargetedRequest(**base)


# --------------------------------------------------------------------------
# the length axis is tokens
# --------------------------------------------------------------------------


@pytest.mark.parametrize("length", tt.TOKEN_LENGTHS)
def test_every_declared_length_is_exact_in_tokens(tokenizer, length: int) -> None:
    # Given: each registered token length.
    # When: the fixture is built.
    task = tt.generate_token_targeted_task(tokenizer, _request(token_length=length))
    # Then: the prompt is that many RWKV TOKENS, which is the whole point -- the
    # character fixture's 65536 cell is really 9-17K tokens.
    assert task.n_prompt_tokens == length
    assert task.request.token_length == length


@pytest.mark.parametrize("fraction", sorted(models.POSITIONS))
def test_the_evidence_sits_at_the_declared_position_fraction(tokenizer, fraction: int) -> None:
    # Given: each declared evidence position.
    task = tt.generate_token_targeted_task(
        tokenizer, _request(position_fraction=fraction, token_length=8192))
    # When/Then: the evidence begins at that fraction of the total, matching the
    # character fixture's convention so the two are comparable on this axis.
    assert task.evidence_start_token == 8192 * fraction // 100
    assert task.n_prompt_tokens == 8192


def test_the_prompt_ends_at_the_query_line(tokenizer) -> None:
    """The character fixture appends its length filler AFTER the query."""
    # Given: a fixture.
    task = tt.generate_token_targeted_task(tokenizer, _request())
    # When: the prompt's tail is decoded.
    tail = tokenizer.decode_ids(tuple(task.prompt_ids[-12:]))
    # Then: it ends with the query line's own encoding, so a generative or
    # mask-canvas model answers where the task intends rather than after a run
    # of filler.
    query_ids = tokenizer.encode_text(tt.QUERY_LINE)
    assert task.prompt_ids[-len(query_ids):] == query_ids
    assert b"QUERY" in tail


def test_the_character_fixture_really_does_bury_its_answer_slot() -> None:
    """The defect, pinned so the fix cannot be silently reverted."""
    # Given: a character fixture.
    task = exact_tasks.generate_exact_task(models.ExactTaskRequest(
        family="associative_recall", data_seed=101, declared_length=4096,
        position_fraction=50, load=8, distractor="none", instance_index=0))
    # When/Then: its prompt ends in filler, not in the query -- which is why a
    # token-targeted variant is needed at all rather than a re-parameterisation.
    assert task.condition.public_prompt.endswith("y")
    assert not task.condition.public_prompt.endswith("ANSWER THE QUERY")


# --------------------------------------------------------------------------
# evidence and gold are the registered generator's
# --------------------------------------------------------------------------


def test_the_gold_is_the_registered_generators_gold(tokenizer) -> None:
    # Given: the same coordinates on both fixtures.
    request = _request()
    task = tt.generate_token_targeted_task(tokenizer, request)
    source = exact_tasks.generate_exact_task(request.as_character_request())
    # When/Then: gold and evidence agree, so the two fixtures test one task and
    # cannot drift apart.
    assert task.answer == source.correct_answer
    assert task.answer_ids == tokenizer.encode_text(source.correct_answer)
    # and the evidence survives into the token prompt verbatim
    decoded = tokenizer.decode_ids(tuple(task.prompt_ids))
    assert tt._evidence_of(source.condition.public_prompt).encode("utf-8") in decoded


def test_as_character_request_preserves_the_coordinates() -> None:
    # Given: a token request.
    request = _request(family="finite_hmm", data_seed=103, position_fraction=90,
                       load=128, distractor="similar", instance_index=199)
    # When: it is mapped onto the character fixture's coordinates.
    mapped = request.as_character_request()
    # Then: everything but the length unit is carried across, and the length is
    # the smallest registered value because it only shapes the framing that this
    # module discards.
    assert (mapped.family, mapped.data_seed) == ("finite_hmm", 103)
    assert (mapped.position_fraction, mapped.load) == (90, 128)
    assert (mapped.distractor, mapped.instance_index) == ("similar", 199)
    assert mapped.declared_length == min(models.LENGTHS)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_an_undeclared_token_length_is_refused() -> None:
    # Given: a length that is not in the registered set -- including the
    # character lengths' neighbours, which are not token lengths.
    # When/Then: it is refused, naming the axis it is measured in.
    with pytest.raises(tt.TokenTargetedError, match="undeclared token length"):
        _request(token_length=2048)
    with pytest.raises(tt.TokenTargetedError, match="RWKV tokens"):
        _request(token_length=100_000)


@pytest.mark.parametrize("field,value", [
    ("family", "needle_in_haystack"),
    ("data_seed", 999),
    ("position_fraction", 33),
    ("load", 7),
    ("distractor", "adversarial"),
    ("instance_index", 200),
])
def test_undeclared_coordinates_are_refused(field: str, value: object) -> None:
    # Given: a coordinate outside the registered matrix.
    # When/Then: it is refused rather than silently generating an unregistered
    # cell -- the registry is the authority on what is in scope.
    with pytest.raises(tt.TokenTargetedError):
        _request(**{field: value})


def test_character_counted_filler_is_refused_and_the_measurement_recorded() -> None:
    """The tempting wrong way to hit a token count."""
    # Given: the retired character-filler path.
    # When/Then: it refuses with a message naming the measurement, because the
    # assumption it encodes (repetition is 1:1 through the byte-trie) is the
    # natural one to make and is wrong.
    with pytest.raises(tt.TokenTargetedError, match="pad with token ids"):
        tt._filler_ids(_FakeEncode(), "x", 10)


def test_the_pad_text_is_a_single_token(tokenizer) -> None:
    # Given: the pad text.
    # When/Then: it encodes to exactly one id, which is what makes id-level
    # padding exact.
    assert len(tokenizer.encode_text(tt.PAD_TEXT)) == 1
    assert tt.pad_token_id(tokenizer) == tokenizer.encode_text(tt.PAD_TEXT)[0]


def test_an_evidence_block_too_large_for_its_position_is_refused() -> None:
    # Given: a fake tokenizer and an evidence long enough that the opening
    # framing cannot fit before the declared evidence start.
    encode = _FakeEncode()
    with pytest.raises(tt.TokenTargetedError, match="does not fit"):
        tt.build_token_prompt(encode, "evidence", target_tokens=8,
                              position_fraction=10)


def test_a_non_positive_target_is_refused() -> None:
    with pytest.raises(tt.TokenTargetedError, match="must be positive"):
        tt.build_token_prompt(_FakeEncode(), "e", target_tokens=0,
                              position_fraction=50)


# --------------------------------------------------------------------------
# nothing existing moves
# --------------------------------------------------------------------------


def test_the_character_length_set_is_untouched() -> None:
    # Given: the module that declares character lengths.
    # When/Then: it still declares characters, and the token module declares a
    # different kind -- so a reader cannot confuse one artifact for the other.
    assert models.LENGTHS == frozenset({4096, 8192, 16384, 32768, 65536})
    assert tt.LENGTH_KIND != "logical_character_fixture"
    assert tt.TOKEN_LENGTHS == tuple(sorted(models.LENGTHS))


def test_the_evidence_parser_reads_the_registered_prompt() -> None:
    # Given: a registered prompt.
    source = exact_tasks.generate_exact_task(models.ExactTaskRequest(
        family="code_dataflow", data_seed=101, declared_length=4096,
        position_fraction=50, load=1, distractor="none", instance_index=0))
    # When: the evidence block is extracted.
    evidence = tt._evidence_of(source.condition.public_prompt)
    # Then: it is the same evidence the registered evaluator re-derives the gold
    # from, so the token fixture inherits the task rather than inventing one.
    assert exact_tasks.public_evidence(source) == evidence


def test_the_evidence_parser_refuses_a_malformed_prompt() -> None:
    with pytest.raises(tt.TokenTargetedError, match="no single evidence block"):
        tt._evidence_of("no framing here at all")


def test_the_delayed_query_family_gold_is_load_invariant() -> None:
    """A recorded observation, deliberately NOT "fixed" here.

    Across 40 instances the ``overwrite_delayed_query`` gold is identical at load
    32 and load 128 while the prompt differs, whereas the other three families'
    gold tracks the load.  For a *delayed-query* task that may be correct by
    design -- the answer is the last write, whatever intervenes -- so this test
    records the behaviour and its reading rather than asserting the generator is
    wrong.  A reviewer with the design intent should settle it.
    """
    def gold(load: int, index: int) -> str:
        return exact_tasks.generate_exact_task(models.ExactTaskRequest(
            family="overwrite_delayed_query", data_seed=101, declared_length=4096,
            position_fraction=50, load=load, distractor="none",
            instance_index=index)).correct_answer

    invariant = [gold(32, i) == gold(128, i) for i in range(8)]
    assert all(invariant), "the recorded observation no longer holds; re-derive it"
    # and the other families do track the load, so this is family-specific
    def other(load: int, index: int) -> str:
        return exact_tasks.generate_exact_task(models.ExactTaskRequest(
            family="associative_recall", data_seed=101, declared_length=4096,
            position_fraction=50, load=load, distractor="none",
            instance_index=index)).correct_answer

    assert any(other(32, i) != other(128, i) for i in range(8))
