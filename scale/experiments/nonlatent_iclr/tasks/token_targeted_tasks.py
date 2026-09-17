"""Token-targeted exact-task fixtures for the long-context matrix.

The registered long-context matrix declares its lengths in **characters**
(``exact_tasks.py`` stamps ``logical_character_fixture``), and the measured
consequence is that the declared 65,536 cell is really 9,027-16,939 RWKV
*tokens*.  A long-context claim needs the length axis to mean tokens, so this
module builds a second, coexisting fixture family whose lengths are exact RWKV
token counts.

Two things it does NOT do, both deliberate:

* It does not touch the character fixtures or their hash-bound matrix.  The
  registry, ``length_matrix.json`` and ``representative_rwkv_lengths.json`` stay
  byte-identical; the new artifacts are separate files with their own scope
  string.  ``exact_tasks.py`` is hash-bound into the tokenizer qualification's
  source hashes, so editing it would invalidate provenance rather than improve
  anything.
* It does not rewrite the evidence generators.  Evidence and gold come from
  ``generate_exact_task`` unchanged, so the token fixtures and the character
  fixtures test the same four families with the same gold; only the framing and
  the length axis differ.

The prompt layout is the substantive change.  The character fixture emits
``CONTEXT <x>\\nBEGIN EVIDENCE\\n<evidence>\\nEND EVIDENCE\\nANSWER THE QUERY\\n<y>``
-- the length filler comes AFTER the query, so the prompt ends in thousands of
filler characters and there is no answer slot.  Here the filler goes BEFORE the
query and the prompt ends at ``ANSWER THE QUERY``, so a generative or
mask-canvas model answers at the position the task intends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from .exact_tasks import ExactTaskRequest, generate_exact_task
from .models import DATA_SEEDS, DISTRACTORS, FAMILIES, LOADS, POSITIONS

#: The length axis, in RWKV tokens.  The same set the protocol registers, but
#: read as tokens rather than as characters.
TOKEN_LENGTHS: Final = (4096, 8192, 16384, 32768, 65536)

#: The declared length kind.  Distinct from ``logical_character_fixture`` so a
#: reader can never mistake one artifact for the other, and so a cell says which
#: unit it is measured in without consulting this module.
LENGTH_KIND: Final = "rwkv_token_target"

PROMPT_OPEN: Final = "CONTEXT "
EVIDENCE_OPEN: Final = "\nBEGIN EVIDENCE\n"
EVIDENCE_CLOSE: Final = "\nEND EVIDENCE\n"
QUERY_LINE: Final = "\nANSWER THE QUERY"

#: The pad token's text.  A single benign content token, repeated to reach an
#: exact token count.  Padding is done in token ids, never in characters: the
#: pinned vocabulary merges repeated characters, so character-counted filler is
#: wrong by roughly a factor of eight at these lengths (measured: ``'x' * 1964``
#: is 246 tokens).
PAD_TEXT: Final = "x"


class Encode(Protocol):
    """The tokenizer interface this module needs: text in, token ids out."""

    def encode_text(self, text: str) -> tuple[int, ...]: ...


class TokenTargetedError(ValueError):
    """A token-targeted fixture cannot be built as declared."""


@dataclass(frozen=True, slots=True)
class TokenTargetedRequest:
    """One token-targeted fixture declaration."""

    family: str
    data_seed: int
    token_length: int
    position_fraction: int
    load: int
    distractor: str
    instance_index: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise TokenTargetedError(f"undeclared family: {self.family}")
        if self.data_seed not in DATA_SEEDS:
            raise TokenTargetedError(f"undeclared data seed: {self.data_seed}")
        if self.token_length not in TOKEN_LENGTHS:
            msg = (f"undeclared token length: {self.token_length}; the matrix "
                   f"registers {list(TOKEN_LENGTHS)} RWKV tokens")
            raise TokenTargetedError(msg)
        if self.position_fraction not in POSITIONS:
            raise TokenTargetedError(f"undeclared position fraction: {self.position_fraction}")
        if self.load not in LOADS:
            raise TokenTargetedError(f"undeclared load: {self.load}")
        if self.distractor not in DISTRACTORS:
            raise TokenTargetedError(f"undeclared distractor: {self.distractor}")
        if not 0 <= self.instance_index < 200:
            raise TokenTargetedError(f"undeclared instance index: {self.instance_index}")

    def as_character_request(self) -> ExactTaskRequest:
        """The same coordinates on the character fixture, for evidence and gold.

        The evidence is generated once, by the registered generator, so the two
        fixtures cannot drift into testing different tasks.  ``declared_length``
        is fixed at the smallest registered value because it only governs the
        character framing, which this module discards.
        """
        return ExactTaskRequest(
            family=self.family, data_seed=self.data_seed, declared_length=4096,
            position_fraction=self.position_fraction, load=self.load,
            distractor=self.distractor, instance_index=self.instance_index)


@dataclass(frozen=True, slots=True)
class TokenTargetedTask:
    """A token-exact prompt with its evaluator-private gold."""

    prompt_ids: tuple[int, ...]
    answer: str
    answer_ids: tuple[int, ...]
    evidence_tokens: int
    evidence_start_token: int
    request: TokenTargetedRequest

    @property
    def n_prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def prompt_text(self) -> str:
        raise TokenTargetedError(
            "a token-targeted prompt has no canonical text form; decode it with "
            "the same tokenizer if text is needed")


def _encode(encode: Encode, text: str) -> tuple[int, ...]:
    ids = tuple(int(token) for token in encode.encode_text(text))
    if not ids and text:
        raise TokenTargetedError(f"encoding produced no tokens for {text[:40]!r}")
    return ids


def _filler_ids(encode: Encode, filler: str, count: int) -> tuple[int, ...]:  # noqa: ARG001
    """Retired: repetition is not 1:1 through a merge-capable byte-trie.

    Kept as a named refusal rather than deleted, because the assumption it
    encodes is the tempting one and the measurement that kills it is worth
    recording: under the pinned vocabulary ``'x' * 1964`` encodes to **246**
    tokens, not 1964.  Any solver that counts filler in CHARACTERS is wrong by
    a factor of eight at these lengths.  Padding is therefore done in token ids
    (see :func:`pad_token_id`), which is exact by construction.
    """
    raise TokenTargetedError(
        "character-counted filler cannot hit an exact token target; pad with "
        "token ids instead")


def pad_token_id(encode: Encode) -> int:
    """The single id repeated to pad a token-exact prompt.

    One benign content token, repeated.  The resulting id sequence decodes to a
    run of that character, which the tokenizer would itself have merged had it
    been asked to encode the same text -- but the model consumes ids, and the
    registered length axis is tokens, so the id-level construction is the one
    that makes the declared length true.
    """
    ids = _encode(encode, PAD_TEXT)
    if len(ids) != 1:
        msg = f"pad text {PAD_TEXT!r} encodes to {len(ids)} tokens, not 1"
        raise TokenTargetedError(msg)
    return ids[0]


def build_token_prompt(encode: Encode, evidence: str, *,
                       target_tokens: int,
                       position_fraction: int) -> tuple[tuple[int, ...], int, int]:
    """A prompt of EXACTLY ``target_tokens`` ids, ending at the query line.

    Returns ``(ids, evidence_token_count, evidence_start_token)``.  The evidence
    begins at ``position_fraction`` of the total, matching the character
    fixture's convention so the two are comparable on that axis.

    The length is exact by construction rather than by search: the framing and
    the evidence contribute whatever they contribute, and two runs of pad tokens
    fill the remainder -- one before the evidence to place it, one after to
    close the query to the declared end.  The total is asserted anyway, because
    an off-by-one here would silently shift every downstream position statistic.
    """
    if target_tokens <= 0:
        raise TokenTargetedError(f"target length must be positive, got {target_tokens}")
    evidence_ids = _encode(encode, evidence)
    # Encoded as one run so a merge across the framing boundary is counted the
    # way the tokenizer would count it.
    open_ids = _encode(encode, PROMPT_OPEN + EVIDENCE_OPEN)
    close_ids = _encode(encode, EVIDENCE_CLOSE)
    query_ids = _encode(encode, QUERY_LINE)
    pad = pad_token_id(encode)

    evidence_start = target_tokens * position_fraction // 100
    prefix_pad = evidence_start - len(open_ids)
    if prefix_pad < 0:
        msg = (f"the evidence does not fit at position {position_fraction}% of "
               f"{target_tokens} tokens: the opening framing alone is "
               f"{len(open_ids)} tokens")
        raise TokenTargetedError(msg)
    body_pad = (target_tokens - evidence_start - len(evidence_ids)
                - len(close_ids) - len(query_ids))
    if body_pad < 0:
        msg = (f"no room for the evidence and query line at {target_tokens} "
               f"tokens: evidence is {len(evidence_ids)} tokens and "
               f"{-body_pad} tokens are missing")
        raise TokenTargetedError(msg)

    ids = (open_ids + (pad,) * prefix_pad + evidence_ids + close_ids
           + (pad,) * body_pad + query_ids)
    if len(ids) != target_tokens:
        msg = (f"prompt assembled to {len(ids)} tokens, not the declared "
               f"{target_tokens}")
        raise TokenTargetedError(msg)
    return ids, len(evidence_ids), evidence_start


def generate_token_targeted_task(encode: Encode,
                                 request: TokenTargetedRequest) -> TokenTargetedTask:
    """Build one token-exact fixture, evidence and gold from the registered generator."""
    source = generate_exact_task(request.as_character_request())
    evidence = _evidence_of(source.condition.public_prompt)
    ids, evidence_tokens, evidence_start = build_token_prompt(
        encode, evidence, target_tokens=request.token_length,
        position_fraction=request.position_fraction)
    return TokenTargetedTask(
        prompt_ids=ids,
        answer=source.correct_answer,
        answer_ids=_encode(encode, source.correct_answer),
        evidence_tokens=evidence_tokens,
        evidence_start_token=evidence_start,
        request=request)


def _evidence_of(prompt: str) -> str:
    """The evidence block from a character fixture's prompt.

    Parsed from the public prompt rather than regenerated, so the token fixture
    inherits the registered generator's output verbatim -- including its
    distractor, which is part of the task.
    """
    opening = "BEGIN EVIDENCE\n"
    closing = "\nEND EVIDENCE"
    if prompt.count("BEGIN EVIDENCE") != 1 or prompt.count("END EVIDENCE") != 1:
        raise TokenTargetedError("the source prompt has no single evidence block")
    remainder = prompt.partition(opening)[2].partition(closing)[0]
    if not remainder:
        raise TokenTargetedError("the source prompt's evidence block is empty")
    return remainder


def expected_answer_tokens(encode: Encode, answer: str) -> int:
    """How many tokens the gold answer occupies, for sizing the mask canvas."""
    return len(_encode(encode, answer))
