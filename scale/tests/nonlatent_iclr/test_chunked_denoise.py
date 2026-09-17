"""Tests for the chunked denoise path that makes a 64K canvas fit on one GPU.

The equivalence test alone is not enough, and the file is arranged so that is
obvious: a function that quietly materializes the whole ``[1, T, V]`` tensor and
then reduces it returns exactly the same ids and confidences as the chunked one,
so equivalence cannot distinguish them.  The instrument that can is the recorded
spans, which is why ``test_no_provider_span_exceeds_the_chunk`` and the
``assert_no_full_logits`` guard exist, and why both a good and a deliberately
bad provider are exercised.
"""

from __future__ import annotations

import pytest
import torch

from scale.experiments.nonlatent_iclr.longcontext import chunked_denoise as cd

#: Equivalence canvas: large enough to exercise many chunks and a partial tail,
#: small enough to hold the reference tensor in the suite.
EQUIV_TOKENS = 1024

#: A prime vocabulary so chunk boundaries fall at odd vocab offsets, which a
#: width mishandling would disturb.
EQUIV_VOCAB = 257

#: Not a divisor of ``EQUIV_TOKENS``: 10 full chunks of 100 plus a 24-token tail.
EQUIV_CHUNK = 100


def _draw(seed: int, *, total_tokens: int, vocab: int) -> torch.Tensor:
    """A deterministic ``[1, T, vocab]`` logits canvas."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, total_tokens, vocab, generator=generator, dtype=torch.float32)


def _reference(full: torch.Tensor) -> tuple[list[int], list[float]]:
    """The unchunked reduction: softmax the whole canvas, then reduce."""
    probabilities = torch.softmax(full.to(torch.float32), dim=-1)
    return (
        [int(token) for token in probabilities.argmax(dim=-1).reshape(-1).tolist()],
        [float(value) for value in probabilities.amax(dim=-1).reshape(-1).tolist()],
    )


class _RecordingProvider:
    """Wraps a canvas and records every span it is asked for.

    ``returns_full`` models the regression the guard must catch: a provider that
    ignores its span and hands back the whole canvas every time.
    """

    def __init__(self, full: torch.Tensor, *, returns_full: bool = False) -> None:
        self._full = full
        self._returns_full = returns_full
        self.spans: list[tuple[int, int]] = []
        self.peak_elements = 0

    def __call__(self, start: int, end: int) -> torch.Tensor:
        self.spans.append((start, end))
        chunk = self._full if self._returns_full else self._full[:, start:end, :]
        self.peak_elements = max(self.peak_elements, int(chunk.numel()))
        return chunk


def _slicing(full: torch.Tensor):
    """A bare provider that honours the span; nothing is kept between calls."""
    return lambda start, end: full[:, start:end, :]


# --------------------------------------------------------------------------
# the reduction is the one the sampler commits on
# --------------------------------------------------------------------------


def test_chunked_matches_the_full_tensor_reference() -> None:
    # Given: a canvas and its unchunked reduction.
    full = _draw(7, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    reference_ids, reference_conf = _reference(full)
    # When: the same canvas is reduced a chunk at a time.
    ids, confidences = cd.chunked_argmax_confidence(
        _slicing(full), total_tokens=EQUIV_TOKENS, chunk=EQUIV_CHUNK
    )
    # Then: every position returns the same id...
    assert len(ids) == EQUIV_TOKENS
    assert ids == reference_ids
    # ...and the same confidence to within the tolerance the task fixes.  This
    # is what "lossless for the commit decision" means: the two vectors are
    # identical, not merely close.
    assert len(confidences) == EQUIV_TOKENS
    assert max(abs(a - b) for a, b in zip(confidences, reference_conf, strict=True)) < 1e-6


def test_a_chunk_larger_than_the_canvas_is_one_call_and_still_correct() -> None:
    # Given: a chunk wider than the whole canvas.
    full = _draw(11, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    provider = _RecordingProvider(full)
    # When: the reduction runs.
    ids, _ = cd.chunked_argmax_confidence(provider, total_tokens=EQUIV_TOKENS, chunk=2 * EQUIV_TOKENS)
    # Then: it is a single legal call spanning exactly the canvas, which does not
    # exceed the chunk -- so no refusal, and the ids still match.
    assert provider.spans == [(0, EQUIV_TOKENS)]
    assert ids == _reference(full)[0]


# --------------------------------------------------------------------------
# the point: no span, and no returned tensor, exceeds the chunk
# --------------------------------------------------------------------------


def test_no_provider_span_exceeds_the_chunk() -> None:
    """The test that a full-tensor implementation of the path would fail.

    Equivalence passes for that implementation too, so this is the only check
    that separates the chunked path from one that materializes the canvas.
    """
    # Given: a recording provider over a canvas.
    full = _draw(13, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    provider = _RecordingProvider(full)
    # When: the whole range is reduced.
    cd.chunked_argmax_confidence(provider, total_tokens=EQUIV_TOKENS, chunk=EQUIV_CHUNK)
    # Then: every requested span is at most the chunk, they tile the canvas
    # without gaps or overlap, and the reduction is exercised more than once.
    assert provider.spans
    assert all(end - start <= EQUIV_CHUNK for start, end in provider.spans)
    assert provider.spans[0][0] == 0
    assert provider.spans[-1][1] == EQUIV_TOKENS
    assert all(
        left[1] == right[0]
        for left, right in zip(provider.spans, provider.spans[1:])
    )
    assert len(provider.spans) > 1  # a one-call run would not be a chunking test


def test_assert_no_full_logits_accepts_a_span_honouring_provider() -> None:
    # Given: a provider that returns exactly the span it is asked for.
    full = _draw(17, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    # When/Then: the guard runs the chunked path and is satisfied.
    cd.assert_no_full_logits(
        _slicing(full), total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB, chunk=EQUIV_CHUNK
    )


def test_assert_no_full_logits_catches_a_full_canvas_provider() -> None:
    """A provider that returns the whole canvas must be refused, not trusted."""
    # Given: a provider that ignores its span and returns ``[1, T, V]`` every
    # time -- the shape a "silently materializes the full tensor" regression has.
    full = _draw(19, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    provider = _RecordingProvider(full, returns_full=True)
    # When/Then: the guard refuses it, and the message names the over-sized
    # return rather than leaving the caller to guess.
    with pytest.raises(cd.ChunkRefusal, match="more than"):
        cd.assert_no_full_logits(
            provider, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB, chunk=EQUIV_CHUNK
        )


def test_assert_no_full_logits_refuses_a_vocab_mismatch() -> None:
    # Given: a provider whose width disagrees with the declared vocab.
    full = _draw(23, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    # When/Then: the declared vocab is checked, because the guard's whole job is
    # to certify the memory contract and a narrower width would hide a resize.
    with pytest.raises(cd.ChunkRefusal, match="disagrees"):
        cd.assert_no_full_logits(
            _slicing(full), total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB + 1, chunk=EQUIV_CHUNK
        )


# --------------------------------------------------------------------------
# the guard's own tripwires, exercised directly
#
# A correct inner loop never asks for a wider-than-chunk span and always covers
# the canvas, so the guard's span and coverage checks are unreachable through
# the real path and are invisible to every other test in this file.  They only
# fire for a caller that regresses, which is the failure they exist to make
# loud, so each is exercised here by substituting that exact regression for
# ``chunked_argmax_confidence``.  Without these three tests the guard's checks
# could be deleted and the suite would stay green while reporting coverage it
# no longer had.
# --------------------------------------------------------------------------


def test_the_span_tripwire_refuses_a_caller_that_stops_chunking(monkeypatch) -> None:
    """The span check, driven by the one-call regression it uniquely catches.

    Replaces the reduction with the "inner loop no longer chunks" mutation: a
    single ``provider(0, total_tokens)`` request.  The guard must refuse that
    request *before* the wrapped provider is called, so the canvas is never
    allocated -- the whole point of the check.
    """
    # Given: a canvas, and a caller that asks for the whole range in one call.
    full = _draw(43, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    allocated: list[tuple[int, int]] = []

    def provider(start: int, end: int) -> torch.Tensor:
        allocated.append((start, end))
        return full[:, start:end, :]

    def unchunked_caller(caller, *, total_tokens: int, chunk: int):
        caller(0, total_tokens)  # the whole canvas in a single request
        return [], []

    monkeypatch.setattr(cd, "chunked_argmax_confidence", unchunked_caller)
    # When/Then: the guard refuses, its message names the chunk it exceeded, and
    # the canvas was never allocated -- the refusal precedes the provider call.
    with pytest.raises(cd.ChunkRefusal, match="wider than the chunk"):
        cd.assert_no_full_logits(
            provider, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB, chunk=EQUIV_CHUNK
        )
    assert allocated == []


def test_the_coverage_tripwire_refuses_a_caller_that_drops_the_tail(monkeypatch) -> None:
    """The coverage check, driven by the tail-dropping regression.

    Replaces the reduction with the "drop the trailing chunk" mutation, whose
    every request is a legal one (at most ``chunk`` wide).  Because no span is
    oversized, coverage is the only possible refusal, which is what lets this
    test attribute the failure to the coverage check and not the span check.
    """
    # Given: a caller that stops at ``total - chunk`` and so omits the tail.
    full = _draw(47, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    spans: list[tuple[int, int]] = []

    def tail_dropping_caller(caller, *, total_tokens: int, chunk: int):
        for start in range(0, total_tokens - chunk, chunk):
            spans.append((start, start + chunk))
            caller(start, start + chunk)
        return [], []

    monkeypatch.setattr(cd, "chunked_argmax_confidence", tail_dropping_caller)
    # When/Then: the guard refuses on coverage...
    with pytest.raises(cd.ChunkRefusal, match="missing positions"):
        cd.assert_no_full_logits(
            _slicing(full), total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB, chunk=EQUIV_CHUNK
        )
    # ...having seen only legal spans that genuinely fail to reach the end, so
    # the span check could not have been the catcher.
    assert spans
    assert all(end - start <= EQUIV_CHUNK for start, end in spans)
    assert sum(end - start for start, end in spans) < EQUIV_TOKENS


def test_the_guard_refuses_an_over_wide_return_on_its_own(monkeypatch) -> None:
    """The guard's return-width check, isolated from the inner loop's twin.

    ``chunked_argmax_confidence`` checks the returned width too, so on the real
    path the inner check fires first and a guard-only deletion looks harmless --
    the inner still refuses, just with different wording.  Substituting a
    pass-through reduction that performs no shape check leaves the guard's check
    as the *only* thing that can refuse the over-wide return, which is what
    makes this test able to tell the guard's layer apart from the inner's.
    """
    # Given: a provider that returns the whole canvas for a legal request, and a
    # reduction that asks only legal spans and does no checking of its own.
    full = _draw(53, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    provider = _RecordingProvider(full, returns_full=True)

    def pass_through(caller, *, total_tokens: int, chunk: int):
        for start in range(0, total_tokens, chunk):
            caller(start, min(start + chunk, total_tokens))
        return [], []

    monkeypatch.setattr(cd, "chunked_argmax_confidence", pass_through)
    # When/Then: the guard refuses on its own message, and the request it made
    # was a legal one -- the over-wide *return* is the only violation present.
    with pytest.raises(cd.ChunkRefusal, match="materialized more than"):
        cd.assert_no_full_logits(
            provider, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB, chunk=EQUIV_CHUNK
        )
    assert provider.spans and all(end - start <= EQUIV_CHUNK for start, end in provider.spans)


# --------------------------------------------------------------------------
# the memory claim, over a canvas too large to materialize
# --------------------------------------------------------------------------


def test_a_64k_canvas_reduces_without_ever_holding_it() -> None:
    """The headline claim, tested at the frozen length with a stand-in vocab.

    A full 65536 x 65536 canvas cannot be allocated in the suite, so the vocab
    is shrunk to keep the per-chunk tensor tiny while the *declared* length stays
    the real 65536.  The assertion is on the provider's peak per-call element
    count, which is the memory the chunked path is allowed to touch.
    """
    # Given: the frozen canvas length, a stand-in vocab, and a chunk.
    total_tokens = cd.CANVAS_TOKENS
    vocab = 8
    chunk = 1024
    generator = torch.Generator().manual_seed(29)

    spans: list[tuple[int, int]] = []
    peak_elements = 0

    def provider(start: int, end: int) -> torch.Tensor:
        # Generated per call: nothing approximating the full canvas is ever
        # built, which is the property under test.
        spans.append((start, end))
        canvas = torch.randn(1, end - start, vocab, generator=generator, dtype=torch.float32)
        nonlocal peak_elements
        peak_elements = max(peak_elements, int(canvas.numel()))
        return canvas

    # When: the whole range is reduced.
    ids, confidences = cd.chunked_argmax_confidence(provider, total_tokens=total_tokens, chunk=chunk)
    # Then: every position is decided...
    assert len(ids) == total_tokens
    assert len(confidences) == total_tokens
    # ...the peak per-call tensor is exactly one chunk of logits, not the canvas...
    assert peak_elements == chunk * vocab
    assert peak_elements < total_tokens * vocab
    # ...and the number of calls is the number of chunks, with the last one the
    # exact remainder.
    assert len(spans) == total_tokens // chunk
    assert spans[-1] == (total_tokens - chunk, total_tokens)
    # The unchunked pair for this geometry is the chunked workspace times the
    # number of chunks -- the factor chunking removes.
    assert cd.full_denoise_bytes(total_tokens, vocab) == (
        total_tokens // chunk
    ) * cd.chunk_denoise_bytes(chunk, vocab)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_zero_or_negative_chunk_is_refused() -> None:
    full = _draw(31, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    # Given/When/Then: chunk = 0 cannot tile the range and is refused...
    with pytest.raises(cd.ChunkRefusal, match="chunk must be positive"):
        cd.chunked_argmax_confidence(_slicing(full), total_tokens=EQUIV_TOKENS, chunk=0)
    # ...as is a negative chunk, for the same reason.
    with pytest.raises(cd.ChunkRefusal, match="chunk must be positive"):
        cd.chunked_argmax_confidence(_slicing(full), total_tokens=EQUIV_TOKENS, chunk=-1)


def test_a_non_positive_total_tokens_is_refused() -> None:
    full = _draw(37, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    # Given/When/Then: an empty or reversed range has no commit decision to make.
    with pytest.raises(cd.ChunkRefusal, match="total_tokens must be positive"):
        cd.chunked_argmax_confidence(_slicing(full), total_tokens=0, chunk=EQUIV_CHUNK)


def test_a_provider_returning_a_wider_span_is_refused() -> None:
    # Given: a provider that returns the whole canvas for a small request.
    full = _draw(41, total_tokens=EQUIV_TOKENS, vocab=EQUIV_VOCAB)
    provider = _RecordingProvider(full, returns_full=True)
    # When/Then: the reduction refuses the over-wide return on shape, because
    # that return is the full-canvas allocation the chunk contract forbids.
    with pytest.raises(cd.ChunkRefusal, match="wider than the request"):
        cd.chunked_argmax_confidence(provider, total_tokens=EQUIV_TOKENS, chunk=EQUIV_CHUNK)


def test_a_changing_vocabulary_is_refused() -> None:
    # Given: a provider whose width changes between chunks.
    def provider(start: int, end: int) -> torch.Tensor:
        width = EQUIV_VOCAB if start == 0 else EQUIV_VOCAB + 1
        return torch.zeros(1, end - start, width)

    # When/Then: mismatched widths are refused, since the ids across chunks would
    # not be comparable and the confidence would be measured against a different
    # distribution.
    with pytest.raises(cd.ChunkRefusal, match="vocabulary changed"):
        cd.chunked_argmax_confidence(provider, total_tokens=EQUIV_TOKENS, chunk=EQUIV_CHUNK)


# --------------------------------------------------------------------------
# the arithmetic the docstring states
# --------------------------------------------------------------------------


def test_the_64k_workspace_constants_are_the_documented_ones() -> None:
    # Given: the frozen canvas and vocabulary widths.
    # When/Then: one fp32 logits tensor is 16 GiB and the logits+prob pair is
    # 34_359_738_368 bytes -- the 32 GiB (34.4 GB) the plan quotes, and the
    # number chunking replaces.
    assert cd.FULL_LOGITS_BYTES == 17_179_869_184
    assert cd.FULL_DENOISE_BYTES == 34_359_738_368
    assert cd.full_denoise_bytes(cd.CANVAS_TOKENS, cd.MODEL_VOCAB_SIZE) == cd.FULL_DENOISE_BYTES
    # And the chunked workspace is independent of the canvas length.
    assert cd.chunk_denoise_bytes(4096, cd.MODEL_VOCAB_SIZE) == 2_147_483_648
    assert cd.chunk_denoise_bytes(4096, cd.MODEL_VOCAB_SIZE) * 16 == cd.FULL_DENOISE_BYTES
