"""Streaming access-mode seam and its audit log.

Streaming is the access mode that makes the constant-recurrent-state advantage
*measurable*: an RWKV-7 layer carries O(1) state, so a chunked forward pass pays
the same per-token cost at the end of a long canvas as at its start, while a
full-canvas run re-attends everything.  The measurement is only meaningful if
the run really streamed -- if it merely *looked* chunked.

That distinction is the whole reason this module exists.  A full-canvas run that
is relabelled "streaming" consumes the same tokens, produces the same logits,
and would report the same numbers, so nothing in the metric stream can tell the
two apart.  The only evidence that chunk ``k`` never saw a token before its own
``start`` is the access log: :class:`AccessLog` records the spans a run actually
read, and :meth:`AccessLog.assert_no_reread` refuses a log in which a chunk
re-reads a token.  The research plan's failure QA probes exactly this -- a
"hidden token reread in streaming mode" -- and this is the artifact that answers
it.

Two honest limits, recorded here rather than smoothed over, because the paper
must present streaming as a *weaker* access mode, not a free one:

1. **The forward stream only.**  Streaming executes the forward direction
   (``alpha == 1``).  The reverse stream runs in flipped coordinates and is
   deliberately denied the cache, so it cannot stream at all.  Mixing a
   chunk-local reverse pass into a streaming run is not the trained
   bidirectional model and would make the access-mode comparison
   uninterpretable; :meth:`AccessLog.assert_forward_only` refuses that log.
2. **The loop arm is not qualified.**  The tied layer-state aliases backing the
   loop arm's prefix cache are ``NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES``,
   so no streaming claim may be attached to the loop arm regardless of how the
   full-model prefix cache fares.  :func:`streaming_is_qualified` returns
   ``False`` unless the recorded full-model prefix cache is literally
   ``QUALIFIED``; it never defaults to ``True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

#: The forward direction is the only one the recurrent state can stream from.
#: ``alpha == 1`` is the RWKV-7 forward stream; the reverse pass runs in flipped
#: coordinates and is denied the cache, so it has no streaming form.
FORWARD_STREAM_ALPHA: Final[int] = 1

#: The reverse stream is intentionally not cached.  Named so a caller cannot
#: read "streaming" as covering both directions.
REVERSE_STREAM_CACHED: Final[bool] = False

#: The only value of the full-model prefix-cache qualification that permits a
#: streaming claim.  Any ``NOT_QUALIFIED_*`` string, an empty string, or an
#: unrecorded value leaves streaming unqualified.
QUALIFIED_VALUE: Final[str] = "QUALIFIED"

#: The qualification this program currently records for the full-model prefix
#: cache (mirrors ``architecture_semantics_models.SemanticsQualification``).
#: The FFN token-shift state is unwired, so the cache does not yet exist.
RECORDED_FULL_MODEL_PREFIX_CACHE: Final[str] = "NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED"

#: The loop arm's prefix cache, which stays unqualified for a second and
#: independent reason: its tied layer-state aliases are not separate states.
RECORDED_LOOP_PREFIX_CACHE: Final[str] = "NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES"


class StreamingRefusal(ValueError):
    """A streaming plan or access log violates the access-mode contract.

    Raised rather than returning a plan or a verdict.  The failure this guards
    is a *false access claim*: a run whose numbers are indistinguishable from
    streaming but whose tokens were re-read (or read out of order) would enter
    the paper as a streaming measurement it is not.  There is no benign default
    -- a malformed plan has no valid interpretation -- so every violation is a
    named refusal, never a silent ``None`` or a partial plan.
    """


@dataclass(frozen=True, slots=True)
class ChunkAccess:
    """One half-open token span ``[start, end)`` a run read, in read order.

    ``index`` is the chunk's position in the streaming plan; it is distinct from
    the span so a log can show both *which* chunk and *which tokens* it read,
    and so an out-of-order replay (same spans, wrong sequence) is observable.
    """

    index: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Log:
    """Mutable storage behind :class:`AccessLog`, kept off the public surface."""

    entries: list[ChunkAccess] = field(default_factory=list)


class AccessLog:
    """Append-only record of the token spans a streaming run actually read.

    The log is the evidence for the access claim, so it must be written by the
    run as it reads and never reconstructed from the plan afterward: a plan and
    an execution can diverge, and only the executed log can falsify the claim.
    """

    __slots__ = ("_log",)

    def __init__(self) -> None:
        self._log = _Log()

    def record(self, access: ChunkAccess) -> None:
        """Append one read span.  Order is significant and never re-sorted."""
        self._log.entries.append(access)

    def entries(self) -> list[ChunkAccess]:
        """Return a copy of the recorded spans in read order.

        A copy, not the live list: a caller that mutates the returned list must
        not be able to rewrite the evidence after an assertion has run.
        """
        return list(self._log.entries)

    def assert_no_reread(self, *, total_tokens: int) -> None:
        """Refuse a log in which any chunk reads a token already read.

        A reread is ``current.start < previous.end``: chunk ``k`` dipped back
        before the end of the chunk recorded before it.  It is refused by name
        so the offending chunk is identifiable.  Two neighbours of the same
        fault are refused too, because each also invalidates the log as
        streaming evidence: a **gap** (``current.start > previous.end``) means
        the covered span is not contiguous, and an **out-of-order record**
        (``current.index`` not equal to its position) means the log is not a
        forward walk at all.  The first span must open at 0 and the last must
        close at ``total_tokens``, so the union of spans covers the canvas
        exactly; anything short of that is not the whole canvas.
        """
        if total_tokens < 0:
            raise StreamingRefusal(f"total_tokens must be nonnegative, got {total_tokens}")
        recorded = self._log.entries
        if not recorded:
            # An empty log is honest only for an empty canvas.
            if total_tokens == 0:
                return
            raise StreamingRefusal(
                f"no chunk access recorded for a {total_tokens}-token canvas: "
                "an unread canvas cannot support a streaming claim"
            )
        if recorded[0].start != 0:
            raise StreamingRefusal(
                f"first recorded chunk {recorded[0].index} starts at {recorded[0].start}, "
                "not 0: the canvas prefix was never read in streaming order"
            )
        for position, access in enumerate(recorded):
            if access.end <= access.start:
                raise StreamingRefusal(
                    f"chunk {access.index} has an empty or inverted span "
                    f"[{access.start}, {access.end})"
                )
            if access.end > total_tokens:
                raise StreamingRefusal(
                    f"chunk {access.index} ends at {access.end}, past the {total_tokens}-token canvas"
                )
            if access.index != position:
                raise StreamingRefusal(
                    f"chunk {access.index} recorded at position {position}: the log is not a "
                    "forward walk, so its access claim cannot be read as streaming"
                )
            if position == 0:
                continue
            previous = recorded[position - 1]
            if access.start < previous.end:
                raise StreamingRefusal(
                    f"chunk {access.index} starts at {access.start} before the end "
                    f"{previous.end} of chunk {previous.index}: token reread in streaming mode"
                )
            if access.start > previous.end:
                raise StreamingRefusal(
                    f"chunk {access.index} starts at {access.start} after the end "
                    f"{previous.end} of chunk {previous.index}: gap leaves the canvas uncovered"
                )
        if recorded[-1].end != total_tokens:
            raise StreamingRefusal(
                f"last recorded chunk {recorded[-1].index} ends at {recorded[-1].end}, "
                f"not {total_tokens}: the canvas tail was never read"
            )

    def assert_forward_only(self) -> None:
        """Refuse a log whose reads do not advance forward through the canvas.

        Streaming executes the forward stream (``alpha == 1``).  The reverse
        stream runs in flipped coordinates and is deliberately denied the cache,
        so it has no streaming form; a chunk-local reverse pass is not the
        trained bidirectional model, and mixing one in would make the
        access-mode comparison uninterpretable -- the chunked run and the
        full-canvas run would no longer be the same function.  This method
        enforces the only part of that decision an access log can witness: the
        recorded reads must advance, strictly, in both chunk index and token
        position.  A repeated or decreasing index is a re-read or a reverse
        traversal, and either falsifies "forward-only".
        """
        previous: ChunkAccess | None = None
        for access in self._log.entries:
            if previous is not None:
                if access.index <= previous.index:
                    raise StreamingRefusal(
                        f"chunk {access.index} recorded after chunk {previous.index}: index did not "
                        "advance, so the walk is not forward-only (alpha == 1)"
                    )
                if access.start <= previous.start:
                    raise StreamingRefusal(
                        f"chunk {access.index} starts at {access.start}, not past {previous.start}: "
                        "the record walks backward, which the forward stream cannot do"
                    )
            previous = access


def plan_chunks(*, total_tokens: int, chunk: int, overlap: int = 0) -> list[ChunkAccess]:
    """Plan half-open chunk spans covering ``[0, total_tokens)`` exactly.

    ``overlap == 0`` yields contiguous, gapless chunks whose union is the whole
    canvas and whose spans partition it -- this is the streaming plan.  A
    positive ``overlap`` shares ``overlap`` tokens between neighbours (stride
    ``chunk - overlap``), which still covers the canvas but re-reads tokens; such
    a plan is *not* streaming, and recording it makes
    :meth:`AccessLog.assert_no_reread` refuse, which is how an overlapping run
    is kept from being reported as a streaming one.

    ``chunk <= 0`` or ``overlap < 0`` refuses.  So does ``overlap >= chunk``:
    the stride ``chunk - overlap`` is then nonpositive, so the plan would never
    advance past its first span and would not cover ``[0, total_tokens)`` at
    all -- a silent partial plan is worse than a refusal.
    """
    if chunk <= 0:
        raise StreamingRefusal(f"chunk must be positive, got {chunk}")
    if overlap < 0:
        raise StreamingRefusal(f"overlap must be nonnegative, got {overlap}")
    if overlap >= chunk:
        raise StreamingRefusal(
            f"overlap {overlap} is not smaller than chunk {chunk}: stride would be "
            f"{chunk - overlap}, so the plan would never advance to cover the canvas"
        )
    if total_tokens < 0:
        raise StreamingRefusal(f"total_tokens must be nonnegative, got {total_tokens}")
    stride = chunk - overlap
    chunks: list[ChunkAccess] = []
    start = 0
    index = 0
    while start < total_tokens:
        end = min(start + chunk, total_tokens)
        chunks.append(ChunkAccess(index=index, start=start, end=end))
        index += 1
        start += stride
    return chunks


def streaming_is_qualified(*, full_model_prefix_cache: str) -> tuple[bool, str]:
    """Verdict and reason for whether streaming may be claimed.

    True only when the recorded full-model prefix cache is literally
    ``QUALIFIED``.  It never returns True by default: an empty or unrecorded
    qualification, and every ``NOT_QUALIFIED_*`` string this program currently
    records, return False with a reason naming the gap.  The loop arm is
    unaffected by this value -- its prefix cache is unqualified for the
    independent reason recorded above -- so a qualified full-model cache would
    not make the loop arm streamable.
    """
    if full_model_prefix_cache == QUALIFIED_VALUE:
        return True, (
            "QUALIFIED: forward-only chunked access is backed by the qualified "
            "full-model prefix cache (alpha == 1)"
        )
    if not full_model_prefix_cache:
        gap = "no full-model prefix-cache qualification was recorded"
    else:
        gap = f"the recorded full-model prefix-cache value is {full_model_prefix_cache!r}"
    reason = (
        f"NOT QUALIFIED: streaming requires the full-model prefix cache to be "
        f"{QUALIFIED_VALUE!r}, but {gap}"
    )
    return False, reason
