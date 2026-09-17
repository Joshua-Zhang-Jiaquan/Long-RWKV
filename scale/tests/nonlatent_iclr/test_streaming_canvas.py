from __future__ import annotations

import pytest

from scale.experiments.nonlatent_iclr.qualification import streaming_canvas as canvas

#: (total_tokens, chunk) pairs spanning divisible, non-divisible, tiny, and
#: single-chunk canvases.  The non-divisible cases are the ones a naive
#: ``range(total // chunk)`` plan silently truncates.
COVERAGE_CASES = [
    (100, 40),
    (100, 25),
    (100, 7),
    (37, 10),
    (1, 1),
    (5, 100),
    (0, 3),
]


@pytest.mark.parametrize(("total_tokens", "chunk"), COVERAGE_CASES)
def test_plan_chunks_covers_canvas_exactly_without_overlap(total_tokens: int, chunk: int) -> None:
    # Given: a canvas and a chunk width, overlap defaulting to zero.
    # When: the streaming plan is built.
    plan = canvas.plan_chunks(total_tokens=total_tokens, chunk=chunk)
    # Then: the spans partition [0, total_tokens) -- half-open, gapless, indices
    # in order, and every span no wider than the chunk.
    if total_tokens == 0:
        assert plan == []
        return
    assert plan[0].start == 0
    assert plan[-1].end == total_tokens
    assert [access.index for access in plan] == list(range(len(plan)))
    for access in plan:
        assert 0 < access.end - access.start <= chunk
    for previous, current in zip(plan, plan[1:]):
        assert current.start == previous.end
    assert sum(access.end - access.start for access in plan) == total_tokens


def test_plan_chunks_shares_tokens_when_overlap_requested() -> None:
    # Given: an overlap smaller than the chunk.
    plan = canvas.plan_chunks(total_tokens=100, chunk=40, overlap=10)
    # When / Then: neighbours share the overlap, and the union still reaches the
    # tail -- the plan is usable evidence that the log can then refute.
    assert plan[0] == canvas.ChunkAccess(index=0, start=0, end=40)
    assert plan[1].start == 30
    assert plan[1].start < plan[0].end
    assert plan[-1].end == 100


def test_plan_chunks_refuses_zero_chunk() -> None:
    # Given: a zero-width chunk, which cannot make forward progress.
    # When / Then: the plan refuses rather than returning an empty or endless one.
    with pytest.raises(canvas.StreamingRefusal, match="chunk must be positive"):
        canvas.plan_chunks(total_tokens=100, chunk=0)


@pytest.mark.parametrize("overlap", [-1, -5])
def test_plan_chunks_refuses_negative_overlap(overlap: int) -> None:
    # Given: a negative overlap, which would rewind the stride.
    with pytest.raises(canvas.StreamingRefusal, match="overlap must be nonnegative"):
        canvas.plan_chunks(total_tokens=100, chunk=40, overlap=overlap)


@pytest.mark.parametrize("overlap", [40, 41])
def test_plan_chunks_refuses_overlap_not_smaller_than_chunk(overlap: int) -> None:
    # Given: an overlap that zeroes or reverses the stride.
    # When / Then: refuse, because the plan would never cover the canvas.
    with pytest.raises(canvas.StreamingRefusal, match="not smaller than chunk"):
        canvas.plan_chunks(total_tokens=100, chunk=40, overlap=overlap)


def test_access_log_returns_entries_in_read_order_as_a_copy() -> None:
    # Given: two recorded spans.
    log = canvas.AccessLog()
    first = canvas.ChunkAccess(index=0, start=0, end=40)
    second = canvas.ChunkAccess(index=1, start=40, end=80)
    log.record(first)
    log.record(second)
    # When: the caller reads and mutates the returned list.
    entries = log.entries()
    entries.append(canvas.ChunkAccess(index=2, start=80, end=100))
    # Then: order is preserved and the live log is untouched, so evidence cannot
    # be rewritten after an assertion has already run.
    assert log.entries() == [first, second]


def test_assert_no_reread_accepts_a_gapless_streaming_plan() -> None:
    # Given: the executed log of a genuine streaming run.
    log = canvas.AccessLog()
    for access in canvas.plan_chunks(total_tokens=100, chunk=40):
        log.record(access)
    # When / Then: every token was read once, in order, so no refusal.
    log.assert_no_reread(total_tokens=100)


def test_assert_no_reread_catches_reread_and_names_the_second_chunk() -> None:
    # Given: a run that re-read tokens 30..39 -- the exact hidden reread the
    # failure QA probes, which produces streaming-looking numbers.
    log = canvas.AccessLog()
    log.record(canvas.ChunkAccess(index=0, start=0, end=40))
    log.record(canvas.ChunkAccess(index=1, start=30, end=70))
    # When: the log is audited.
    with pytest.raises(canvas.StreamingRefusal) as error:
        log.assert_no_reread(total_tokens=100)
    # Then: the refusal names the offending chunk and the reread.
    assert "chunk 1" in str(error.value)
    assert "reread" in str(error.value)


def test_assert_no_reread_catches_a_gap_and_names_the_second_chunk() -> None:
    # Given: a log that skips tokens 40..49 (a non-contiguous read).
    log = canvas.AccessLog()
    log.record(canvas.ChunkAccess(index=0, start=0, end=40))
    log.record(canvas.ChunkAccess(index=1, start=50, end=100))
    # When: the log is audited.
    with pytest.raises(canvas.StreamingRefusal) as error:
        log.assert_no_reread(total_tokens=100)
    # Then: a gap is refused as loudly as a reread.
    assert "chunk 1" in str(error.value)
    assert "gap" in str(error.value)


def test_assert_no_reread_catches_a_non_contiguous_index() -> None:
    # Given: spans that tile the canvas but whose chunk indices skip.
    log = canvas.AccessLog()
    log.record(canvas.ChunkAccess(index=0, start=0, end=40))
    log.record(canvas.ChunkAccess(index=1, start=40, end=80))
    log.record(canvas.ChunkAccess(index=3, start=80, end=100))
    # When / Then: the out-of-order record is refused by position.
    with pytest.raises(canvas.StreamingRefusal, match="position 2"):
        log.assert_no_reread(total_tokens=100)


def test_assert_no_reread_refuses_an_unread_canvas() -> None:
    # Given: no reads recorded for a nonempty canvas.
    log = canvas.AccessLog()
    # When / Then: silence is not a streaming claim.
    with pytest.raises(canvas.StreamingRefusal, match="no chunk access recorded"):
        log.assert_no_reread(total_tokens=10)


def test_assert_no_reread_accepts_an_empty_canvas() -> None:
    # Given: an empty canvas, whose empty plan reads nothing.
    log = canvas.AccessLog()
    # When / Then: the empty log is honest here.
    assert canvas.plan_chunks(total_tokens=0, chunk=8) == []
    log.assert_no_reread(total_tokens=0)


def test_assert_no_reread_refuses_a_truncated_canvas() -> None:
    # Given: a log that stops before the tail (last end != total).
    log = canvas.AccessLog()
    log.record(canvas.ChunkAccess(index=0, start=0, end=40))
    log.record(canvas.ChunkAccess(index=1, start=40, end=80))
    # When / Then: an unread tail is refused.
    with pytest.raises(canvas.StreamingRefusal, match="tail"):
        log.assert_no_reread(total_tokens=100)


def test_assert_forward_only_accepts_a_forward_plan() -> None:
    # Given: a forward-only streaming log.
    log = canvas.AccessLog()
    for access in canvas.plan_chunks(total_tokens=100, chunk=40):
        log.record(access)
    # When / Then: the walk advances strictly, so it passes.
    log.assert_forward_only()


def test_assert_forward_only_refuses_a_backward_record() -> None:
    # Given: a reversed traversal -- a chunk-local reverse pass leaking in.
    log = canvas.AccessLog()
    log.record(canvas.ChunkAccess(index=1, start=40, end=80))
    log.record(canvas.ChunkAccess(index=0, start=0, end=40))
    # When: the direction decision is enforced.
    with pytest.raises(canvas.StreamingRefusal) as error:
        log.assert_forward_only()
    # Then: the backward step is named and no reverse stream is accepted.
    assert "chunk 0" in str(error.value)
    assert "forward-only" in str(error.value)


def test_assert_forward_only_refuses_a_repeated_index() -> None:
    # Given: the same chunk recorded twice, which is a re-read, not forward walk.
    log = canvas.AccessLog()
    log.record(canvas.ChunkAccess(index=0, start=0, end=40))
    log.record(canvas.ChunkAccess(index=0, start=0, end=40))
    # When / Then: the repeated index is refused.
    with pytest.raises(canvas.StreamingRefusal, match="index did not advance"):
        log.assert_forward_only()


@pytest.mark.parametrize(
    "recorded",
    [
        "",
        canvas.RECORDED_FULL_MODEL_PREFIX_CACHE,
        canvas.RECORDED_LOOP_PREFIX_CACHE,
        "qualified",
    ],
)
def test_streaming_is_qualified_stays_false_without_the_exact_value(recorded: str) -> None:
    # Given: an empty, a recorded NOT_QUALIFIED, or a near-miss qualification.
    # When: the streaming verdict is requested.
    qualified, reason = canvas.streaming_is_qualified(full_model_prefix_cache=recorded)
    # Then: never True by default, and the reason states the gap.
    assert qualified is False
    assert reason.startswith("NOT QUALIFIED")


def test_streaming_is_qualified_reason_names_the_recorded_value() -> None:
    # Given: the program's actual full-model prefix-cache qualification.
    # When: the verdict is requested.
    qualified, reason = canvas.streaming_is_qualified(
        full_model_prefix_cache=canvas.RECORDED_FULL_MODEL_PREFIX_CACHE
    )
    # Then: the reason quotes the recorded NOT_QUALIFIED string rather than
    # asserting a gap the reader cannot see.
    assert qualified is False
    assert canvas.RECORDED_FULL_MODEL_PREFIX_CACHE in reason


def test_streaming_is_qualified_true_only_for_the_exact_value() -> None:
    # Given: the literal qualified value.
    # When / Then: it is the sole input that qualifies streaming.
    qualified, reason = canvas.streaming_is_qualified(
        full_model_prefix_cache=canvas.QUALIFIED_VALUE
    )
    assert qualified is True
    assert reason.startswith("QUALIFIED")
