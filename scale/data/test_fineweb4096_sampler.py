"""The packed-data sampler must not re-read its own head after a resume.

The defect these tests pin was measured on the 2.9B Lane A continuation: six
resumes, each rebuilding the identical permutation and restarting at row 0, so a
schedule claiming 5.3M rows had touched about 1.43M unique rows of a 24.39M-row
blend.  Two separate causes, and a test for each:

* the position was never recovered, so a resume began again;
* the epoch never advanced, so a second pass over the loader repeated the first.

Both are silent.  Nothing crashes, nothing logs a warning, and `tokens_seen`
keeps climbing -- which is why the tests check the DATA rather than the counter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scale.data.fineweb4096_sampler import (  # noqa: E402
    ROWS_PER_STEP_ENV, START_STEP_ENV, PositionRefusal, ShardContiguousSampler,
    position_from_env, resume_position)


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------


def test_a_resume_lands_where_the_uninterrupted_run_would_have_been() -> None:
    # Given: a run that consumed 1000 steps of 8 rows each, in epochs of 300 rows.
    epoch, skip = resume_position(start_step=1000, rows_per_step=8, rows_per_epoch=300)
    # When/Then: 8000 rows consumed = 26 whole epochs and 200 rows into the 27th.
    assert epoch == 8000 // 300
    assert skip == 8000 % 300
    assert epoch * 300 + skip == 8000


def test_a_fresh_run_starts_at_the_beginning() -> None:
    assert resume_position(start_step=0, rows_per_step=8, rows_per_epoch=300) == (0, 0)


def test_a_position_that_exactly_ends_an_epoch_does_not_skip_into_the_next() -> None:
    # Given: a step count that lands exactly on an epoch boundary.
    epoch, skip = resume_position(start_step=75, rows_per_step=4, rows_per_epoch=300)
    # When/Then: it is the start of the NEXT epoch, with nothing skipped -- the
    # off-by-one that would drop 300 rows of real data.
    assert (epoch, skip) == (1, 0)


@pytest.mark.parametrize("kwargs,pattern", [
    ({"start_step": -1, "rows_per_step": 8, "rows_per_epoch": 300}, "non-negative"),
    ({"start_step": 1, "rows_per_step": 0, "rows_per_epoch": 300}, "rows_per_step"),
    ({"start_step": 1, "rows_per_step": 8, "rows_per_epoch": 0}, "rows_per_epoch"),
])
def test_an_impossible_position_is_refused_not_clamped(kwargs, pattern: str) -> None:
    # Given: a position nobody can honour.
    # When/Then: refused. Clamping a negative step to zero would turn a
    # bookkeeping bug into a quietly restarted run, which is the failure the fix
    # exists to remove.
    with pytest.raises(PositionRefusal, match=pattern):
        resume_position(**kwargs)


# --------------------------------------------------------------------------
# the environment contract
# --------------------------------------------------------------------------


def test_the_launcher_silence_is_the_old_behaviour() -> None:
    # Given: no position in the environment.
    # When/Then: epoch 0, no skip -- exactly what the sampler did before, so an
    # unrelated caller is not silently moved.
    assert position_from_env(rows_per_epoch=300, env={}) == (0, 0)


def test_half_a_position_is_refused() -> None:
    """Honouring half would place the run somewhere nobody chose."""
    # Given: a step with no rows-per-step, then the reverse.
    with pytest.raises(PositionRefusal, match=ROWS_PER_STEP_ENV):
        position_from_env(rows_per_epoch=300, env={START_STEP_ENV: "1000"})
    with pytest.raises(PositionRefusal, match=START_STEP_ENV):
        position_from_env(rows_per_epoch=300, env={ROWS_PER_STEP_ENV: "8"})


def test_a_non_integer_position_is_refused() -> None:
    with pytest.raises(PositionRefusal, match="non-integer"):
        position_from_env(rows_per_epoch=300,
                          env={START_STEP_ENV: "later", ROWS_PER_STEP_ENV: "8"})


def test_the_environment_position_reaches_the_sampler(monkeypatch) -> None:
    # Given: a sampler built under a resume position.
    monkeypatch.setenv(START_STEP_ENV, "100")
    monkeypatch.setenv(ROWS_PER_STEP_ENV, "4")
    sampler = ShardContiguousSampler([300], 1, 0, 0, 42)
    # When/Then: 400 rows consumed of a 300-row epoch is epoch 1 with 100 skipped.
    assert sampler.epoch == 1
    assert sampler.pending_skip == 100


# --------------------------------------------------------------------------
# the behaviour that actually matters: the rows
# --------------------------------------------------------------------------


def _sampler(*, boundaries=(12,), seed: int = 42) -> ShardContiguousSampler:
    return ShardContiguousSampler(list(boundaries), 1, 0, 1, seed)


def test_a_second_pass_reads_different_rows() -> None:
    """The intra-run half of the defect: the loader is re-created when exhausted."""
    # Given: a sampler over a 12-row epoch.
    sampler = _sampler()
    first = list(sampler)
    second = list(sampler)
    # When/Then: the two passes are permutations of the same data but NOT the
    # same order -- with the epoch pinned they were identical, so a run could
    # report billions of tokens while re-reading its first few percent.
    assert sorted(first) == sorted(second)
    assert first != second


def test_a_resumed_run_skips_the_rows_it_already_saw(monkeypatch) -> None:
    # Given: the order an uninterrupted run would have seen, and the same run
    # resumed after consuming its first 5 rows.
    order = list(_sampler())
    monkeypatch.setenv(START_STEP_ENV, "5")
    monkeypatch.setenv(ROWS_PER_STEP_ENV, "1")
    resumed = _sampler()
    # When: the resumed run iterates.
    rest = list(resumed)
    # Then: it continues the SAME permutation exactly where it stopped. Not a
    # restart from the head, which is what the 2.9B run did six times over.
    assert rest == order[5:]
    assert rest != order[: len(rest)]
    assert resumed.pending_skip == 0


def test_the_skip_is_consumed_by_one_iteration_only(monkeypatch) -> None:
    # Given: a resumed sampler.
    monkeypatch.setenv(START_STEP_ENV, "5")
    monkeypatch.setenv(ROWS_PER_STEP_ENV, "1")
    sampler = _sampler()
    assert sampler.pending_skip == 5
    first = list(sampler)
    # When/Then: the first pass yields only the unseen tail, and the second
    # yields a FULL epoch -- applying the skip twice would silently rewind the
    # run and re-read the head a third time.
    assert len(first) == 7
    assert sampler.pending_skip == 0
    assert len(list(sampler)) == 12


def test_a_run_that_never_resumes_never_skips() -> None:
    # Given: a sampler with no resume position.
    sampler = _sampler()
    # When/Then: the first pass covers the whole epoch.
    assert len(list(sampler)) == 12


def test_the_epoch_advances_rather_than_being_pinned() -> None:
    # Given: a sampler.
    sampler = _sampler()
    # When/Then: each iteration moves the epoch on, which is what makes the
    # permutation seed change between passes.
    assert sampler.epoch == 0
    _ = list(sampler)
    assert sampler.epoch == 1
    _ = list(sampler)
    assert sampler.epoch == 2


def test_per_rank_blocks_stay_disjoint_across_world_sizes() -> None:
    # Given: two ranks over the same boundaries.
    rank0 = ShardContiguousSampler([8, 16], 2, 0, 1, 42)
    rank1 = ShardContiguousSampler([8, 16], 2, 1, 1, 42)
    # When/Then: their passes partition the data and do not overlap, so the
    # position fix cannot have introduced cross-rank duplication.
    a, b = set(rank0), set(rank1)
    assert not (a & b)
    assert len(a) == len(list(rank0))


def test_a_degenerate_boundary_list_is_tolerated() -> None:
    # Given: boundaries that give no rows to a rank.
    sampler = ShardContiguousSampler([1], 8, 7, 1, 42)
    # When/Then: it yields nothing rather than raising -- the previous version
    # tolerated this and the position fix must not turn it into a crash.
    assert sampler.epoch == 0
    assert list(sampler) == []
