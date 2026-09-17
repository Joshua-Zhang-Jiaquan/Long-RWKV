"""Deterministic shard-contiguous sampling for packed FineWeb data.

Two independent resume defects lived in this sampler, and both are fixed here:

1. **``set_epoch`` was never called.**  ``__iter__`` seeds its permutation from
   ``seed + epoch`` and traverses from the first row, so with ``epoch`` pinned
   at its default 0 every call yielded the identical order starting at row 0.
   The trainer re-creates ``iter(train_loader)`` when the iterator is exhausted,
   so even a single uninterrupted run re-read the same rows pass after pass; and
   every checkpoint resume restarted at row 0 again.
2. **No data position was persisted anywhere.**  ``meta.json`` records steps and
   ``tokens_seen``, which is steps x batch -- a count of row-READS, not of data
   seen.  A run that reports 16B tokens over a 24.4M-row blend may have touched
   a few percent of it several times.

Measured consequence on the 2.9B Lane A continuation: about **1.43M unique rows
of 24.39M (5.9 %)** across six resumes, against a schedule that claims 5.3M
rows.  The fix has to make a resumed run continue where it stopped rather than
begin again, and this file does it without touching the trainer -- which is
hash-pinned in ``architecture_contract.json`` and must not be edited to paper
over a data bug.

How the position is recovered
-----------------------------

The launcher passes the step being resumed FROM, and the per-rank rows consumed
per optimizer step, through the environment:

``RWKV_DATA_START_STEP``      global step the run resumes from (0 for a fresh run)
``RWKV_DATA_ROWS_PER_STEP``   per-rank rows per step = microbatch x grad_accum

From those and the sampler's own per-epoch length this file derives the epoch and
the offset within it, so pass *e* of the resumed run is the same pass the
uninterrupted run would have been in.  Nothing is persisted by the sampler: the
checkpoint's step is the position, and deriving it keeps the two from drifting.

With the launcher silent (both variables absent) the behaviour is exactly the old
behaviour, so an unrelated caller is not silently changed -- but it is also then
still defective, which is why the launcher always sets them.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypedDict, Unpack, final

import numpy as np
from torch.utils.data import Sampler
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Environment variables carrying the resume position.
START_STEP_ENV = "RWKV_DATA_START_STEP"
ROWS_PER_STEP_ENV = "RWKV_DATA_ROWS_PER_STEP"


class PositionRefusal(ValueError):
    """The declared resume position cannot be honoured."""


def resume_position(*, start_step: int, rows_per_step: int,
                    rows_per_epoch: int) -> tuple[int, int]:
    """``(epoch, skip_rows)`` for a run resuming at ``start_step``.

    ``skip_rows`` is the offset INTO that epoch, in this rank's rows.  Together
    they place the resumed run at the same point in the same permutation the
    uninterrupted run would have reached, which is the whole point: without it
    the resumed run re-reads the head of the data it already trained on.

    Refuses a negative position rather than clamping.  A negative start step
    means the caller's arithmetic is wrong, and silently treating it as zero
    would turn a bookkeeping bug into a quietly restarted run.
    """
    if start_step < 0:
        raise PositionRefusal(f"start_step must be non-negative, got {start_step}")
    if rows_per_step <= 0:
        raise PositionRefusal(f"rows_per_step must be positive, got {rows_per_step}")
    if rows_per_epoch <= 0:
        raise PositionRefusal(
            f"rows_per_epoch must be positive, got {rows_per_epoch}; the sampler "
            f"has no rows to place a position in")
    consumed = start_step * rows_per_step
    return consumed // rows_per_epoch, consumed % rows_per_epoch


def position_from_env(*, rows_per_epoch: int,
                      env: dict[str, str] | None = None) -> tuple[int, int]:
    """Read the launch-time position, defaulting to "start from the beginning".

    Absent variables are the honest default for a caller that is not a resumed
    training run, and they mean epoch 0, offset 0 -- which is exactly what the
    sampler did before this fix.
    """
    source = os.environ if env is None else env
    raw_step = source.get(START_STEP_ENV)
    raw_rows = source.get(ROWS_PER_STEP_ENV)
    if raw_step is None and raw_rows is None:
        return 0, 0
    if raw_step is None or raw_rows is None:
        missing = START_STEP_ENV if raw_step is None else ROWS_PER_STEP_ENV
        msg = (f"{missing} is not set but its partner is; the resume position "
               f"needs both, and honouring half of it would place the run at a "
               f"position nobody chose")
        raise PositionRefusal(msg)
    try:
        start_step = int(raw_step)
        rows_per_step = int(raw_rows)
    except ValueError as exc:
        msg = f"non-integer resume position: {raw_step!r} / {raw_rows!r}"
        raise PositionRefusal(msg) from exc
    return resume_position(start_step=start_step, rows_per_step=rows_per_step,
                           rows_per_epoch=rows_per_epoch)


class _SamplerOptions(TypedDict, total=False):
    world_size: int
    rank: int
    shuffle: bool
    seed: int


@final
class ShardContiguousSampler(Sampler[int]):
    """Yield stable per-rank blocks while retaining one-shard cache locality."""

    def __init__(
        self,
        boundaries: Sequence[int],
        *args: int,
        **options: Unpack[_SamplerOptions],
    ) -> None:
        """Accept the legacy positional and keyword sampler options."""
        values = [1, 0, 1, 42]
        for index, value in enumerate(args):
            if index >= len(values):
                message = "ShardContiguousSampler accepts four optional positional arguments"
                raise TypeError(message)
            values[index] = value
        world_size = options.get("world_size", values[0])
        rank = options.get("rank", values[1])
        shuffle = options.get("shuffle", bool(values[2]))
        seed = options.get("seed", values[3])
        super().__init__()
        self.boundaries = list(boundaries)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._starts = [0, *self.boundaries[:-1]]
        self._counts = [
            boundary - self._starts[index] for index, boundary in enumerate(self.boundaries)
        ]
        self._per_rank = [count // self.world_size for count in self._counts]
        self._length = sum(self._per_rank)
        #: Epoch and in-epoch offset the run is resuming at.  Read once, here,
        #: because the position belongs to the run rather than to an iteration.
        #: A zero-length sampler keeps the old tolerant behaviour instead of
        #: raising: it has no rows to place a position in, and refusing there
        #: would turn a degenerate boundary list into a startup crash that the
        #: previous version did not have.
        if self._length > 0:
            self.epoch, self._pending_skip = position_from_env(
                rows_per_epoch=self._length)
        else:
            self.epoch, self._pending_skip = 0, 0

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch seed offset."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """Return the equal post-rounding sample count for this rank."""
        return self._length

    @property
    def pending_skip(self) -> int:
        """Rows still to be skipped in this rank's stream, for the next iteration."""
        return self._pending_skip

    @override
    def __iter__(self) -> Iterator[int]:
        """Yield the original RNG state consumption and shard traversal order.

        The skip applies to the FIRST iteration only: it exists to place a
        resumed run at its position inside an epoch it had already partly seen,
        and applying it again on a later pass would silently rewind the run.
        The epoch advances after each iteration so that a second pass over the
        loader reads DIFFERENT data rather than repeating the first -- which is
        the second half of the defect, and the reason a long run could report
        billions of tokens while touching a few percent of the blend.
        """
        generator = np.random.default_rng(self.seed + self.epoch)
        skip = self._pending_skip
        self._pending_skip = 0
        self.epoch += 1
        shard_order = list(range(len(self._counts)))
        if self.shuffle:
            generator.shuffle(shard_order)
        for shard_index in shard_order:
            per_rank = self._per_rank[shard_index]
            if per_rank <= 0:
                continue
            local = list(range(self._counts[shard_index]))
            if self.shuffle:
                generator.shuffle(local)
            block = local[self.rank * per_rank : (self.rank + 1) * per_rank]
            start = self._starts[shard_index]
            for local_index in block:
                if skip > 0:
                    skip -= 1
                    continue
                yield start + local_index
