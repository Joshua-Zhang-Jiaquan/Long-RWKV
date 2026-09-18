"""The runner must report what it measured, and must not report what it did not.

The failure this file exists against is a cell that produced nothing parseable being
scored as zero accuracy: at that point a harness failure and a model failure are the
same number, and the long-context claim inherits whichever one happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.longcontext import runner as rn
from scale.experiments.nonlatent_iclr.tasks.token_targeted_matrix import (
    BOUNDED_LENGTHS,
    BOUNDED_LOADS,
    CellCoordinates,
)

LENGTHS = BOUNDED_LENGTHS
BOUNDED_LOAD = BOUNDED_LOADS[0]


class FakeEncoder:
    """One id per character, and a decode that inverts it exactly."""

    def encode_text(self, text: str) -> list[int]:
        return [(ord(ch) % 30000) + 1000 for ch in text]

    def decode_ids(self, ids) -> str:
        return "".join(chr((int(i) - 1000) % 30000) for i in ids)


def _gen(text: str):
    """A generator that always returns the same text."""
    def generate(prompt_ids, *, max_new_tokens):
        return text
    return generate


@pytest.fixture()
def encode() -> FakeEncoder:
    return FakeEncoder()


def _cell(family: str = "associative_recall", length: int = LENGTHS[0],
          load: int = BOUNDED_LOAD) -> CellCoordinates:
    return CellCoordinates(family=family, token_length=length, load=load, data_seed=101)


# --------------------------------------------------------------------------
# what was measured
# --------------------------------------------------------------------------


def test_a_cell_whose_completions_parse_is_measured(encode: FakeEncoder) -> None:
    # Given: a generator that always emits a value token the family's grammar accepts.
    run = rn.run_cell(_cell(), encode=encode, generate=_gen("the value is v123"),
                      instances=2)
    # When/Then: the cell is measured, and nothing is reported missing.
    assert run.measured is True
    assert run.instances_scored == 2
    assert run.n_missing == 0
    assert run.accuracy is not None


def test_a_cell_whose_completions_do_not_parse_is_UNMEASURED_not_zero(
        encode: FakeEncoder) -> None:
    """The defect this file exists against.

    A harness failure and a model failure must not produce the same number, so a cell
    with nothing parseable reports no accuracy at all rather than 0.0 -- and the count
    of missing completions is reported beside it rather than absorbed into it.
    """
    # Given: a generator whose output has no answer token for this family.
    run = rn.run_cell(_cell(), encode=encode, generate=_gen("I cannot answer that."),
                      instances=3)
    # When/Then: nothing was scored, and the cell says so.
    assert run.instances_scored == 0
    assert run.n_missing == 3
    assert run.measured is False


def test_the_grid_names_the_unmeasured_cells_rather_than_dropping_them(
        encode: FakeEncoder) -> None:
    # Given: a grid whose completions never parse.
    grid = rn.run_grid(encode=encode, generate=_gen("nothing to score"), units=[0, 1],
                       instances=2)
    # When/Then: the macro score is absent, and the cells it would have covered are named.
    assert grid["cells_measured"] == 0
    assert grid["macro_score"] is None
    assert len(grid["cells_unmeasured"]) == 2


def test_the_macro_score_covers_only_the_cells_it_measured(encode: FakeEncoder) -> None:
    # Given: a grid where every cell parses.
    grid = rn.run_grid(encode=encode, generate=_gen("v7"), units=[0, 1, 2], instances=2)
    # When/Then: the score is present, covers the measured cells, and says so.
    assert grid["macro_score"] is not None
    assert grid["cells_unmeasured"] == []
    assert "measured cells only" in grid["note"]


# --------------------------------------------------------------------------
# the access modes
# --------------------------------------------------------------------------


def test_a_full_canvas_cell_makes_no_chunk_accesses(encode: FakeEncoder) -> None:
    run = rn.run_cell(_cell(), encode=encode, generate=_gen("v1"), instances=1,
                      access_mode=rn.FULL_CANVAS)
    assert run.chunks == []
    assert run.reread_checked is False


def test_a_streaming_cell_records_its_chunks_and_checks_the_log(
        encode: FakeEncoder) -> None:
    """The access claim travels with the number it produced.

    A full-canvas run relabelled as streaming produces identical scores, so the chunks
    and the checked log ARE the evidence that the mode is what its name says.
    """
    run = rn.run_cell(_cell(length=LENGTHS[0]), encode=encode, generate=_gen("v1"),
                      instances=1, access_mode=rn.STREAMING, stream_chunk=4096)
    assert run.chunks, "a streaming cell must record its accesses"
    assert run.reread_checked is True
    # contiguous and gapless over the prompt
    assert run.chunks[0].start == 0
    for a, b in zip(run.chunks, run.chunks[1:], strict=False):
        assert a.end == b.start


def test_the_two_modes_are_paired_on_the_same_instances(encode: FakeEncoder) -> None:
    # Given: two generators over the same cells.
    result = rn.paired_access_modes(encode=encode, generate_a=_gen("v1"),
                                    generate_b=_gen("v2"), units=[0, 1], instances=2)
    # When/Then: the pairing is reported per cell, not pooled.
    assert result["paired_cells"] == 2
    assert len(result["delta_streaming_minus_full"]) == 2


def test_an_unknown_access_mode_is_refused(encode: FakeEncoder) -> None:
    with pytest.raises(rn.RunnerRefusal, match="unknown access mode"):
        rn.run_cell(_cell(), encode=encode, generate=_gen("v1"), access_mode="hybrid")


def test_zero_instances_is_refused(encode: FakeEncoder) -> None:
    with pytest.raises(rn.RunnerRefusal, match="must be positive"):
        rn.run_cell(_cell(), encode=encode, generate=_gen("v1"), instances=0)


def test_each_instance_gets_its_own_access_log(encode: FakeEncoder) -> None:
    r"""A per-CELL log refuses every multi-instance streaming cell, correctly.

    Each instance is its own stream starting at position 0, so a shared log reads
    instance 1's first chunk as a backwards jump and refuses -- which it must, because
    "no chunk sees a token before its own start" is a claim about ONE stream. The bug was
    the sharing, not the refusal, and it made every streaming cell with more than one
    instance unreadable.
    """
    run = rn.run_cell(_cell(), encode=encode, generate=_gen("v1"), instances=3,
                      access_mode=rn.STREAMING, stream_chunk=4096)
    # Then: all three instances are scored, and the access claim was checked for each.
    assert run.instances_scored == 3
    assert run.reread_checked is True


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def test_the_grid_records_the_mode_semantics_not_just_its_name(
        encode: FakeEncoder) -> None:
    """`streaming` as a bare label is a claim; the semantics are what was done."""
    grid = rn.run_grid(encode=encode, generate=_gen("v1"), units=[0], instances=1,
                       access_mode=rn.STREAMING)
    assert grid["access_mode"] == rn.STREAMING
    assert "chunks" in grid["access_mode_semantics"]
    assert "before its own start" in grid["access_mode_semantics"]


def test_the_grid_round_trips_through_disk(tmp_path: Path, encode: FakeEncoder) -> None:
    import json
    grid = rn.run_grid(encode=encode, generate=_gen("v1"), units=[0], instances=1)
    path = rn.write_grid(grid, tmp_path / "grid.json")
    again = json.loads(path.read_text())
    assert again["schema"] == rn.SCHEMA
    assert again["cells_measured"] == 1


def test_the_macro_key_does_not_silently_drop_the_load_axis(encode: FakeEncoder) -> None:
    r"""Four loads per (family, length) overwrote each other under a two-part key.

    The macro key is (family, token_length) -- nine cells, as the protocol declares -- but
    the bounded cut has four LOADS per family and length. Keying on the first two alone let
    each load overwrite the previous one: measured on the cut, 36 cells collapsed to 9 keys
    and 27 vanished from the score with no error, because assigning over a duplicate dict
    key is not a failure. The macro then reported one load as if it were the cell.

    A skip is a claim, and this was a skip with no denominator.
    """
    from scale.experiments.nonlatent_iclr.tasks.token_targeted_matrix import (
        BOUNDED_LOADS, cell_coordinates, cell_count)

    # Given: units that share a family and length but differ in load.
    units = [u for u in range(cell_count())
             if (lambda c: (c.family, c.token_length) == ("associative_recall", LENGTHS[0]))
             (cell_coordinates(u))]
    assert len(units) == len(BOUNDED_LOADS), "the cut must have one cell per load"
    # When: the grid runs.
    grid = rn.run_grid(encode=encode, generate=_gen("v1"), units=units, instances=2)
    # Then: ONE macro cell, averaged over all four loads -- and the per-cell average is
    # reported, so the averaging is inspectable rather than trusted.
    assert grid["cells_measured"] == 1
    assert len(grid["cells_averaged"]) == 1
    key, cell = next(iter(grid["cells_averaged"].items()))
    assert key == f"associative_recall@{LENGTHS[0]}"
    # THE COUNT IS THE ASSERTION. Averaging reads 4; overwriting reads 1, because a
    # duplicate dict key keeps only the last value. The first version of this test asserted
    # only the averaged value and could not tell the two apart -- it survived the mutation
    # restoring the collapse, which is what a decoration test does.
    assert cell["n_loads"] == len(BOUNDED_LOADS), (
        f"the macro cell averaged {cell['n_loads']} loads, not {len(BOUNDED_LOADS)}: the "
        f"load axis is being dropped rather than averaged")
