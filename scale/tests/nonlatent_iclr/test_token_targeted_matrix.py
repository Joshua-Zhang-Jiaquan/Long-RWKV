"""The token-targeted grid must be exact, and the artifact must refuse when it is not.

Every test here exists because the character matrix has the defect this grid was
built to fix: its ``65536`` cells are 65,536 CHARACTERS, i.e. 9K--17K RWKV tokens,
so its labels are not its units.  A second grid that made the same mistake one
layer down would be worse than the first, because the label would finally look
right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.tasks import token_targeted_matrix as ttm


class FakeEncoder:
    """One id per character, deterministic, so lengths are hand-checkable."""

    def encode_text(self, text: str) -> list[int]:
        return [(ord(ch) % 60000) + 100 for ch in text]


@pytest.fixture()
def encode() -> FakeEncoder:
    return FakeEncoder()


# --------------------------------------------------------------------------
# the coordinates
# --------------------------------------------------------------------------


def test_the_bounded_cut_has_the_declared_size() -> None:
    # Given: 3 families x 3 lengths x 4 loads x 1 seed.
    # When/Then: the flat index covers exactly that, or the artifact would claim
    # a grid it does not have.
    assert ttm.cell_count() == 3 * 3 * 4 * 1 == 36


def test_coordinates_round_trip_through_the_flat_index() -> None:
    """The forward and inverse maps must not drift apart."""
    for unit in range(ttm.cell_count()):
        assert ttm.coordinate_units(ttm.cell_coordinates(unit)) == unit


def test_the_round_trip_is_injective() -> None:
    # Given: every unit in the cut.
    cells = {ttm.cell_coordinates(u).cell_id for u in range(ttm.cell_count())}
    # When/Then: no two units share a cell -- a collision would make the artifact
    # hold fewer cells than it reports.
    assert len(cells) == ttm.cell_count()


def test_a_unit_outside_the_cut_is_refused_not_clamped() -> None:
    # Given: an index past the end.
    # When/Then: refused, because clamping would alias it onto a real cell and
    # the artifact would report coverage it does not have.
    for unit in (-1, ttm.cell_count()):
        with pytest.raises(ttm.MatrixRefusal, match="outside the"):
            ttm.cell_coordinates(unit)


def test_the_excluded_family_is_actually_excluded() -> None:
    """finite_hmm answers with a parsed state sequence, so it averages different
    units with the other three."""
    assert "finite_hmm" not in ttm.BOUNDED_FAMILIES
    with pytest.raises(ttm.MatrixRefusal, match="not on the bounded cut"):
        ttm.coordinate_units(ttm.CellCoordinates(
            family="finite_hmm", token_length=16384, load=1, data_seed=101))


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def test_a_measured_cell_is_exact_at_every_declared_length(encode: FakeEncoder) -> None:
    # Given: the shortest and longest cells in the cut.
    for length in ttm.BOUNDED_LENGTHS:
        cell = ttm.CellCoordinates(family="associative_recall", token_length=length,
                                   load=1, data_seed=101)
        # When: it is measured.
        measurement = ttm.measure_cell(cell, encode, instances=2)
        # Then: the measured count EQUALS the declared length -- the property the
        # whole cohort exists for.
        assert measurement.measured_token_count == length
        assert measurement.exact is True
        assert measurement.length_kind == ttm.LENGTH_KIND


def test_the_length_kind_is_not_the_character_kind() -> None:
    # Given: the two artifacts' declared kinds.
    # When/Then: they differ, so a reader can tell which unit a cell is in without
    # consulting this module.
    assert ttm.LENGTH_KIND == "rwkv_token_target"
    assert ttm.LENGTH_KIND != "logical_character_fixture"


def test_an_unregistered_token_length_is_refused(encode: FakeEncoder) -> None:
    with pytest.raises(ttm.MatrixRefusal, match="not a registered token length"):
        ttm.measure_cell(ttm.CellCoordinates(family="associative_recall",
                                             token_length=12345, load=1, data_seed=101),
                         encode, instances=1)


def test_zero_instances_is_refused(encode: FakeEncoder) -> None:
    with pytest.raises(ttm.MatrixRefusal, match="must be positive"):
        ttm.measure_cell(ttm.CellCoordinates(family="associative_recall",
                                             token_length=16384, load=1, data_seed=101),
                         encode, instances=0)


def test_the_prompt_hash_commits_to_the_fixtures(encode: FakeEncoder) -> None:
    # Given: the same cell measured twice.
    cell = ttm.CellCoordinates(family="code_dataflow", token_length=16384, load=8,
                               data_seed=101)
    first = ttm.measure_cell(cell, encode, instances=2)
    second = ttm.measure_cell(cell, encode, instances=2)
    # When/Then: the hash is reproducible, so the artifact commits to the exact
    # fixtures rather than to a recipe that could produce different ones.
    assert first.prompt_sha256 == second.prompt_sha256
    # and a different load is a different cell with a different hash
    other = ttm.measure_cell(ttm.CellCoordinates(family="code_dataflow",
                                                 token_length=16384, load=32, data_seed=101),
                             encode, instances=2)
    assert other.prompt_sha256 != first.prompt_sha256


# --------------------------------------------------------------------------
# the guard that is live: an artifact whose label is not its unit
# --------------------------------------------------------------------------


def _measurement(length: int, measured: int | None = None) -> ttm.CellMeasurement:
    return ttm.CellMeasurement(
        cell_id=f"associative_recall@{length}L1s101", family="associative_recall",
        declared_token_length=length, measured_token_count=measured if measured is not None
        else length, length_kind=ttm.LENGTH_KIND, load=1, data_seed=101, instances=2,
        prompt_sha256="0" * 64, gold_sha256="1" * 64)


def test_an_inexact_cell_is_refused_at_write_time() -> None:
    """The defect this whole module exists to prevent, caught at the boundary.

    A cell whose declared length is 65536 and whose fixture measures 17000 is
    EXACTLY the character matrix's failure one layer down. It must not be
    serialisable.
    """
    # Given: a measurement that disagrees with its own label.
    bad = _measurement(65536, measured=17000)
    # When/Then: refused, and the message names the cell.
    with pytest.raises(ttm.MatrixRefusal, match="do not measure their declared token length"):
        ttm.artifact_bytes([bad])


def test_the_refusal_lists_the_offending_cells() -> None:
    with pytest.raises(ttm.MatrixRefusal) as caught:
        ttm.artifact_bytes([_measurement(16384), _measurement(65536, measured=1)])
    assert "associative_recall@65536L1s101" in str(caught.value)


def test_a_complete_grid_serialises_and_verifies(tmp_path: Path, encode: FakeEncoder) -> None:
    # Given: every cell of the bounded cut, measured.
    measurements = [ttm.measure_cell(ttm.cell_coordinates(u), encode, instances=2)
                    for u in range(ttm.cell_count())]
    # When: the artifact is written and verified.
    path = ttm.write_matrix(measurements, tmp_path / ttm.MATRIX_NAME)
    # Then: it verifies, and says which grid produced it.
    assert ttm.verify_matrix_artifact(path) is True
    document = json.loads(path.read_text())
    assert document["length_kind"] == ttm.LENGTH_KIND
    assert document["declared_grid"]["families"] == list(ttm.BOUNDED_FAMILIES)
    assert len(document["cells"]) == ttm.cell_count()
    # and the scope string must not let a reader mistake the cut for the full grid
    assert "BOUNDED" in document["scope"]
    assert "CHARACTERS" in document["why_separate_from_length_matrix"]


def test_an_incomplete_cell_list_is_refused(tmp_path: Path, encode: FakeEncoder) -> None:
    # Given: an artifact missing cells.
    measurements = [ttm.measure_cell(ttm.cell_coordinates(u), encode, instances=2)
                    for u in range(ttm.cell_count() - 1)]
    path = tmp_path / "partial.json"
    path.write_bytes(ttm.artifact_bytes(measurements))
    # When/Then: verification refuses, because a grid that quietly lost a cell
    # would read as a grid that ran.
    with pytest.raises(ttm.MatrixRefusal, match="but the bounded cut has"):
        ttm.verify_matrix_artifact(path)


def test_a_doctored_artifact_is_refused(tmp_path: Path, encode: FakeEncoder) -> None:
    """The check must fail on a hand-edited artifact, or it guards nothing."""
    # Given: a valid artifact with one cell's measured length edited to lie.
    measurements = [ttm.measure_cell(ttm.cell_coordinates(u), encode, instances=2)
                    for u in range(ttm.cell_count())]
    path = ttm.write_matrix(measurements, tmp_path / ttm.MATRIX_NAME)
    document = json.loads(path.read_text())
    document["cells"][0]["measured_token_count"] += 1
    path.write_text(json.dumps(document))
    # When/Then: refused.
    with pytest.raises(ttm.MatrixRefusal, match="measured"):
        ttm.verify_matrix_artifact(path)


def test_resampling_re_derives_the_fixtures(tmp_path: Path, encode: FakeEncoder) -> None:
    # Given: a valid artifact.
    measurements = [ttm.measure_cell(ttm.cell_coordinates(u), encode, instances=2)
                    for u in range(ttm.cell_count())]
    path = ttm.write_matrix(measurements, tmp_path / ttm.MATRIX_NAME)
    # When/Then: re-measuring a sample agrees with what was recorded.
    assert ttm.verify_matrix_artifact(path, sample=3, encode=encode) is True


def test_resampling_a_changed_fixture_is_refused(tmp_path: Path, encode: FakeEncoder) -> None:
    # Given: an artifact whose recorded prompt hash does not match the fixtures.
    measurements = [ttm.measure_cell(ttm.cell_coordinates(u), encode, instances=2)
                    for u in range(ttm.cell_count())]
    path = ttm.write_matrix(measurements, tmp_path / ttm.MATRIX_NAME)
    document = json.loads(path.read_text())
    document["cells"][0]["prompt_sha256"] = "f" * 64
    path.write_text(json.dumps(document))
    # When/Then: refusing, because the fixtures changed under the artifact.
    with pytest.raises(ttm.MatrixRefusal, match="fixtures changed"):
        ttm.verify_matrix_artifact(path, sample=1, encode=encode)


def test_the_wrong_length_kind_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"length_kind": "logical_character_fixture", "cells": []}))
    with pytest.raises(ttm.MatrixRefusal, match="not 'rwkv_token_target'"):
        ttm.verify_matrix_artifact(path)
