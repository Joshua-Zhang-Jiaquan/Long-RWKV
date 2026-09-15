"""Length-matrix tests: shard coverage, digest-only rows, and replay rejection."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import os
from typing import Final, cast

import pytest

from scale.experiments.nonlatent_iclr.tasks import length_matrix as m
from scale.experiments.nonlatent_iclr.tasks.length_qualification import QualificationInputs
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import SafeRWKVTokenizer, VOCAB_NAME, VOCAB_SHA256, load_safe_tokenizer

MODEL: Final = Path(os.environ.get(
    "NONLATENT_MODEL_DIR",
    Path(__file__).resolve().parents[3] / "external/models/RWKV7-Goose-World3-2.9B-HF"))
ASSETS: Final = Path("DAN/nonlatent_iclr/task4_assets/tokenizer")
RECEIPT: Final = Path(".omo/evidence/nonlatent-rnn-dlm-iclr-research/task-04/tokenizer-qualification/8ef417180818408d8bc65efcb4dcd75c/receipt.json")
INPUTS: Final = QualificationInputs(MODEL, ASSETS, RECEIPT)


@pytest.fixture(scope="module")
def tokenizer() -> SafeRWKVTokenizer:
    return load_safe_tokenizer(MODEL / VOCAB_NAME, VOCAB_SHA256)


def test_shard_bounds_partition_every_unit_exactly_once() -> None:
    # Given: the declared unit grid.
    total = m.unit_count()
    # When: it is split into an awkward number of shards.
    covered = [unit for index in range(7) for unit in range(*m.shard_bounds(index, 7))]
    # Then: the shards tile the grid with no gap and no overlap.
    assert covered == list(range(total))
    assert total == 720 * 5


def test_shard_bounds_reject_invalid_coordinates() -> None:
    # Given: out-of-range shard coordinates.
    # When / Then: they are refused rather than silently clamped.
    for index, count in ((0, 0), (3, 3), (-1, 4)):
        with pytest.raises(ValueError, match="shard"):
            _ = m.shard_bounds(index, count)


def test_unit_coordinates_cover_cells_and_seeds() -> None:
    # Given: the flat unit grid.
    coordinates = {m.unit_coordinates(unit) for unit in range(m.unit_count())}
    # When / Then: every (cell, seed) pair appears exactly once.
    assert len(coordinates) == m.unit_count()
    assert {seed for _cell, seed in coordinates} == set(range(len(m.DATA_SEEDS)))


def test_measure_unit_is_digest_only_and_reproducible(tokenizer: SafeRWKVTokenizer) -> None:
    # Given: one representative unit.
    # When: it is measured twice.
    first = m.measure_unit(0, tokenizer)
    second = m.measure_unit(0, tokenizer)
    # Then: the result is deterministic and carries no prompt text.
    assert first == second
    keys = set(asdict(first))
    assert not [key for key in keys if "prompt" in key or "text" in key]
    assert first.feasible_instances + first.infeasible_instances == m.INSTANCES_PER_CELL
    assert first.token_min <= first.token_max
    assert first.token_total >= first.token_min * first.feasible_instances


def test_write_shard_is_write_once(tmp_path: Path) -> None:
    # Given: a published shard.
    output = tmp_path / m.MATRIX_NAME
    rows: list[dict[str, object]] = [{"family": "x"}]
    path = m.write_shard(output, 0, 2, rows)
    # When: it is rewritten unchanged, then with different rows.
    assert m.write_shard(output, 0, 2, rows) == path
    # Then: a conflicting rewrite is refused.
    with pytest.raises(RuntimeError, match="already exists"):
        _ = m.write_shard(output, 0, 2, [{"family": "y"}])


def test_assemble_requires_full_coverage(tmp_path: Path) -> None:
    # Given: a single shard covering only part of the grid.
    output = tmp_path / m.MATRIX_NAME
    path = m.write_shard(output, 0, 4, [{"family": "x"}])
    # When / Then: assembling an incomplete set is refused.
    with pytest.raises(RuntimeError, match="expected"):
        _ = m.assemble([path], output, INPUTS)


def test_verify_rejects_structural_damage_without_replaying(tmp_path: Path) -> None:
    # Given: artifacts whose envelope is wrong in one way each.
    good = cast("dict[str, object]", json.loads(m.artifact_bytes([{"family": "x"}] * m.unit_count(), INPUTS).decode("utf-8")))
    cases: dict[str, dict[str, object]] = {}
    tampered = dict(good)
    tampered["schema_version"] = 99
    cases["schema"] = tampered
    tampered = dict(good)
    tampered["unit_count"] = 1
    cases["units"] = tampered
    tampered = dict(good)
    tampered["source_hashes"] = {"other.py": "0" * 64}
    cases["sources"] = tampered
    tampered = dict(good)
    tampered["digests"] = [{"family": "x"}]
    cases["digests"] = tampered
    for name, payload in cases.items():
        path = tmp_path / f"{name}.json"
        _ = path.write_text(json.dumps(payload), encoding="utf-8")
        # When / Then: each is refused.
        assert m.verify_matrix_artifact(path, INPUTS, sample=1) is False, name


def test_verify_rejects_a_row_that_does_not_replay(tmp_path: Path) -> None:
    # Given: a full-length artifact whose rows are placeholders.
    path = tmp_path / m.MATRIX_NAME
    _ = path.write_bytes(m.artifact_bytes([{"family": "placeholder"}] * m.unit_count(), INPUTS))
    # When: replay checks the sampled units.
    # Then: the fabricated rows are detected.
    assert m.verify_matrix_artifact(path, INPUTS, sample=2) is False


def test_source_hashes_bind_the_matrix_module_itself() -> None:
    # Given: the matrix artifact's source bindings.
    hashes = m.source_hashes(INPUTS)
    # When / Then: the measuring code is part of what it binds, so edits invalidate it.
    assert "length_matrix.py" in hashes
    assert "repo" not in hashes
    assert all(len(value) == 64 for value in hashes.values())


def test_artifact_never_serialises_prompt_text() -> None:
    # Given: one measured row.
    payload = m.artifact_bytes([{"family": "associative_recall", "token_total": 5}], INPUTS).decode("utf-8")
    # When / Then: the serialized artifact is aggregates only, and marks itself qualified.
    parsed = cast("dict[str, object]", json.loads(payload))
    assert parsed["matrix_token_lengths_qualified"] is True
    assert parsed["instances_per_cell"] == m.INSTANCES_PER_CELL
    assert "BEGIN EVIDENCE" not in payload


# The artifacts these tests exercise are not part of the repository: they are
# large, rights-gated, and produced on a cluster. A clone must SKIP rather than
# fail, so the suite reports honestly what it can verify without them. Point
# NONLATENT_MODEL_DIR at a prepared root to run them.
pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason=f"external artifact absent: {MODEL} -- set the matching "
           f"NONLATENT_* environment variable to a prepared root")
