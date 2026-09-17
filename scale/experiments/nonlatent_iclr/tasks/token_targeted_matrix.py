"""The token-targeted fixture matrix: a SECOND artifact, never a revision of the first.

Why a second artifact rather than a corrected first one
------------------------------------------------------

``length_matrix.json`` qualifies 3,600 character-declared cells and is
hash-bound into the preregistration.  Its ``65536`` cells are 65,536 *characters*,
which is 9K--17K RWKV tokens depending on the evidence, so a reader who takes the
label for the unit would believe this program has measured contexts it has never
run.  Rewriting that artifact would break every hash that references it, and it
would hide the discrepancy rather than record it.  So the token-exact grid is a
new artifact with its own ``LENGTH_KIND``, and both stay.

The invariant that makes it worth having
----------------------------------------

Every cell declares a token length and records the length it ACTUALLY measured.
Those two must be equal, and :func:`verify_matrix_artifact` refuses the artifact
if any cell disagrees.  That check is the entire reason this module exists: a
fixture that is "about 16K tokens" would reproduce exactly the defect the
character matrix has, one layer down, where it would be much harder to notice
because the label would finally look right.

The bounded cut
---------------

The full grid is 720 cells x 200 instances x 5 seeds x 6 models x 2 access modes,
which is not a first run.  :data:`BOUNDED_*` is the cut the plan runs first ---
three families, three lengths, one seed, 50 instances --- and it is declared here
rather than at the call site so an artifact says which grid produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol, Sequence

from .token_targeted_tasks import (
    LENGTH_KIND,
    TOKEN_LENGTHS,
    Encode,
    TokenTargetedError,
    TokenTargetedRequest,
    generate_token_targeted_task,
)

SCHEMA_VERSION: Final = 1
MATRIX_NAME: Final = "token_targeted_matrix.json"

#: The families whose units match the macro score.  ``finite_hmm`` is excluded by
#: the protocol for a reason that is not editorial: its answer is a parsed state
#: sequence, not a single value, so averaging it with the others would average
#: different units.
BOUNDED_FAMILIES: Final = ("associative_recall", "overwrite_delayed_query",
                           "code_dataflow")

#: The bounded cut's length axis.  4096 and 8192 are inside the training canvas
#: and already covered by the character matrix at their true token count.
BOUNDED_LENGTHS: Final = (16384, 32768, 65536)

#: The evidence-distance loads the bounded cut spans.
BOUNDED_LOADS: Final = (1, 8, 32, 128)

#: One data seed for the first cut.  The five registered seeds are for the full
#: grid; using one here is what makes the cut affordable, and it is declared so
#: the artifact cannot be mistaken for a five-seed result.
BOUNDED_DATA_SEEDS: Final = (101,)

#: Instances per cell in the bounded cut.  The full grid registers 200.
BOUNDED_INSTANCES: Final = 50

#: The evidence position fraction.  One value: the position axis is a separate
#: registered axis and folding it in here would multiply the cut by three.
BOUNDED_POSITION: Final = 50

#: The distractor.  ``similar`` is the hardest and is what the bounded cut uses;
#: the axis is registered and not varied here for the same reason as position.
BOUNDED_DISTRACTOR: Final = "similar"


class MatrixRefusal(ValueError):
    """The token-targeted matrix cannot be built or verified as declared."""


class SafeEncoder(Encode, Protocol):
    """The tokenizer interface this module needs, named for the reader."""


@dataclass(frozen=True, slots=True)
class CellCoordinates:
    """One bounded-cut cell: which family, which token length, which load."""

    family: str
    token_length: int
    load: int
    data_seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.family}@{self.token_length}L{self.load}s{self.data_seed}"


@dataclass(frozen=True, slots=True)
class CellMeasurement:
    """What one cell actually produced, as opposed to what it declared."""

    cell_id: str
    family: str
    declared_token_length: int
    measured_token_count: int
    length_kind: str
    load: int
    data_seed: int
    instances: int
    prompt_sha256: str
    gold_sha256: str

    @property
    def exact(self) -> bool:
        """Whether the fixture is what it says it is.

        The whole artifact is indexed by this property, so it is a property
        rather than a convention.
        """
        return self.measured_token_count == self.declared_token_length


def cell_count() -> int:
    """Cells in the bounded cut: families x lengths x loads x seeds."""
    return (len(BOUNDED_FAMILIES) * len(BOUNDED_LENGTHS) * len(BOUNDED_LOADS)
            * len(BOUNDED_DATA_SEEDS))


def cell_coordinates(unit: int) -> CellCoordinates:
    """Map a flat unit index onto the bounded cut's coordinates.

    Raising rather than clamping: a unit index past the end would otherwise
    silently alias onto a real cell and make the artifact claim coverage it does
    not have.
    """
    total = cell_count()
    if not 0 <= unit < total:
        raise MatrixRefusal(f"unit {unit} is outside the {total}-cell bounded cut")
    seed_span = len(BOUNDED_DATA_SEEDS)
    load_span = len(BOUNDED_LOADS) * seed_span
    length_span = len(BOUNDED_LENGTHS) * load_span
    family_index, remainder = divmod(unit, length_span)
    length_index, remainder = divmod(remainder, load_span)
    load_index, seed_index = divmod(remainder, seed_span)
    return CellCoordinates(
        family=BOUNDED_FAMILIES[family_index],
        token_length=BOUNDED_LENGTHS[length_index],
        load=BOUNDED_LOADS[load_index],
        data_seed=BOUNDED_DATA_SEEDS[seed_index],
    )


def coordinate_units(coordinates: CellCoordinates) -> int:
    """The inverse of :func:`cell_coordinates`, so the two cannot drift."""
    try:
        family_index = BOUNDED_FAMILIES.index(coordinates.family)
        length_index = BOUNDED_LENGTHS.index(coordinates.token_length)
        load_index = BOUNDED_LOADS.index(coordinates.load)
        seed_index = BOUNDED_DATA_SEEDS.index(coordinates.data_seed)
    except ValueError as exc:
        raise MatrixRefusal(f"{coordinates} is not on the bounded cut") from exc
    seed_span = len(BOUNDED_DATA_SEEDS)
    return ((family_index * len(BOUNDED_LENGTHS) + length_index) * len(BOUNDED_LOADS)
            + load_index) * seed_span + seed_index


def measure_cell(coordinates: CellCoordinates, encode: SafeEncoder, *,
                 instances: int = BOUNDED_INSTANCES) -> CellMeasurement:
    """Build every instance in a cell and record what the fixture actually is.

    The prompt hash is over the concatenated instance prompt ids, so the artifact
    commits to the exact fixtures a run consumed rather than to a recipe that
    could produce different ones later.
    """
    if instances <= 0:
        raise MatrixRefusal(f"instances must be positive, got {instances}")
    if coordinates.token_length not in TOKEN_LENGTHS:
        raise MatrixRefusal(
            f"{coordinates.token_length} is not a registered token length "
            f"{list(TOKEN_LENGTHS)}")
    digest = hashlib.sha256()
    gold = hashlib.sha256()
    measured: set[int] = set()
    for index in range(instances):
        request = TokenTargetedRequest(
            family=coordinates.family, data_seed=coordinates.data_seed,
            token_length=coordinates.token_length,
            position_fraction=BOUNDED_POSITION, load=coordinates.load,
            distractor=BOUNDED_DISTRACTOR, instance_index=index)
        try:
            task = generate_token_targeted_task(encode, request)
        except TokenTargetedError as exc:
            raise MatrixRefusal(f"{coordinates.cell_id} instance {index}: {exc}") from exc
        measured.add(task.n_prompt_tokens)
        digest.update(",".join(str(token) for token in task.prompt_ids).encode("ascii"))
        digest.update(b"|")
        gold.update(",".join(str(token) for token in task.answer_ids).encode("ascii"))
        gold.update(b"|")
    if len(measured) != 1:
        # A CROSS-CHECK, not a live guard, and labelled as one so it is not
        # mistaken for evidence of coverage.  build_token_prompt pads to the
        # declared target by construction, so this cannot fire while that holds;
        # it fires only if that construction changes -- which is exactly when a
        # silent length drift would otherwise reach the artifact.  The live guard
        # is `exact` and the refusal in artifact_bytes, which catch a
        # measurement that disagrees with its own label.
        raise MatrixRefusal(
            f"{coordinates.cell_id} produced {len(measured)} distinct token counts "
            f"{sorted(measured)}; a token-targeted cell must be exact, and a cell "
            f"whose instances differ in length is not a cell")
    return CellMeasurement(
        cell_id=coordinates.cell_id, family=coordinates.family,
        declared_token_length=coordinates.token_length,
        measured_token_count=measured.pop(), length_kind=LENGTH_KIND,
        load=coordinates.load, data_seed=coordinates.data_seed,
        instances=instances, prompt_sha256=digest.hexdigest(), gold_sha256=gold.hexdigest())


def artifact_bytes(measurements: Sequence[CellMeasurement], *,
                   source_hashes: dict[str, str] | None = None) -> bytes:
    """Serialise the artifact, refusing one with an inexact cell."""
    inexact = [m.cell_id for m in measurements if not m.exact]
    if inexact:
        raise MatrixRefusal(
            f"{len(inexact)} cell(s) do not measure their declared token length: "
            f"{inexact[:5]}. Refusing to write an artifact whose labels are not its "
            f"units -- that is precisely the defect this grid exists to fix.")
    document = {
        "schema_version": SCHEMA_VERSION,
        "matrix_name": MATRIX_NAME,
        "length_kind": LENGTH_KIND,
        "scope": (
            "the BOUNDED cut: 3 families x 3 token lengths x 4 loads x 1 data seed, "
            "50 instances per cell. Not the full registered grid (720 cells x 200 "
            "instances x 5 seeds), and not comparable to length_matrix.json, which "
            "is declared in CHARACTERS."),
        "why_separate_from_length_matrix": (
            "length_matrix.json is hash-bound into the preregistration and its "
            "'65536' cells are 65,536 CHARACTERS, i.e. 9K-17K RWKV tokens. A second "
            "artifact records the token-exact grid without breaking those hashes or "
            "hiding the discrepancy."),
        "declared_grid": {
            "families": list(BOUNDED_FAMILIES),
            "token_lengths": list(BOUNDED_LENGTHS),
            "loads": list(BOUNDED_LOADS),
            "data_seeds": list(BOUNDED_DATA_SEEDS),
            "instances_per_cell": BOUNDED_INSTANCES,
            "position_fraction": BOUNDED_POSITION,
            "distractor": BOUNDED_DISTRACTOR,
            "excluded_families": {"finite_hmm": "answer is a parsed state sequence, not a value"},
        },
        "source_hashes": source_hashes or {},
        "cells": [asdict(m) for m in measurements],
    }
    return (json.dumps(document, indent=2, sort_keys=False) + "\n").encode("utf-8")


def verify_matrix_artifact(path: Path, *, sample: int | None = None,
                           encode: SafeEncoder | None = None) -> bool:
    """Re-derive cells and check the artifact still describes them.

    ``sample`` re-measures the first N cells; ``None`` checks every cell's
    internal consistency without rebuilding (which needs no tokenizer).
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("length_kind") != LENGTH_KIND:
        raise MatrixRefusal(
            f"artifact declares length_kind {document.get('length_kind')!r}, not {LENGTH_KIND!r}")
    cells = document.get("cells")
    if not isinstance(cells, list) or not cells:
        raise MatrixRefusal("artifact carries no cells")
    if len(cells) != cell_count():
        raise MatrixRefusal(
            f"artifact carries {len(cells)} cells but the bounded cut has {cell_count()}")
    for entry in cells:
        if entry["measured_token_count"] != entry["declared_token_length"]:
            raise MatrixRefusal(
                f"{entry['cell_id']}: measured {entry['measured_token_count']} tokens "
                f"against a declared {entry['declared_token_length']}")
        if entry["family"] not in BOUNDED_FAMILIES:
            raise MatrixRefusal(f"{entry['cell_id']}: family outside the bounded cut")
    if sample is not None:
        if encode is None:
            raise MatrixRefusal("re-measuring a sample needs an encoder")
        for entry in cells[:sample]:
            again = measure_cell(
                CellCoordinates(family=entry["family"], token_length=entry["declared_token_length"],
                                load=entry["load"], data_seed=entry["data_seed"]),
                encode, instances=entry["instances"])
            if again.prompt_sha256 != entry["prompt_sha256"]:
                raise MatrixRefusal(
                    f"{entry['cell_id']}: fixtures changed since the artifact was written "
                    f"({again.prompt_sha256[:12]} != {entry['prompt_sha256'][:12]})")
    return True


def write_matrix(measurements: Sequence[CellMeasurement], output: Path, *,
                 source_hashes: dict[str, str] | None = None) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(artifact_bytes(measurements, source_hashes=source_hashes))
    return output


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI shell
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--instances", type=int, default=BOUNDED_INSTANCES)
    args = parser.parse_args(argv)
    raise MatrixRefusal(
        "building the artifact needs the qualified tokenizer; run it in the "
        "qualification lane with the receipt in hand rather than from this shell")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
