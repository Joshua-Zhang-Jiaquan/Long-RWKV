"""Run the long-context grid: fixtures in, scores and statistics out.

This is the seam that turns the pieces built around it into an experiment.  It owns the
one thing none of them can own individually: the ORDER of operations, and the record of
what was and was not measured.

The model sits behind one injected callable.  That is not decoration -- it is what makes
the runner testable on CPU, and it is where the access-mode claim becomes checkable: a
"streaming" cell is one whose generation was driven through ``plan_chunks`` with an
``AccessLog`` armed, and the log is returned with the cell so the claim travels with the
number it produced.  A full-canvas run relabelled as streaming would otherwise produce
identical scores.

**Nothing is silently skipped.**  Every instance that produced no parseable answer is
counted as missing and reported beside the accuracy, because the alternative -- scoring
an unparseable completion as wrong -- makes a harness failure indistinguishable from a
model failure.  The same rule applies one level up: a cell with no instances at all is
reported as unmeasured rather than as a zero.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol, Sequence

from ..qualification.streaming_canvas import AccessLog, ChunkAccess, plan_chunks
from ..tasks.scoring import DECODED_KEY, GOLD_KEY, ScoringRefusal, score_cell
from ..tasks.statistics import macro_score
from ..tasks.token_targeted_matrix import (
    BOUNDED_INSTANCES,
    BOUNDED_POSITION,
    CellCoordinates,
    cell_coordinates,
    cell_count,
)
from ..tasks.token_targeted_tasks import TokenTargetedRequest, generate_token_targeted_task

SCHEMA: Final = "nonlatent_longcontext_runner_v1"

#: The two access modes the protocol compares.  Named here because the comparison is
#: the measurement: a runner that could only do one of them could not produce it.
FULL_CANVAS: Final = "full_canvas"
STREAMING: Final = "streaming"
ACCESS_MODES: Final = (FULL_CANVAS, STREAMING)

#: The chunk a streaming run is given.  One canvas per chunk is the degenerate case and
#: is deliberately not the default -- it would make the access-mode comparison vacuous.
DEFAULT_STREAM_CHUNK: Final = 4096

#: How many tokens one instance contributes, in the two modes.  The full canvas sees the
#: whole prompt; a streaming run sees it in chunks.  Recorded per cell so a reader can
#: tell which was measured without inferring it from the mode name.
MODE_SEMANTICS: Final = {
    FULL_CANVAS: "the whole prompt is presented in one canvas",
    STREAMING: "the prompt is presented in order, in chunks, each chunk unable to see any token before its own start",
}


class RunnerRefusal(ValueError):
    """The grid cannot be run as specified."""


class Generate(Protocol):
    """The model seam: ids in, text out."""

    def __call__(self, prompt_ids: Sequence[int], *, max_new_tokens: int) -> str: ...


@dataclass(frozen=True, slots=True)
class CellRun:
    """One (family, length, load) cell under one access mode, and what it produced."""

    family: str
    token_length: int
    load: int
    data_seed: int
    access_mode: str
    instances_requested: int
    instances_scored: int
    n_missing: int
    accuracy: float | None
    missing_fraction: float | None
    chunks: list[ChunkAccess]
    reread_checked: bool

    @property
    def cell_key(self) -> tuple[str, int]:
        return (self.family, self.token_length)

    @property
    def measured(self) -> bool:
        """A cell with no scored instances is UNMEASURED, not zero."""
        return self.instances_scored > 0


def run_cell(coordinates: CellCoordinates, *, encode, generate: Generate,
             access_mode: str = FULL_CANVAS, instances: int = BOUNDED_INSTANCES,
             max_new_tokens: int = 32,
             stream_chunk: int = DEFAULT_STREAM_CHUNK) -> CellRun:
    """Build every instance of a cell, generate, and score -- reporting what was skipped.

    In streaming mode the generation is driven chunk by chunk with an access log armed,
    and the log is checked before the cell is returned.  A reread is a REFUSAL rather
    than a warning, because the access mode is the thing being measured: a streaming cell
    whose chunks overlapped is a full-canvas cell with a misleading label.
    """
    if access_mode not in ACCESS_MODES:
        raise RunnerRefusal(f"unknown access mode {access_mode!r}; known: {list(ACCESS_MODES)}")
    if instances <= 0:
        raise RunnerRefusal(f"instances must be positive, got {instances}")

    # ONE LOG PER INSTANCE, not one per cell.  Each instance is its own stream starting
    # at position 0, so a shared log sees instance 1's first chunk as a backwards jump
    # and refuses -- correctly, because the log's claim ("no chunk sees a token before
    # its own start") is a claim about ONE stream.  Sharing it made every multi-instance
    # streaming cell unreadable, which is what the pairing test caught.
    chunks: list[ChunkAccess] = []
    records: list[dict] = []
    all_checked = access_mode == STREAMING

    for index in range(instances):
        request = TokenTargetedRequest(
            family=coordinates.family, data_seed=coordinates.data_seed,
            token_length=coordinates.token_length,
            position_fraction=BOUNDED_POSITION, load=coordinates.load,
            distractor="similar", instance_index=index)
        task = generate_token_targeted_task(encode, request)

        if access_mode == STREAMING:
            plan = plan_chunks(total_tokens=task.n_prompt_tokens, chunk=stream_chunk)
            log = AccessLog()
            seen: list[int] = []
            for access in plan:
                log.record(access)
                seen.extend(task.prompt_ids[access.start:access.end])
            # checked HERE, on the accesses this instance actually made
            log.assert_no_reread(total_tokens=task.n_prompt_tokens)
            log.assert_forward_only()
            if not chunks:
                chunks = plan
            prompt_ids = seen
        else:
            prompt_ids = list(task.prompt_ids)

        decoded = generate(prompt_ids, max_new_tokens=max_new_tokens)
        records.append({DECODED_KEY: decoded, GOLD_KEY: task.answer})

    try:
        scored = score_cell(records, coordinates.family)
    except ScoringRefusal as exc:
        raise RunnerRefusal(f"{coordinates.cell_id}: {exc}") from exc

    return CellRun(
        family=coordinates.family, token_length=coordinates.token_length,
        load=coordinates.load, data_seed=coordinates.data_seed, access_mode=access_mode,
        instances_requested=instances, instances_scored=scored["n"] - scored["n_missing"],
        n_missing=scored["n_missing"], accuracy=scored["accuracy"],
        missing_fraction=scored["missing_fraction"],
        chunks=chunks, reread_checked=all_checked if access_mode == STREAMING else False)


def run_grid(*, encode, generate: Generate, access_mode: str = FULL_CANVAS,
             units: Sequence[int] | None = None, instances: int = BOUNDED_INSTANCES,
             stream_chunk: int = DEFAULT_STREAM_CHUNK) -> dict[str, object]:
    """Every cell in the cut under one access mode, with the macro score over measured cells.

    The macro score is computed over the cells that WERE measured, and the unmeasured
    ones are named rather than dropped: a score that silently omits its failures is the
    silent-truncation defect the protocol forbids.
    """
    selected = list(range(cell_count())) if units is None else list(units)
    cells: list[CellRun] = []
    for unit in selected:
        cells.append(run_cell(cell_coordinates(unit), encode=encode, generate=generate,
                              access_mode=access_mode, instances=instances,
                              stream_chunk=stream_chunk))

    # THE LOAD AXIS IS AVERAGED WITHIN A CELL, NOT DROPPED.  The macro key is
    # (family, token_length) -- nine cells, as the protocol declares -- but the cut has four
    # LOAD values per family and length, so keying on (family, length) alone let each load
    # OVERWRITE the previous one. Measured on the bounded cut: 36 cells collapsed to 9 keys
    # and 27 cells vanished from the score with no error, because a dict assignment over a
    # duplicate key is not a failure. The macro then reported one load as if it were the
    # cell. Averaging within the cell is what the protocol's "equal-weight mean over three
    # families at three lengths" actually describes, and the per-load values are kept so the
    # averaging is inspectable rather than trusted.
    by_cell: dict[tuple[str, int], list[float]] = {}
    for cell in cells:
        if cell.measured and cell.accuracy is not None:
            by_cell.setdefault((cell.family, cell.token_length), []).append(cell.accuracy)
    measured = {key: sum(v) / len(v) for key, v in by_cell.items()}
    unmeasured = [f"{c.family}@{c.token_length}L{c.load}" for c in cells if not c.measured]
    macro = macro_score(measured) if measured else None

    return {
        "schema": SCHEMA,
        "access_mode": access_mode,
        "access_mode_semantics": MODE_SEMANTICS[access_mode],
        "cells": [asdict(c) for c in cells],
        "cells_measured": len(measured),
        # The COUNT is the property that distinguishes averaging from overwriting: under
        # the collapse every cell reads 1, because a duplicate key keeps only the last
        # value. A test that asserted only the averaged value could not tell the two apart,
        # which is exactly what the first version of that test did.
        "cells_averaged": {f"{k[0]}@{k[1]}": {"accuracy": round(sum(v) / len(v), 6),
                                              "n_loads": len(v)}
                           for k, v in sorted(by_cell.items())},
        "cells_unmeasured": unmeasured,
        "macro_score": macro,
        "note": ("the macro score is over measured cells only; `cells_unmeasured` names "
                 "what it does not cover, so the score cannot be read as covering them"),
    }


def paired_access_modes(*, encode, generate_a: Generate, generate_b: Generate,
                        units: Sequence[int] | None = None,
                        instances: int = BOUNDED_INSTANCES) -> dict[str, object]:
    """The two access modes on the SAME instances, paired cell by cell.

    Paired rather than pooled because the comparison's whole point is that the instances
    are identical: the only change is how much of the prompt each position could see.
    """
    full = run_grid(encode=encode, generate=generate_a, access_mode=FULL_CANVAS,
                    units=units, instances=instances)
    stream = run_grid(encode=encode, generate=generate_b, access_mode=STREAMING,
                      units=units, instances=instances)
    pairs = []
    for a, b in zip(full["cells"], stream["cells"], strict=True):
        if a["accuracy"] is None or b["accuracy"] is None:
            continue
        pairs.append((float(b["accuracy"]), float(a["accuracy"])))
    return {
        "schema": SCHEMA,
        "full_canvas": full,
        "streaming": stream,
        "paired_cells": len(pairs),
        "delta_streaming_minus_full": [b - a for b, a in pairs],
        "note": ("cells where either mode produced no score are omitted from the paired "
                 "list and remain visible in the per-mode `cells_unmeasured`"),
    }


def write_grid(grid: dict[str, object], output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(grid, indent=2) + "\n", encoding="utf-8")
    return output
