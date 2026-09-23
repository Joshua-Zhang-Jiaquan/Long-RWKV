"""The E3 lane: the paper's 324-cell long-context grid wired to the LC aggregate.

Three vocabularies name the same three task constructions, and this module is where
they are reconciled **once**, by composition rather than by a fourth hard-coded table:

===================  ==========================  ==================
paper CSV            pinned generator            eval runner
===================  ==========================  ==================
associative_recall   associative_recall          recall
overwrite_delayed_query  latest_write            overwrite
code_dataflow        pointer_dataflow            dataflow
===================  ==========================  ==================

The left-to-middle column is :data:`lrwkv_evidence.e3.grid324.FAMILY_MAP` (the grid's
own record of the paper's spelling); the middle-to-right column is the generator's
:data:`longrwkv.tasks.registry.FAMILY_MAP`, which is *its own* declaration and is not
restated here.  :func:`check_family_chain` composes the two and proves the result is a
bijection onto the runner's families -- so a future generator bump that renamed
``latest_write`` fails a test instead of silently reporting three families as two.

Why this matters more than it looks: the paper's aggregate is

    LC_N(a) = (1/3) * sum_{f in {recall, overwrite, dataflow}} s_{a,f,N}
    LC(a)   = (1/9) * sum_f sum_{N in {16K,32K,64K}} s_{a,f,N}

so a cell whose family is spelled with the generator's vocabulary reaches
``longcontext_runner.CellSpec`` and raises ``Refusal("unknown family ...")`` -- a cell
that is *unsupported* by accident.  A cell quietly filed under the wrong family is
worse: it is averaged into the wrong third of ``LC_N`` and the total still looks like a
number.

The decoder operating point is deliberately **not** defaulted here.  The project plan
(Q2) requires the sampler operating point -- steps x tau x commit rule -- to be chosen on
a dev-only grid and frozen into ``protocol.json["quality"]`` *before* any confirmation
run, and :func:`require_frozen_decoder` is the refusal that enforces it at the point of
use.  A default would make the first un-pinned E3 run look exactly like a pinned one.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Sequence

from lrwkv_evidence.e3 import grid324 as G


def _ensure_generator_path() -> str:
    """Put ``GENERATOR_ROOT`` on ``sys.path`` and return it.

    Shared by every importer here rather than only by :func:`_runner_module`.  Measured
    2026-09-21: :func:`runner_distractor` imported ``longrwkv.tasks`` directly, so it
    worked only when the runner happened to have been imported first -- a call-ordering
    dependency that a test running it first turned into ``ModuleNotFoundError``.
    """
    root = str(G.GENERATOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _runner_module():
    """``longrwkv.eval.longcontext_runner``, or a refusal naming the path.

    Imported rather than re-implemented: the LC aggregate, the access-mode
    qualification rules and the item/coverage accounting are already written and
    frozen there, and a second copy would be a second definition of ``LC``.

    ``GENERATOR_ROOT`` is the directory *containing* the ``longrwkv`` package (that is
    how :func:`lrwkv_evidence.e3.grid324._import_generator` uses it), so it goes on the
    path as-is -- ``.parent`` would put ``$G`` there and the import would resolve to
    something else entirely.
    """
    root = _ensure_generator_path()
    try:
        from longrwkv.eval import longcontext_runner as runner
    except Exception as exc:  # noqa: BLE001 - the message is the deliverable
        raise SystemExit(
            f"cannot import longrwkv.eval.longcontext_runner from {root}: "
            f"{type(exc).__name__}: {exc}. The E3 lane aggregates through that "
            f"module's frozen LC definition; without it there is no LC to report.")
    return runner


#: The generator's own metric id, taken from the banked grid records rather than
#: chosen here.  Every instance of every cell carries it, and the loader below refuses
#: a cell whose instances declare anything else -- a metric id is a claim about what
#: the number means, and the grid is the authority on which one it recorded.
METRIC_ID: Final = "final_answer_exact_match"

#: The paper's confirmation grid is a full-canvas panel: the model gets the whole
#: 16K/32K/64K canvas.  ``state_only`` and ``retrieval_assisted`` are the other two
#: evidence sets and each needs a qualification record before it may be measured, so
#: they are never selected by default.
DEFAULT_ACCESS_MODE: Final = "full_canvas"


def runner_family(paper_family: str) -> str:
    """Paper-CSV family -> the eval runner's family, through the generator's map."""
    return check_family_chain()[paper_family]


def check_family_chain() -> dict[str, str]:
    """The composed paper -> runner map, with the generator's map checked for shape.

    Raises rather than returning a partial map: a missing or non-injective link means
    some cell cannot be filed, and filing it under a default would put it in the wrong
    third of ``LC_N``.
    """
    generator_side = G.FAMILY_MAP                    # paper -> generator
    runner = _runner_module()
    from longrwkv.tasks import registry as _registry  # noqa: PLC0415
    runner_side = _registry.FAMILY_MAP               # runner -> generator

    # Invert runner -> generator.  A generator family reachable from two runner
    # families would make the composition ambiguous, which is the one way this can
    # silently mis-file a cell.
    by_generator: dict[str, list[str]] = {}
    for runner_family_name, generator_family in runner_side.items():
        by_generator.setdefault(generator_family, []).append(runner_family_name)
    ambiguous = {g: names for g, names in by_generator.items() if len(names) > 1}
    if ambiguous:
        raise SystemExit(
            f"the generator's FAMILY_MAP is not injective on the generator side: "
            f"{ambiguous}. A paper family would compose to two runner families.")

    composed: dict[str, str] = {}
    for paper_family, generator_family in generator_side.items():
        names = by_generator.get(generator_family)
        if not names:
            raise SystemExit(
                f"the paper family {paper_family!r} maps to the generator family "
                f"{generator_family!r}, which the generator's own registry.FAMILY_MAP "
                f"does not know ({sorted(runner_side)}). The grid and the generator "
                f"disagree about which construction this is.")
        composed[paper_family] = names[0]

    unknown = sorted(set(composed.values()) - set(runner.FAMILIES))
    if unknown:
        raise SystemExit(
            f"composed runner families {unknown} are not in "
            f"longcontext_runner.FAMILIES={list(runner.FAMILIES)}; their cells could "
            f"not enter the LC aggregate.")
    missing = sorted(set(runner.FAMILIES) - set(composed.values()))
    if missing:
        raise SystemExit(
            f"the runner declares families {missing} that no paper family composes "
            f"to, so LC_N would average over fewer than three families while still "
            f"reading as a mean of three.")
    return composed


def runner_distractor(grid_distractor: str) -> str:
    """Grid distractor name -> the runner's, through the generator's own map.

    The generator's ``DISTRACTOR_MAP`` is many-to-one (``absent`` and ``none`` both
    name the generator's ``none``), so the exact key is preferred when it exists and
    the alphabetically first synonym is taken otherwise -- deterministic either way,
    and the ambiguity is in the generator's table, not invented here.
    """
    _ensure_generator_path()
    from longrwkv.tasks import registry as _registry  # noqa: PLC0415

    table = _registry.DISTRACTOR_MAP
    if grid_distractor in table:
        return grid_distractor
    synonyms = sorted(k for k, v in table.items() if v == grid_distractor)
    if not synonyms:
        raise SystemExit(
            f"distractor {grid_distractor!r} is not in the generator's "
            f"DISTRACTOR_MAP image {sorted(set(table.values()))}")
    return synonyms[0]


@dataclass(frozen=True)
class GridInstance:
    """One grid instance, with the fields an E3 item needs to be re-derived."""

    cell_id: str
    data_seed: int
    instance_index: int
    generator_seed: int
    input_ids_sha256: str
    realized_tokens: int
    queries: tuple[str, ...]
    answers: tuple[str, ...]
    metric: str
    target_span: tuple[int, int] | None


def load_cell_record(path: Path) -> dict:
    """Read one cell JSON and refuse a cell that cannot enter the panel.

    Refuses an ``unsupported`` cell (the grid records 60 of them: 6 over-canvas and 54
    infeasible) and a mixed-metric cell.  An unsupported cell is reported as
    ``unsupported`` by the collector; it must never be silently skipped, because a
    panel of 264 cells labelled as the paper's 324 is the same number with a different
    meaning.
    """
    import json

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("status") != "ok":
        raise SystemExit(
            f"{record.get('cell_id')}: status={record.get('status')!r} "
            f"({record.get('unsupported_reason')}). Report it as unsupported at the "
            f"paper's declared cell count; do not drop it from the panel.")
    metrics = {inst["metric"] for inst in record["instances"]}
    if metrics != {METRIC_ID}:
        raise SystemExit(
            f"{record['cell_id']}: instances declare metrics {sorted(metrics)}, not "
            f"the grid's {METRIC_ID!r}. Two metrics in one cell would be averaged "
            f"into one score.")
    return record


def instances_of(record: dict) -> list[GridInstance]:
    out = []
    for inst in record["instances"]:
        span = inst.get("target_span")
        out.append(GridInstance(
            cell_id=inst["cell_id"], data_seed=int(inst["data_seed"]),
            instance_index=int(inst["instance_index"]),
            generator_seed=int(inst["generator_seed"]),
            input_ids_sha256=inst["input_ids_sha256"],
            realized_tokens=int(inst["realized_tokens"]),
            queries=tuple(inst["queries"]), answers=tuple(inst["answers"]),
            metric=inst["metric"],
            target_span=tuple(span) if span else None))
    return out


def cell_specs(records: Sequence[dict], *, arm: str,
               access_mode: str = DEFAULT_ACCESS_MODE):
    """The paper's cells as runner ``CellSpec``s, one per grid cell.

    ``arm`` is the checkpoint's paper arm, not its track id: the runner keys its
    ``ScoreTable`` on ``(arm, family, length, access_mode)``, and two tracks of the
    same arm would otherwise collapse into one row.
    """
    runner_family_of = check_family_chain()
    CellSpec = _runner_module().CellSpec
    specs = []
    for record in records:
        family = runner_family_of[record["family"]]
        spans = [i["target_span"] for i in record["instances"] if i.get("target_span")]
        specs.append(CellSpec(
            arm=arm, family=family, length=int(record["history_length"]),
            access_mode=access_mode,
            hops=G.DEPTH, binding_load=int(record["binding_load"]),
            evidence_fraction=float(record["evidence_position_fraction"]),
            distractor=runner_distractor(record["distractors"]),
            metric_id=METRIC_ID,
            output_capacity_tokens=max((int(s[1] - s[0]) for s in spans), default=0),
        ))
    return specs


#: The keys a frozen decoder operating point must carry.  Named here so the refusal
#: below can say exactly what is missing rather than "the protocol is incomplete".
DECODER_KEYS: Final = ("sampler", "steps", "tau", "commit_order")


def require_frozen_decoder(protocol: dict | None) -> dict:
    """The frozen sampler operating point, or a refusal naming what is unpinned.

    The plan's Q2 requires the operating point to be selected on a **dev-only** grid and
    frozen before a confirmation run, and the underlying reason is the same one that
    made the T=0 batching gate blind: a number produced at an operating point nobody
    declared cannot be compared to a number produced at a different one, and the two
    render identically in a table.

    Deliberately a refusal rather than a default. Measured 2026-09-21: the grid, the
    family chain and the emitter are all buildable, and it is tempting to run E3 at some
    plausible NFE to "have the numbers" -- which would produce exactly the artifact this
    function exists to prevent.
    """
    quality = (protocol or {}).get("quality") or {}
    decoder = quality.get("decoder") or {}
    missing = [k for k in DECODER_KEYS if decoder.get(k) in (None, "")]
    if missing:
        raise SystemExit(
            f"the E3 decoder operating point is unpinned: {missing} are unset in "
            f"protocol['quality']['decoder']. Run the dev-only grid "
            f"(lrwkv_evidence.e3.dev_grid) and freeze {list(DECODER_KEYS)} before any "
            f"confirmation cell is scored. The 324-cell grid is the confirmation set; "
            f"choosing the sampler on it would be choosing it on the test data.")
    return dict(decoder)


def lc_aggregate(records: Iterable, *, arm: str,
                 access_mode: str = DEFAULT_ACCESS_MODE) -> dict:
    """``LC_N`` and ``LC`` for one arm, through the runner's frozen definition."""
    runner = _runner_module()
    rows = list(records)
    per_length = {
        int(length): runner.aggregate_lc_n(rows, arm=arm, length=int(length),
                                           access_mode=access_mode)
        for length in runner.PRIMARY_LENGTHS}
    return {
        "arm": arm,
        "access_mode": access_mode,
        "lc_n": per_length,
        "lc": runner.aggregate_lc(rows, arm=arm, access_mode=access_mode),
        "primary_lengths": list(runner.PRIMARY_LENGTHS),
        "anchor_lengths": list(runner.ANCHOR_LENGTHS),
    }
