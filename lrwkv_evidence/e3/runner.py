"""E3 scoring: re-derive each banked prompt in-pod, score it, emit ItemOutcomes.

The grid banks an ``input_ids_sha256`` per instance instead of the token ids, on the
premise that a *different* process -- this one -- rebuilds the prompt from
``(cell, seed, index)`` and the hash proves it rebuilt the same prompt.  That premise is
only worth anything if the rebuild actually happens and a mismatch actually stops the
run, so :func:`require_rederived` is the first thing every scored item goes through.  An
evaluator that trusted the record would report a number for prompts nobody checked, and
the number would look exactly like a checked one.

Two deliberate non-choices:

* **The answer grammar is imported, not restated.**  ``FAMILY_GRAMMAR``,
  ``METRIC`` and ``ANSWER_SEPARATOR`` come from the pinned generator, so a generator bump
  that changes how an answer is written changes this too.  A local copy of
  ``r"v\\d{4}"`` would keep matching after the generator stopped emitting that shape --
  the failure mode where a guard paraphrases the thing it guards and then fails
  permissively on exactly the inputs it was written for.
* **The decoder is a parameter.**  :data:`Decoder` is the only model-facing surface, so
  everything in this module is exercisable on CPU with a stub, and the GPU seam is one
  named function rather than a scatter of ``torch`` calls.

``correct`` and ``joint_exact`` are carried separately because the generator's metric
name covers both.  On the paper's 324-cell grid they coincide by construction --
``N_QUERIES = 1`` -- and :func:`require_single_query` asserts that rather than assuming
it, so widening the grid fails here instead of silently averaging two statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from lrwkv_evidence.e3 import grid324 as G
from lrwkv_evidence.e3 import lane as L


def metric_id() -> str:
    """The generator's own metric name, read from the generator rather than restated."""
    hop, _, _ = G._import_generator()
    return str(hop.METRIC)


@dataclass(frozen=True)
class Decode:
    """One decoder result: the text it produced and what it cost.

    ``calls`` is the actual number of network evaluations.  The protocol requires the
    real call count rather than the nominal schedule, because a schedule of 16 steps
    that stops early and one that runs to completion are different work, and only the
    second is what "16 steps" is supposed to mean.
    """

    text: str
    calls: int = 0


class Decoder(Protocol):
    """The model seam: a masked canvas and the span to fill -> text.

    ``input_ids`` is the canvas with ``target_span`` already replaced by the mask
    token, so the decoder never receives the answer it is being asked for.
    """

    def __call__(self, input_ids: Sequence[int],
                 target_span: tuple[int, int]) -> Decode: ...


def extract_answer(text: str, family: str) -> str | None:
    """The first answer the family's grammar matches, or ``None``.

    ``None`` is not ``""``: "the model wrote something that is not an answer of this
    family" and "the model wrote nothing" are different outcomes, and the protocol
    wants the unparseable ones counted rather than folded into wrong answers.
    """
    grammar = _grammar(family)
    match = grammar.search(text)
    return match.group(0) if match else None


def _grammar(family: str):
    """The pinned generator's grammar for a family, refusing an unknown one."""
    hop, _, _ = G._import_generator()
    table = hop.FAMILY_GRAMMAR
    if family not in table:
        raise SystemExit(
            f"family {family!r} has no answer grammar in the pinned generator "
            f"({sorted(table)}); a guessed grammar would extract something the "
            f"generator never emits.")
    return table[family]


def require_single_query(family: str) -> None:
    """Assert the grid's one-query shape, which is what makes correct == joint_exact."""
    if G.N_QUERIES != 1:
        raise SystemExit(
            f"the runner is written for N_QUERIES=1 but the grid declares "
            f"{G.N_QUERIES}. With more than one query, 'correct' and 'joint_exact' are "
            f"different statistics and both must be reported; this runner would "
            f"collapse them.")


def require_rederived(hop_tasks, token_prompt, encoder, cell, data_seed: int,
                      instance_index: int, banked_sha: str, pools=None):
    """Rebuild one instance and prove it is the one that was banked.

    Raises on a mismatch rather than scoring it: a prompt that does not re-derive means
    the banked hash describes a prompt this process cannot produce, and any score
    computed here would belong to a different item than the record names.
    """
    task, canvas, reason = G.build_instance(hop_tasks, token_prompt, encoder, cell,
                                            data_seed, instance_index, pools, G.GRID_SPLIT)
    if reason is not None:
        raise ValueError(
            f"{cell.cell_id} seed={data_seed} i={instance_index}: the cell is recorded "
            f"as built but now refuses ({reason}). The record and the generator "
            f"disagree about this cell.")
    got = G.ids_sha256(canvas.input_ids)
    if got != banked_sha:
        raise ValueError(
            f"{cell.cell_id} seed={data_seed} i={instance_index}: the prompt re-derived "
            f"to {got[:16]}... but the record banks {banked_sha[:16]}... The banked "
            f"hash is the only thing tying a recorded prompt to a rebuildable one, so "
            f"scoring this instance would attach a score to an unverified prompt.")
    return task, canvas


def score_instance(task, canvas, instance: L.GridInstance, decode: Decoder,
                   mask_id: int):
    """One item: mask the target span, ask the decoder, compare to the gold.

    Returns a runner-module ``ItemOutcome`` so the LC aggregate can consume it
    unchanged.  ``parse_ok`` is False when the decoder produced nothing the family's
    grammar recognises, which is a *measured* outcome and not a dropped item.

    ``mask_id`` is a parameter rather than a module import: this module has to be
    importable on a CPU host with no model, and the mask constant lives in the model
    package.  Passing it also makes the "which token did you mask with" question
    answerable at the call site, where a wrong value would otherwise be invisible.
    """
    family = task.request.family
    require_single_query(family)
    span = instance.target_span
    if not span:
        raise ValueError(
            f"{instance.cell_id} seed={instance.data_seed} i={instance.instance_index}: "
            f"no target span, so there is nothing to mask and nothing to score.")
    lo, hi = int(span[0]), int(span[1])
    ids = list(canvas.input_ids)
    masked = ids[:lo] + [int(mask_id)] * (hi - lo) + ids[hi:]
    decoded = decode(masked, (lo, hi))
    got = extract_answer(decoded.text, family)
    gold = tuple(instance.answers)
    correct = got is not None and len(gold) == 1 and got == gold[0]
    runner = L._runner_module()
    return runner.ItemOutcome(
        item_id=f"{instance.cell_id}:{instance.data_seed}:{instance.instance_index}",
        correct=bool(correct),
        # One query per instance on this grid, asserted above; the two statistics are
        # carried separately so a widened grid cannot silently conflate them.
        joint_exact=bool(correct),
        parse_ok=got is not None,
        supported=True,
        answer=decoded.text,
        cluster=instance.cell_id,
    )


def run_cell_record(record: dict, decode: Decoder, *, mask_id: int,
                    pools=None):
    """Every instance of one cell, in the runner's own record shape.

    Cell-level refusals propagate: a cell that cannot be re-derived must not become a
    partial row that still looks complete.
    """
    hop, token_prompt, registry = G._import_generator()
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    family = record["family"]
    cell = G.Cell(family, int(record["history_length"]),
                  int(round(100 * float(record["evidence_position_fraction"]))),
                  int(record["binding_load"]), record["distractors"])
    outcomes = []
    for instance in L.instances_of(record):
        if instance.metric != metric_id():
            raise ValueError(
                f"{instance.cell_id}: instance declares metric {instance.metric!r}, not "
                f"the generator's {metric_id()!r}")
        task, canvas = require_rederived(hop, token_prompt, encoder, cell,
                                         instance.data_seed, instance.instance_index,
                                         instance.input_ids_sha256, pools)
        outcomes.append(score_instance(task, canvas, instance, decode, mask_id))
    runner = L._runner_module()
    spec = L.cell_specs([record], arm=record.get("arm", "unset"))[0]
    return runner.run_cell(spec, lambda _spec: outcomes)
