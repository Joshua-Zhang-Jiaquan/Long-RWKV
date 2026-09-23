"""Build the paper's 324-cell long-context grid, token-exact under the RWKV vocab.

``protocol.json["long_context"]`` declares 3 families x 3 lengths x 3 evidence
positions x 4 binding loads x 3 distractor classes = 324 cells, 200 instances each
(40 from each of seeds 101-105) = 64,800 examples per checkpoint.  The instances
are produced by the *pinned* generator in
``$G/rwkv04b_v7/longrwkv/tasks/{hop_tasks,token_prompt}.py``, which is never
edited here: this module extends it the way ``protocol_patch`` extends the paper's
arm table.

Three measured mismatches between the paper's grid and that generator
-------------------------------------------------------------------
1. **Family names.**  The paper says ``associative_recall`` /
   ``overwrite_delayed_query`` / ``code_dataflow``; the generator declares
   ``associative_recall`` / ``latest_write`` / ``pointer_dataflow``.  These are
   the same three constructions under two vocabularies -- the generator's own
   ``registry.FAMILY_MAP`` already maps ``overwrite -> latest_write`` and
   ``dataflow -> pointer_dataflow``.  :data:`FAMILY_MAP` here records the paper's
   spelling explicitly so a cell id can never be read as a fourth family.

2. **Data seeds.**  ``hop_tasks.SPLIT_DATA_SEEDS`` gives 101-103 to *train*,
   201-203 to dev and 301-303 to test, and refuses seed 101 on the test split.
   The paper's confirmation grid is seeds 101-105.  Reusing the train split would
   draw test examples from the train alphabet -- exactly the leak the split-owned
   alphabets exist to prevent.  So a **new** split is registered
   (:data:`GRID_SPLIT`) with its own key tag, its own disjoint value window and
   its own seed block, and :func:`check_split_is_disjoint` asserts the
   disjointness against all three shipped splits rather than asserting it in prose.

3. **Instances per cell.**  The generator caps ``instance_index`` at
   ``INSTANCES_PER_CELL = 64``.  The paper wants 200 per cell but as *40 per seed
   x 5 seeds*, and 40 <= 64, so the cap needs no patch: the index runs 0..39 and
   the seed carries the rest.  This is why the seed axis is not cosmetic.

What does not build, and why it is recorded rather than dropped
--------------------------------------------------------------
Measured over all 324 cells (2026-09-21), depth=1, one query:

* **264 cells build** token-exactly at every length.
* **54 cells are infeasible by construction**: ``load=1`` for the two non-recall
  families.  ``latest_write`` needs ``n_queries * depth * OVERWRITES_PER_LINK``
  = 2 binding records and ``pointer_dataflow`` needs ``depth + 1`` = 2, so a
  one-record table cannot host the chain.  The generator refuses instead of
  emitting a truncated chain under a depth-1 label.
* **6 cells exceed the canvas**: ``(16384, position 90%, load 128)`` with
  ``random`` or ``similar`` distractors, all three families.  A 128-record table
  plus its 128-record distractor block is a 2,690-token payload, and starting it
  at 90 % of 16,384 tokens leaves 1,071 tokens too few for the payload, query line
  and answer region.

Those 60 cells are emitted with ``status="unsupported"`` and the generator's own
refusal message.  They are **not** dropped and **not** zero-filled: a macro that
averages "within difficulty cell, then equally across families and lengths" is a
different statistic if a cell silently disappears, and 54 of the 60 are
``load=1`` cells -- the easiest load, whose absence would bias the load curve
upward for two of the three families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

G: Final = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")

#: The pinned generator's checkout.  ``rwkv04b_v7`` is the newest of seven copies
#: and the one whose ``tasks/`` carries ``token_prompt.py``; the version is part of
#: the grid's provenance, so it is a constant here rather than a search.
GENERATOR_ROOT: Final = G / "rwkv04b_v7"

#: The tokenizer the length axis is denominated in.  ``LENGTH_KIND`` in
#: ``token_prompt`` is ``rwkv_token_target``, so "16384 tokens" means 16384 ids
#: under *this* vocabulary and nothing else.
VOCAB: Final = G / "models" / "rwkv7-0.4B-world" / "rwkv_vocab_v20230424.txt"

OUT_ROOT: Final = G / "capability_eval_data" / "lc_grid324"

# --- the paper's grid, verbatim from protocol.json["long_context"] ----------

PAPER_FAMILIES: Final = ("associative_recall", "overwrite_delayed_query",
                         "code_dataflow")
LENGTHS: Final = (16384, 32768, 65536)
POSITION_FRACTIONS: Final = (10, 50, 90)
BINDING_LOADS: Final = (1, 8, 32, 128)
DISTRACTORS: Final = ("none", "random", "similar")
DATA_SEEDS: Final = (101, 102, 103, 104, 105)
INSTANCES_PER_SEED: Final = 40
CELLS: Final = 324
EXAMPLES_PER_CHECKPOINT: Final = 64_800

#: The paper's family name -> the pinned generator's.  The generator's own
#: ``registry.FAMILY_MAP`` makes the same two identifications from its shorter
#: evaluation-side names ("overwrite", "dataflow").
FAMILY_MAP: Final = {
    "associative_recall": "associative_recall",
    "overwrite_delayed_query": "latest_write",
    "code_dataflow": "pointer_dataflow",
}

#: The grid's own split.  Not ``test``: that split owns seeds 301-303 and would
#: refuse 101, and borrowing ``train`` (which does own 101-103) would draw the
#: confirmation set from the training alphabet.
GRID_SPLIT: Final = "iclr2027_grid"
GRID_KEY_TAG: Final = "g7"
GRID_VALUE_WINDOW_START: Final = 2800
GRID_SEED_BLOCK: Final = 3 << 40

#: One query per instance, depth 1.  The paper's 324-cell grid has no hop-depth
#: axis and no answer-vector axis -- its difficulty axes are length, position,
#: load and distractors -- so both are pinned at the generator's minimum and
#: recorded in every cell record, rather than left to a default.
DEPTH: Final = 1
N_QUERIES: Final = 1

STATUS_OK: Final = "ok"
STATUS_UNSUPPORTED: Final = "unsupported"


def _import_generator():
    """Import the pinned generator, and nothing torch-shaped.

    ``longrwkv.tasks`` is importable without torch by design (its own docstring
    says so); ``longrwkv.eval`` is not, which is why only ``tasks`` is touched.
    """
    root = str(GENERATOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from longrwkv.tasks import hop_tasks, registry, token_prompt
    return hop_tasks, token_prompt, registry


# --- the constant-gold defect ----------------------------------------------

#: The pinned generator's ``_Pool.take()`` returns ``0, 1, 2, ...`` in order, and
#: ``generate_hop_task`` allocates the *query chain first*.  So the queried key is
#: always label index 0 and its value is always value index 0, for every instance
#: of every cell.  Measured 2026-09-21 over all three families, four loads, three
#: distractor classes, three positions and all five seeds:
#:
#:     associative_recall       distinct golds = 1   [('v2800',)]
#:     overwrite_delayed_query  distinct golds = 1   [('v2800',)]
#:     code_dataflow            distinct golds = 14  (its answer is computed, not
#:                                                   drawn, so the pool bias only
#:                                                   reaches it indirectly)
#:
#: and the same on the shipped splits: ``test`` always answers ``v2200``, ``train``
#: always ``v1000``.  The metric is ``final_answer_exact_match``, so a model that
#: emits the single string ``v2800`` scores 100 % on 64,800 examples without
#: reading the canvas.  At ``load=1`` it is worse still: the whole prompt is one
#: record, so 200 instances rendered 1 distinct prompt.
#:
#: This is a defect in the generator, not in the grid, and it reaches the banked
#: 16K A3 readings too (they used this generator).  It is fixed *here*, for the
#: instances this module builds, by permuting each pool per instance -- the pinned
#: file is not edited, in keeping with how ``protocol_patch`` extends rather than
#: rewrites.  :func:`check_golds_vary` is the standing guard.
POOL_DEMAND_MEASURED: Final = {
    # (family, load=128, distractor) -> (key labels taken, values taken).  The
    # permutation must not exhaust a pool, so the worst demand is recorded: 255 of
    # LABEL_POOL_SIZE=384 and 193 of VALUE_WINDOW_WIDTH=512.
    "worst_key_labels": 255,
    "worst_values": 193,
    "label_pool_size": 384,
    "value_window_width": 512,
}


class _PermutingPoolFactory:
    """Replaces ``hop_tasks._Pool`` so each instance draws a different gold.

    ``_Pool.__init__`` takes ``(size, what)`` and no randomness, so the permutation
    source is held here and re-seeded per instance by :meth:`seed`.  Single-threaded
    and seeded from the instance's own ``generator_seed``, so the build stays exactly
    reproducible -- which is checked by rebuilding a cell and comparing hashes.

    Exhaustion is still a refusal, never a wrap-around: the permutation is a
    bijection on ``range(size)``, so ``take()`` can hand out at most ``size``
    distinct indices, the same bound the original enforces.
    """

    def __init__(self, base: type) -> None:
        self._base = base
        self._rng: random.Random | None = None
        factory = self

        class PermutedPool(base):  # type: ignore[misc, valid-type]
            def __init__(self, size: int, what: str) -> None:
                super().__init__(size, what)
                rng = factory._rng
                if rng is None:
                    raise ValueError(
                        "a permuted pool was built with no seeded source; the "
                        "instance's generator seed must be set first or the draw "
                        "is not reproducible")
                order = list(range(size))
                rng.shuffle(order)
                self._order = order

            def take(self) -> int:
                return self._order[super().take()]

        self.pool = PermutedPool

    def seed(self, value: int) -> None:
        """Re-seed the permutation source for the next instance.

        The digest is deliberately ``hashlib``, not ``hash()``.  An earlier version
        used ``("pool", value).__hash__() ^ value``, and a tuple containing a *string*
        hashes under ``PYTHONHASHSEED``, which is randomized per process: the build
        was exactly reproducible *within* one process and drew different permutations
        in the next one.  Measured 2026-09-21 by ``--verify``: 37,000 of 52,800
        written instances failed to re-derive.  Since the pod-side evaluator rebuilds
        each prompt from (cell, seed, index) and compares against the banked hash,
        that would have made every cell record unusable -- and nothing in a
        single-process build could have noticed.
        """
        digest = hashlib.sha256(b"pool:" + value.to_bytes(16, "little", signed=True))
        self._rng = random.Random(int.from_bytes(digest.digest()[:16], "little"))


def check_pool_demand_fits(hop_tasks) -> None:
    """The permutation is only sound if no cell exhausts a pool.

    A permuted pool hands out the same *number* of indices as the original, so if
    the original never exhausted, neither does this one.  What changes is *which*
    indices, and a cell that took 255 of 384 labels sequentially still takes 255
    after permuting.  The two declared sizes are checked against the measured worst
    demand so a future grid widening (more queries, deeper chains) fails here
    rather than as a mid-build refusal.
    """
    m = POOL_DEMAND_MEASURED
    if hop_tasks.LABEL_POOL_SIZE != m["label_pool_size"]:
        raise ValueError(
            f"the generator's label pool is now {hop_tasks.LABEL_POOL_SIZE}, not "
            f"{m['label_pool_size']}; re-measure the worst-cell demand "
            f"({m['worst_key_labels']}) before trusting it")
    if hop_tasks.VALUE_WINDOW_WIDTH != m["value_window_width"]:
        raise ValueError(
            f"the generator's value window is now {hop_tasks.VALUE_WINDOW_WIDTH}, "
            f"not {m['value_window_width']}; re-measure the worst-cell demand")
    if m["worst_key_labels"] > m["label_pool_size"]:
        raise ValueError("the worst cell demands more key labels than the pool has")
    if m["worst_values"] > m["value_window_width"]:
        raise ValueError("the worst cell demands more values than the window has")


def check_golds_vary(records: list[dict], min_distinct: int = 20,
                     min_fraction: float = 0.25) -> None:
    """Refuse a cell whose gold answer is the same string every time.

    This is the guard for the defect above, and it is stated per *cell* because
    that is the unit the metric averages over: a cell with one gold is a cell on
    which a constant predictor scores 1.0, whatever the other 323 cells do.
    ``code_dataflow`` computes its answer from the flow rather than drawing it, so
    its distinct count is naturally lower -- the floor is deliberately loose.

    The floor is ``min(min_distinct, ceil(min_fraction * n))`` rather than a bare
    constant, because a smoke run with a smaller ``--instances-per-seed`` has fewer
    instances than the constant and would fail for having only 9 distinct golds out
    of 10 -- which is healthy.  Tying it to ``n`` keeps one threshold meaningful at
    both sizes; the measured defect produced *one* distinct gold at every size, so
    no plausible floor misses it.
    """
    for rec in records:
        if rec["status"] != STATUS_OK:
            continue
        n = len(rec["instances"])
        golds = {tuple(inst["answers"]) for inst in rec["instances"]}
        floor = min(min_distinct, max(2, math.ceil(min_fraction * n)))
        if len(golds) < floor:
            raise ValueError(
                f"{rec['cell_id']}: only {len(golds)} distinct gold answer(s) over "
                f"{n} instances ({sorted(golds)[:3]}), below the floor of {floor}; "
                f"a constant-gold cell is scored 1.0 by a constant predictor")



def register_grid_split(hop_tasks) -> None:
    """Add :data:`GRID_SPLIT` to the pinned generator's split table.

    Idempotent.  Done by *construction* -- building a real ``SplitAlphabet`` and
    inserting it -- rather than by relaxing the seed check, so every guard the
    generator applies to a shipped split applies to this one too.
    """
    if GRID_SPLIT in hop_tasks._SPLIT_ALPHABETS:
        return
    alphabet = hop_tasks.SplitAlphabet(
        split=GRID_SPLIT, key_tag=GRID_KEY_TAG,
        value_window=range(GRID_VALUE_WINDOW_START,
                           GRID_VALUE_WINDOW_START + hop_tasks.VALUE_WINDOW_WIDTH),
        data_seeds=DATA_SEEDS, seed_block=GRID_SEED_BLOCK)
    check_split_is_disjoint(hop_tasks, alphabet)
    hop_tasks._SPLIT_ALPHABETS[GRID_SPLIT] = alphabet
    if GRID_SPLIT not in hop_tasks.SPLITS:
        hop_tasks.SPLITS = (*hop_tasks.SPLITS, GRID_SPLIT)


def check_split_is_disjoint(hop_tasks, alphabet) -> None:
    """Refuse a grid alphabet that collides with any shipped split.

    Three separate collisions are possible and each would be invisible in the
    output: a shared key tag makes a grid key indistinguishable from a train key,
    a shared value window makes a grid *answer* indistinguishable from a train
    answer, and an overlapping seed block makes a grid instance a replay of a
    train instance.  All three are checked.
    """
    for name, other in hop_tasks._SPLIT_ALPHABETS.items():
        if other.key_tag == alphabet.key_tag:
            raise ValueError(
                f"grid key tag {alphabet.key_tag!r} collides with split {name!r}; "
                f"a grid key would be indistinguishable from a {name} key")
        shared = set(other.value_window) & set(alphabet.value_window)
        if shared:
            raise ValueError(
                f"grid value window overlaps split {name!r} on {len(shared)} "
                f"values, so a grid answer could be a {name} answer")
    lo = alphabet.seed_block + min(alphabet.data_seeds) * hop_tasks.SEED_INSTANCE_STRIDE
    hi = (alphabet.seed_block + max(alphabet.data_seeds) * hop_tasks.SEED_INSTANCE_STRIDE
          + hop_tasks.INSTANCES_PER_CELL)
    for name in hop_tasks.SPLITS:
        if name == alphabet.split:
            continue
        olo, ohi = hop_tasks.seed_block(name)
        if not (hi <= olo or ohi <= lo):
            raise ValueError(
                f"grid seed block [{lo},{hi}) overlaps split {name!r}'s "
                f"[{olo},{ohi}); a grid instance would replay a {name} instance")


def cell_id(family: str, length: int, position: int, load: int,
            distractor: str) -> str:
    """The cell id, in ``long_context_cells.csv``'s exact spelling.

    The CSV is the join key for every downstream table, so this is a *transcription*
    of its format (``associative_recall_L16384_p10_b1_none``), not a new naming
    scheme that happens to look similar.
    """
    return f"{family}_L{length}_p{position}_b{load}_{distractor}"


@dataclass(frozen=True, slots=True)
class Cell:
    """One declared cell of the grid, before any instance is drawn."""

    family: str
    length: int
    position: int
    load: int
    distractor: str

    @property
    def cell_id(self) -> str:
        return cell_id(self.family, self.length, self.position, self.load,
                       self.distractor)

    @property
    def generator_family(self) -> str:
        return FAMILY_MAP[self.family]


def all_cells() -> list[Cell]:
    """The 324 declared cells, in the CSV's order."""
    cells = [Cell(f, n, p, b, d)
             for f in PAPER_FAMILIES
             for n in LENGTHS
             for p in POSITION_FRACTIONS
             for b in BINDING_LOADS
             for d in DISTRACTORS]
    if len(cells) != CELLS:
        raise ValueError(f"grid enumerates {len(cells)} cells, protocol says {CELLS}")
    return cells


def build_instance(hop_tasks, token_prompt, encoder, cell: Cell, data_seed: int,
                   instance_index: int, pools: "_PermutingPoolFactory | None" = None,
                   split: str = GRID_SPLIT):
    """One instance, or a refusal string naming why the cell cannot be built.

    Returns ``(task, canvas, None)`` on success and ``(None, None, reason)`` when
    the generator refuses.  The refusal text is the generator's own -- it names the
    shortfall in records or tokens -- because a paraphrase would lose the number
    that makes an ``unsupported`` row diagnosable.

    ``pools`` installs :class:`_PermutingPoolFactory` for the duration of the draw,
    which is what makes the gold answer vary between instances (see
    :data:`POOL_DEMAND_MEASURED`).  It is restored afterwards even on a refusal, so
    a caller that shares the module with anything else is unaffected.
    """
    try:
        request = hop_tasks.HopTaskRequest(
            family=cell.generator_family, split=split,
            token_length=cell.length, depth=DEPTH, load=cell.load,
            position_fraction=cell.position, distractor=cell.distractor,
            n_queries=N_QUERIES, data_seed=data_seed,
            instance_index=instance_index)
        request.require_feasible()
    except hop_tasks.Refusal as exc:
        return None, None, f"infeasible_cell: {exc}"
    seed = request.generator_seed()
    # The cell's own coordinates must reach the draw: generator_seed() is a
    # function of (split, data_seed, instance_index) only, so two different cells
    # at the same index would otherwise share a permutation.
    cell_salt = int.from_bytes(
        hashlib.sha256(cell.cell_id.encode("utf-8")).digest()[:8], "little")
    saved = hop_tasks._Pool
    if pools is not None:
        pools.seed(seed ^ cell_salt)
        hop_tasks._Pool = pools.pool
    try:
        task = hop_tasks.generate_hop_task(
            request, hop_tasks.as_draws(random.Random(seed ^ cell_salt)))
        canvas = token_prompt.render_token_prompt(
            encoder, task, target_tokens=cell.length,
            position_fraction=cell.position)
    except token_prompt.TokenPromptRefusal as exc:
        return None, None, f"canvas_too_small: {exc}"
    except hop_tasks.Refusal as exc:
        return None, None, f"generator_refusal: {exc}"
    finally:
        hop_tasks._Pool = saved
    if len(canvas.input_ids) != cell.length:
        raise ValueError(
            f"{cell.cell_id} seed={data_seed} i={instance_index}: rendered "
            f"{len(canvas.input_ids)} tokens for a declared length of {cell.length}; "
            f"the length axis is only comparable if this is exact")
    if not hop_tasks.validate_gold(task):
        raise ValueError(
            f"{cell.cell_id} seed={data_seed} i={instance_index}: the generator's "
            f"own gold re-derivation disagrees with the stored answer")
    if not token_prompt.validate_rendered_prompt(encoder, task, canvas):
        raise ValueError(
            f"{cell.cell_id} seed={data_seed} i={instance_index}: the rendered "
            f"canvas does not round-trip to the logical task")
    return task, canvas, None


def ids_sha256(ids) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(int(i).to_bytes(4, "little"))
    return h.hexdigest()


def build_cell(hop_tasks, token_prompt, encoder, cell: Cell,
               instances_per_seed: int = INSTANCES_PER_SEED,
               pools: "_PermutingPoolFactory | None" = None,
               split: str = GRID_SPLIT,
               data_seeds: tuple[int, ...] = DATA_SEEDS) -> dict:
    """Build every instance of one cell, or record the cell as unsupported.

    A cell is unsupported *as a whole* or not at all: the refusals measured here
    are functions of (family, length, position, load, distractor) only, never of
    the seed or the instance index, so a partial cell would mean a refusal we do
    not understand.  That is asserted rather than assumed -- a per-instance
    refusal on an otherwise-building cell raises.
    """
    instances, first_reason = [], None
    for data_seed in data_seeds:
        for index in range(instances_per_seed):
            task, canvas, reason = build_instance(
                hop_tasks, token_prompt, encoder, cell, data_seed, index, pools,
                split)
            if reason is not None:
                if instances:
                    raise ValueError(
                        f"{cell.cell_id}: instance (seed={data_seed}, i={index}) "
                        f"was refused after {len(instances)} instances built: "
                        f"{reason}. A refusal that depends on the seed means the "
                        f"cell's supported/unsupported status is not a property "
                        f"of the cell, which the unsupported rows assume")
                first_reason = first_reason or reason
                continue
            instances.append({
                "cell_id": cell.cell_id,
                "data_seed": data_seed,
                "instance_index": index,
                "generator_seed": hop_tasks.generator_seed(
                    split, data_seed, index),
                "input_ids_sha256": ids_sha256(canvas.input_ids),
                "realized_tokens": len(canvas.input_ids),
                "queries": list(task.queries),
                "answers": list(task.answers),
                "metric": task.metric,
                "target_span": [int(canvas.target_mask.index(True)),
                                int(len(canvas.target_mask))]
                if True in canvas.target_mask else None,
            })
    if first_reason is not None and instances:
        raise ValueError(f"{cell.cell_id}: mixed refusal and success")
    if first_reason is not None:
        return {"cell_id": cell.cell_id, "family": cell.family,
                "history_length": cell.length,
                "evidence_position_fraction": cell.position / 100,
                "binding_load": cell.load, "distractors": cell.distractor,
                "status": STATUS_UNSUPPORTED, "unsupported_reason": first_reason,
                "total_instances": 0, "instances": []}
    expected = len(data_seeds) * instances_per_seed
    if len(instances) != expected:
        raise ValueError(
            f"{cell.cell_id}: built {len(instances)} instances, declared {expected}")
    return {"cell_id": cell.cell_id, "family": cell.family,
            "history_length": cell.length,
            "evidence_position_fraction": cell.position / 100,
            "binding_load": cell.load, "distractors": cell.distractor,
            "status": STATUS_OK, "unsupported_reason": None,
            "total_instances": len(instances), "instances": instances}


#: Share of one cell's instances that may repeat another instance's prompt before the
#: build refuses.  200 instances drawn from a finite pool collide by the birthday
#: bound even when every coordinate reaches the draw; the *systematic* failure this
#: replaces (a cell's coordinates missing the draw) makes all 200 identical. 10 % sits
#: far above the measured tail and far below the failure.
MAX_DUPLICATE_SHARE: Final = 0.10


def check_prompts_are_distinct(records: list[dict]) -> dict:
    """Assert what is guaranteed, and *disclose* what is not.

    The failure this check exists for is specific and silent: every instance of every
    cell is drawn at ``depth=1, n_queries=1`` from ``random.Random(generator_seed)``,
    and ``generator_seed`` is a function of (split, data_seed, instance_index)
    *only* -- it does not include the cell.  If the cell's own coordinates did not
    reach the draw, all 324 cells would render the same 200 prompts and the whole
    difficulty grid would be one cell measured 324 times.

    The assertion that catches that is **within one seed**: if the instance index (or
    the cell) were not reaching the draw, the 40 instances of a seed would be one
    prompt repeated 40 times.  That is achievable by construction and is asserted here.

    Distinctness *across* seeds is not achievable.  Measured 2026-09-21 on
    ``associative_recall_L16384_p10_b1_none``: 40/40 distinct inside each of the five
    seeds, and exactly one collision over the cell's 200 instances --
    (seed 103, i 16) and (seed 104, i 19) render the same canvas and the same gold
    ``v3115``.  That is a birthday coincidence over a finite set of (key, value)
    renderings, not the hazard above, and the previous version of this function
    demanded its absence: it was a lottery the design could lose, and the pre-fix
    build only passed it because a process-random permutation happened to win
    ([[a-build-reproducible-in-one-process-is-not-reproducible]]).

    So a cross-seed repeat is **counted and returned** for the manifest -- a reviewer
    is entitled to the number -- and only a *rate* above :data:`MAX_DUPLICATE_SHARE`
    refuses, which is what a cell whose coordinates stopped reaching the draw looks
    like.  What is never tolerated is a repeat inside one seed.
    """
    seen: dict[str, tuple[str, int, int]] = {}
    duplicates: list[dict] = []
    for rec in records:
        by_seed: dict[int, set[str]] = {}
        for inst in rec["instances"]:
            key = inst["input_ids_sha256"]
            where = (rec["cell_id"], inst["data_seed"], inst["instance_index"])
            by_seed.setdefault(inst["data_seed"], set())
            if key in by_seed[inst["data_seed"]]:
                raise ValueError(
                    f"{rec['cell_id']}: two instances of seed {inst['data_seed']} "
                    f"share a prompt (the second is {where}). The instance index is "
                    f"not reaching the generator's draw, so this cell holds one "
                    f"example repeated, not {len(rec['instances'])}.")
            by_seed[inst["data_seed"]].add(key)
            if key in seen:
                duplicates.append({"cell_id": rec["cell_id"], "at": list(where),
                                   "also_at": list(seen[key])})
            else:
                seen[key] = where
        built = len(rec["instances"])
        cell_dups = sum(1 for d in duplicates if d["cell_id"] == rec["cell_id"])
        if built and cell_dups > MAX_DUPLICATE_SHARE * built:
            raise ValueError(
                f"{rec['cell_id']}: {cell_dups} of {built} instances repeat another "
                f"instance's prompt, over the {MAX_DUPLICATE_SHARE:.0%} ceiling. A "
                f"rate this high is not the birthday tail -- the cell's coordinates "
                f"are barely reaching the draw, so the grid is measuring fewer "
                f"distinct examples than it declares.")
    return {
        "instances_seen": len(seen) + len(duplicates),
        "distinct_prompts": len(seen),
        "cross_seed_duplicates": len(duplicates),
        "duplicate_examples": duplicates[:20],
    }


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == STATUS_OK]
    unsupported = [r for r in records if r["status"] == STATUS_UNSUPPORTED]
    by_reason: dict[str, int] = {}
    for r in unsupported:
        kind = r["unsupported_reason"].split(":", 1)[0]
        by_reason[kind] = by_reason.get(kind, 0) + 1
    built = sum(r["total_instances"] for r in ok)
    return {
        "cells_declared": len(records),
        "cells_ok": len(ok),
        "cells_unsupported": len(unsupported),
        "unsupported_by_reason": by_reason,
        "unsupported_cells": [r["cell_id"] for r in unsupported],
        "instances_built": built,
        "examples_per_checkpoint_declared": EXAMPLES_PER_CHECKPOINT,
        # Stated, not silently reconciled: the shortfall IS the unsupported cells.
        "instances_not_built": EXAMPLES_PER_CHECKPOINT - built,
        "split": GRID_SPLIT,
        "data_seeds": list(DATA_SEEDS),
        "instances_per_seed": INSTANCES_PER_SEED,
        "depth": DEPTH,
        "n_queries": N_QUERIES,
        "length_kind": "rwkv_token_target",
        "vocab": str(VOCAB),
        "generator_root": str(GENERATOR_ROOT),
    }


def verify_written_grid(hop_tasks, token_prompt, encoder, out: Path,
                        pools: "_PermutingPoolFactory | None" = None,
                        instances_per_seed: int = INSTANCES_PER_SEED) -> dict:
    """Re-derive every written instance in a fresh process and compare hashes.

    This is the property the whole design rests on and the reason the cell records
    store an ``input_ids_sha256`` instead of 9 GiB of token ids: the pod-side
    evaluator rebuilds each prompt from (cell, seed, index) and the banked hash is
    what proves it rebuilt the *same* prompt.  An unverified hash would be
    decoration -- it would agree with whatever the evaluator happened to draw.

    Checked here rather than asserted: a permutation seeded from module-level state
    is exactly the kind of thing that reproduces within one process (where the
    factory was constructed once) and diverges across two.
    """
    manifest_path = out / "grid_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path} is absent; nothing to verify")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = mismatched = 0
    cells_ok = cells_unsupported = 0
    bad: list[str] = []
    for cell in all_cells():
        path = out / f"{cell.cell_id}.json"
        if not path.exists():
            raise ValueError(f"{cell.cell_id}: declared by the grid, absent from {out}")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["status"] != STATUS_OK:
            cells_unsupported += 1
            # An unsupported cell must still be unsupported on re-derivation, or
            # its status was a property of the run rather than of the cell.
            _, _, reason = build_instance(hop_tasks, token_prompt, encoder, cell,
                                          DATA_SEEDS[0], 0, pools)
            if reason is None:
                bad.append(f"{cell.cell_id}: recorded unsupported but builds now")
            continue
        cells_ok += 1
        for inst in rec["instances"]:
            task, canvas, reason = build_instance(
                hop_tasks, token_prompt, encoder, cell, inst["data_seed"],
                inst["instance_index"], pools)
            checked += 1
            if reason is not None:
                bad.append(f"{cell.cell_id} seed={inst['data_seed']} "
                           f"i={inst['instance_index']}: refused on re-derivation "
                           f"({reason})")
                mismatched += 1
                continue
            if ids_sha256(canvas.input_ids) != inst["input_ids_sha256"]:
                bad.append(f"{cell.cell_id} seed={inst['data_seed']} "
                           f"i={inst['instance_index']}: prompt hash differs")
                mismatched += 1
            elif list(task.answers) != list(inst["answers"]):
                bad.append(f"{cell.cell_id} seed={inst['data_seed']} "
                           f"i={inst['instance_index']}: gold differs "
                           f"({list(task.answers)} vs {inst['answers']})")
                mismatched += 1
    if bad:
        raise ValueError(
            f"{len(bad)} of {checked} instances did not re-derive; the banked hashes "
            f"cannot gate a pod-side rebuild. First three: {bad[:3]}")
    return {"verified_instances": checked, "mismatched": mismatched,
            "cells_ok": cells_ok, "cells_unsupported": cells_unsupported,
            "manifest_instances_built": manifest["instances_built"],
            "agrees_with_manifest": checked == manifest["instances_built"],
            "instances_per_seed": instances_per_seed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--instances-per-seed", type=int, default=INSTANCES_PER_SEED)
    ap.add_argument("--limit-cells", type=int, default=0,
                    help="build only the first N cells (a smoke run, labelled as one)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and check everything, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="re-derive every written instance and compare its hash and "
                         "gold against the record; builds nothing and writes nothing")
    args = ap.parse_args(argv)

    hop_tasks, token_prompt, registry = _import_generator()
    register_grid_split(hop_tasks)
    check_pool_demand_fits(hop_tasks)
    pools = _PermutingPoolFactory(hop_tasks._Pool)
    encoder = registry.TrieEncoder.from_vocab(str(VOCAB))

    if args.verify:
        report = verify_written_grid(hop_tasks, token_prompt, encoder, args.out,
                                     pools, args.instances_per_seed)
        report["out"] = str(args.out)
        print(json.dumps(report, indent=2))
        return 0

    cells = all_cells()
    if args.limit_cells:
        cells = cells[:args.limit_cells]
    records = [build_cell(hop_tasks, token_prompt, encoder, c,
                          args.instances_per_seed, pools) for c in cells]
    distinctness = check_prompts_are_distinct(records)
    check_golds_vary(records)
    report = summarize(records)
    report["distinctness"] = distinctness
    report["partial_run"] = bool(args.limit_cells) or (
        args.instances_per_seed != INSTANCES_PER_SEED)

    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        for rec in records:
            (args.out / f"{rec['cell_id']}.json").write_text(
                json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        (args.out / "grid_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["out"] = str(args.out)
    report["wrote"] = not args.dry_run
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
