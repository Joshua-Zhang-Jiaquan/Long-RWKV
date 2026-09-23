"""CPU tests for the 324-cell long-context grid.

Three of these tests exist because of a defect measured in the *pinned* generator
on 2026-09-21, not because of anything this module got wrong:

``hop_tasks._Pool.take()`` hands out ``0, 1, 2, ...`` in order and
``generate_hop_task`` allocates the query chain first, so the queried key is always
label index 0 and its value always value index 0.  Every instance of every cell
therefore has the *same* gold answer string, and the grid's metric is
``final_answer_exact_match`` -- a model that emits that one string scores 100 % on
all 64,800 examples without reading the canvas.  At ``load=1`` the whole prompt is
one record, so the prompts collapse too.

:class:`TestTheConstantGoldDefect` pins the defect (so a future generator bump that
fixes it upstream is noticed rather than silently making the workaround dead code),
pins the fix, and pins the two guards.  The rest of the file checks the properties
the grid's *statistics* depend on: the cell set and its order against the shipped
CSV, token-exact realized lengths, gold re-derivation, split disjointness, and that
an unsupported cell is unsupported as a whole.

Everything runs at a reduced ``instances_per_seed`` -- the properties under test are
per-instance or per-cell, and the full 40 would make the suite minutes long for no
extra coverage.  The one thing that genuinely needs the real count (the floor in
:func:`check_golds_vary` scaling with ``n``) is tested on synthetic records.
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lrwkv_evidence.e3 import grid324 as G  # noqa: E402

CELLS_CSV = ROOT / "paper" / "protocol" / "long_context_cells.csv"

#: A cell that builds at every length, and cheaply: 16K canvas, load 8, no
#: distractor block.  Used wherever the test is about instances, not about which
#: cells are feasible.
EASY = G.Cell("associative_recall", 16384, 10, 8, "none")


def _fixture():
    """Import the pinned generator once and hand back the build context."""
    hop_tasks, token_prompt, registry = G._import_generator()
    G.register_grid_split(hop_tasks)
    G.check_pool_demand_fits(hop_tasks)
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    return hop_tasks, token_prompt, encoder


class _Ctx(unittest.TestCase):
    """Base: the generator import is ~5 s, so it is done once per class."""

    @classmethod
    def setUpClass(cls):
        cls.hop, cls.tp, cls.enc = _fixture()

    def build(self, cell, n=2, pools=True):
        factory = G._PermutingPoolFactory(self.hop._Pool) if pools else None
        return G.build_cell(self.hop, self.tp, self.enc, cell, n, factory)


class TestTheDeclaredGrid(_Ctx):
    def test_the_cells_are_the_shipped_csvs_cells_in_its_order(self):
        # The CSV is the join key for every downstream table, so a set match is
        # not enough -- a reordering would silently misalign a positional read.
        with CELLS_CSV.open(newline="", encoding="utf-8") as f:
            shipped = [r["cell_id"] for r in csv.DictReader(f)]
        mine = [c.cell_id for c in G.all_cells()]
        self.assertEqual(len(shipped), G.CELLS)
        self.assertEqual(mine, shipped)

    def test_the_csv_declares_the_same_seeds_and_instance_count(self):
        with CELLS_CSV.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            self.assertEqual([int(s) for s in r["data_seeds"].split(";")],
                             list(G.DATA_SEEDS), r["cell_id"])
            self.assertEqual(int(r["instances_per_seed"]), G.INSTANCES_PER_SEED)
            self.assertEqual(int(r["total_instances"]),
                             len(G.DATA_SEEDS) * G.INSTANCES_PER_SEED)

    def test_the_declared_example_count_is_the_grids_arithmetic(self):
        self.assertEqual(
            G.EXAMPLES_PER_CHECKPOINT,
            G.CELLS * len(G.DATA_SEEDS) * G.INSTANCES_PER_SEED)

    def test_every_paper_family_maps_to_a_generator_family(self):
        for fam in G.PAPER_FAMILIES:
            self.assertIn(fam, G.FAMILY_MAP)
            self.assertIn(G.FAMILY_MAP[fam], self.hop.FAMILIES, fam)
        # Distinct targets: two paper families collapsing onto one generator
        # family would make the family axis three columns of two constructions.
        self.assertEqual(len(set(G.FAMILY_MAP.values())), len(G.PAPER_FAMILIES))


class TestTheConstantGoldDefect(_Ctx):
    def test_the_pinned_generator_still_has_the_defect(self):
        # Pinned deliberately.  If a generator bump fixes this upstream, this test
        # fails and the permuting factory becomes removable -- which is a decision
        # to take on purpose, not to discover from an unexplained passing suite.
        rec = self.build(EASY, n=4, pools=False)
        golds = {tuple(i["answers"]) for i in rec["instances"]}
        self.assertEqual(
            len(golds), 1,
            f"the pinned generator no longer returns a constant gold ({golds}); "
            f"re-check whether _PermutingPoolFactory is still needed")

    def test_the_fix_makes_the_gold_vary(self):
        rec = self.build(EASY, n=4, pools=True)
        golds = {tuple(i["answers"]) for i in rec["instances"]}
        self.assertGreater(len(golds), len(rec["instances"]) // 2,
                           f"only {len(golds)} distinct golds over "
                           f"{len(rec['instances'])} instances")

    def test_the_gold_guard_would_have_caught_it(self):
        # The guard is what stands between a future regression and a published
        # number, so run it against the *actual* defective records.
        bad = self.build(EASY, n=4, pools=False)
        with self.assertRaises(ValueError) as cm:
            G.check_golds_vary([bad])
        self.assertIn("constant predictor", str(cm.exception))
        G.check_golds_vary([self.build(EASY, n=4, pools=True)])  # must not raise

    def test_the_gold_guards_floor_scales_with_the_instance_count(self):
        # A 10-instance smoke run with 9 distinct golds is healthy; a bare
        # min_distinct=20 rejected it.  Both directions are checked so the fix
        # cannot be "lower the floor until nothing fails".
        def rec(golds):
            return {"cell_id": "synthetic", "status": G.STATUS_OK,
                    "instances": [{"answers": [g]} for g in golds]}
        G.check_golds_vary([rec([f"v{i}" for i in range(9)] + ["v0"])])
        with self.assertRaises(ValueError):
            G.check_golds_vary([rec(["v0"] * 10)])
        with self.assertRaises(ValueError):
            G.check_golds_vary([rec(["v0"] * 200)])
        G.check_golds_vary([rec([f"v{i}" for i in range(200)])])

    def test_an_unsupported_cell_is_exempt_from_the_gold_guard(self):
        # It has no instances, so a distinct-count floor on it would be a floor
        # on the empty set -- the guard must skip it, not fail it.
        infeasible = G.Cell("overwrite_delayed_query", 16384, 10, 1, "none")
        rec = self.build(infeasible, n=2)
        self.assertEqual(rec["status"], G.STATUS_UNSUPPORTED)
        G.check_golds_vary([rec])

    def test_the_pool_class_is_restored_after_a_refusal(self):
        # build_instance monkey-patches hop_tasks._Pool.  A refusal path that
        # skipped the restore would leave the pinned module permanently altered
        # for every later caller in the process.
        before = self.hop._Pool
        self.build(G.Cell("code_dataflow", 16384, 90, 128, "similar"), n=1)
        self.assertIs(self.hop._Pool, before)
        self.build(EASY, n=1)
        self.assertIs(self.hop._Pool, before)


class TestInstances(_Ctx):
    def test_realized_lengths_are_token_exact_at_every_length(self):
        # "16384 tokens" has to mean 16384 ids under the RWKV vocab or the length
        # axis is a label, not a measurement.  build_instance raises on a mismatch;
        # this asserts it at all three lengths rather than trusting the one.
        for length in G.LENGTHS:
            rec = self.build(G.Cell("associative_recall", length, 50, 8, "none"), n=1)
            self.assertEqual(rec["status"], G.STATUS_OK, rec.get("unsupported_reason"))
            for inst in rec["instances"]:
                self.assertEqual(inst["realized_tokens"], length, rec["cell_id"])

    def test_the_gold_is_re_derivable_from_the_task(self):
        # The stored answer must survive the generator's own re-derivation, or the
        # scorer and the record disagree about what "correct" is.
        pools = G._PermutingPoolFactory(self.hop._Pool)
        for index in range(3):
            task, canvas, reason = G.build_instance(
                self.hop, self.tp, self.enc, EASY, 101, index, pools)
            self.assertIsNone(reason)
            self.assertTrue(self.hop.validate_gold(task))
            self.assertTrue(self.tp.validate_rendered_prompt(self.enc, task, canvas))

    def test_the_build_is_reproducible(self):
        # A re-run must give the same prompts, or nothing downstream can be
        # re-derived from the manifest's hashes.
        a = self.build(EASY, n=3)
        random.seed(999)  # a shared global RNG would break this
        b = self.build(EASY, n=3)
        self.assertEqual([i["input_ids_sha256"] for i in a["instances"]],
                         [i["input_ids_sha256"] for i in b["instances"]])
        self.assertEqual([i["answers"] for i in a["instances"]],
                         [i["answers"] for i in b["instances"]])

    def test_the_build_is_reproducible_across_processes(self):
        # The one that matters, and the one the in-process test above cannot see.
        # The pod-side evaluator rebuilds each prompt in a *different* process and
        # compares it against the banked hash, so a permutation seeded from anything
        # process-local silently invalidates every cell record.
        #
        # This is not hypothetical: the first version seeded the pool from
        # ``("pool", value).__hash__()``, a tuple containing a string, which hashes
        # under the randomized PYTHONHASHSEED.  Measured 2026-09-21, 37,000 of 52,800
        # written instances failed to re-derive -- while the in-process test passed.
        script = (
            "import sys, json; sys.path.insert(0, %r)\n"
            "from lrwkv_evidence.e3 import grid324 as G\n"
            "hop, tp, reg = G._import_generator(); G.register_grid_split(hop)\n"
            "pools = G._PermutingPoolFactory(hop._Pool)\n"
            "enc = reg.TrieEncoder.from_vocab(str(G.VOCAB))\n"
            "rec = G.build_cell(hop, tp, enc, "
            "G.Cell('associative_recall', 16384, 10, 8, 'none'), 2, pools)\n"
            "print(json.dumps([[i['input_ids_sha256'], i['answers']]"
            " for i in rec['instances']]))\n" % str(ROOT))
        out = []
        for hashseed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": hashseed}
            proc = subprocess.run([sys.executable, "-c", script], env=env,
                                  capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            out.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        self.assertEqual(out[0], out[1])
        self.assertEqual(out[0], out[2])
        # And the child's draw must equal this process's, not merely be stable
        # among the children.
        mine = [[i["input_ids_sha256"], i["answers"]]
                for i in self.build(EASY, n=2)["instances"]]
        self.assertEqual(out[0], mine)

    def test_the_pool_seed_does_not_use_the_randomized_builtin_hash(self):
        # A direct check on the mechanism, so the cause is named even if the
        # cross-process test above is ever skipped for cost.
        factory = G._PermutingPoolFactory(self.hop._Pool)
        factory.seed(12345)
        first = [factory.pool(32, "keys").take() for _ in range(5)]
        factory.seed(12345)
        self.assertEqual([factory.pool(32, "keys").take() for _ in range(5)], first)
        factory.seed(12346)
        self.assertNotEqual([factory.pool(32, "keys").take() for _ in range(5)],
                            first, "different seeds gave the same permutation")

    def test_prompts_are_distinct_within_and_across_cells(self):
        # generator_seed() is a function of (split, data_seed, instance_index)
        # only.  If the cell's coordinates did not reach the draw, these two cells
        # would render the same prompts and the grid would be one cell measured
        # 324 times.
        recs = [self.build(G.Cell("associative_recall", 16384, p, 8, "none"), n=2)
                for p in (10, 50)]
        report = G.check_prompts_are_distinct(recs)
        hashes = [i["input_ids_sha256"] for r in recs for i in r["instances"]]
        self.assertEqual(len(set(hashes)), len(hashes))
        self.assertEqual(report["cross_seed_duplicates"], 0)
        self.assertEqual(report["distinct_prompts"], len(hashes))

    def test_the_distinct_prompt_guard_fires_on_a_repeat_inside_one_seed(self):
        # The load-bearing assertion.  A cell whose index (or whose own coordinates)
        # is not reaching the draw renders one prompt N times inside a single seed.
        dup = {"cell_id": "a", "instances": [
            {"input_ids_sha256": "x", "data_seed": 101, "instance_index": 0},
            {"input_ids_sha256": "x", "data_seed": 101, "instance_index": 1}]}
        with self.assertRaises(ValueError) as cm:
            G.check_prompts_are_distinct([dup])
        self.assertIn("instance index is not reaching", str(cm.exception))

    def test_a_birthday_collision_across_seeds_is_counted_not_fatal(self):
        # Measured 2026-09-21: one collision in 200 instances of
        # associative_recall_L16384_p10_b1_none -- (seed 103, i 16) vs (seed 104,
        # i 19), same canvas and same gold v3115.  A finite pool makes that
        # inevitable, so it is disclosed in the manifest rather than refused; the
        # guard must not be a lottery the design can lose.
        insts = [{"input_ids_sha256": f"h{i}", "data_seed": 101, "instance_index": i}
                 for i in range(20)]
        insts.append({"input_ids_sha256": "h7", "data_seed": 102, "instance_index": 3})
        report = G.check_prompts_are_distinct(
            [{"cell_id": "a", "instances": insts}])
        self.assertEqual(report["cross_seed_duplicates"], 1)
        self.assertEqual(report["duplicate_examples"][0]["at"], ["a", 102, 3])
        self.assertEqual(report["duplicate_examples"][0]["also_at"], ["a", 101, 7])

    def test_the_guard_refuses_a_cell_that_is_mostly_repeats(self):
        # Both directions, so the ceiling cannot be satisfied by simply dropping it:
        # a cell whose coordinates stopped reaching the draw repeats at a rate the
        # birthday tail cannot produce, and that must still refuse.
        # The shape this must still catch: every seed's own instances are distinct
        # (so the index reaches the draw), but the *data_seed* does not, so each seed
        # re-renders the previous seed's prompts.  Half the cell is repeats -- a rate
        # the birthday tail cannot produce.
        insts = [{"input_ids_sha256": f"h{i}", "data_seed": 101,
                  "instance_index": i} for i in range(40)]
        insts += [{"input_ids_sha256": f"h{i}", "data_seed": 102,
                   "instance_index": i} for i in range(40)]
        with self.assertRaises(ValueError) as cm:
            G.check_prompts_are_distinct([{"cell_id": "a", "instances": insts}])
        self.assertIn("ceiling", str(cm.exception))

    def test_each_instance_carries_its_own_seed_provenance(self):
        rec = self.build(EASY, n=2)
        for inst in rec["instances"]:
            self.assertEqual(
                inst["generator_seed"],
                self.hop.generator_seed(G.GRID_SPLIT, inst["data_seed"],
                                        inst["instance_index"]))
        self.assertEqual(
            sorted({i["data_seed"] for i in rec["instances"]}), list(G.DATA_SEEDS))


class TestUnsupportedCells(_Ctx):
    def test_a_load_one_non_recall_cell_is_unsupported_not_truncated(self):
        # latest_write needs 2 binding records and pointer_dataflow needs depth+1;
        # a one-record table cannot host the chain.  The honest output is an
        # unsupported row carrying the generator's own shortfall message.
        for fam in ("overwrite_delayed_query", "code_dataflow"):
            rec = self.build(G.Cell(fam, 16384, 50, 1, "none"), n=2)
            self.assertEqual(rec["status"], G.STATUS_UNSUPPORTED, fam)
            self.assertTrue(rec["unsupported_reason"].startswith("infeasible_cell:"))
            self.assertEqual(rec["instances"], [])
            self.assertEqual(rec["total_instances"], 0)

    def test_the_oversized_canvas_cell_is_unsupported_for_the_canvas_reason(self):
        rec = self.build(G.Cell("associative_recall", 16384, 90, 128, "similar"), n=2)
        self.assertEqual(rec["status"], G.STATUS_UNSUPPORTED)
        self.assertTrue(rec["unsupported_reason"].startswith("canvas_too_small:"),
                        rec["unsupported_reason"])

    def test_the_summary_states_the_shortfall_rather_than_reconciling_it(self):
        ok = self.build(EASY, n=1)
        bad = self.build(G.Cell("code_dataflow", 16384, 50, 1, "none"), n=1)
        s = G.summarize([ok, bad])
        self.assertEqual(s["cells_ok"], 1)
        self.assertEqual(s["cells_unsupported"], 1)
        self.assertEqual(s["unsupported_cells"], [bad["cell_id"]])
        self.assertEqual(s["instances_built"], ok["total_instances"])
        self.assertEqual(s["instances_not_built"],
                         G.EXAMPLES_PER_CHECKPOINT - s["instances_built"])
        self.assertEqual(s["unsupported_by_reason"], {"infeasible_cell": 1})


class TestTheGridSplit(_Ctx):
    def test_the_grid_split_is_registered_and_owns_the_papers_seeds(self):
        alphabet = self.hop._SPLIT_ALPHABETS[G.GRID_SPLIT]
        self.assertEqual(tuple(alphabet.data_seeds), G.DATA_SEEDS)
        self.assertIn(G.GRID_SPLIT, self.hop.SPLITS)

    def test_registering_it_again_changes_nothing(self):
        before = self.hop._SPLIT_ALPHABETS[G.GRID_SPLIT]
        n = len(self.hop.SPLITS)
        G.register_grid_split(self.hop)
        self.assertIs(self.hop._SPLIT_ALPHABETS[G.GRID_SPLIT], before)
        self.assertEqual(len(self.hop.SPLITS), n)

    def test_it_is_disjoint_from_every_shipped_split(self):
        alphabet = self.hop._SPLIT_ALPHABETS[G.GRID_SPLIT]
        for name, other in self.hop._SPLIT_ALPHABETS.items():
            if name == G.GRID_SPLIT:
                continue
            self.assertNotEqual(other.key_tag, alphabet.key_tag, name)
            self.assertEqual(set(other.value_window) & set(alphabet.value_window),
                             set(), name)

    def test_a_colliding_key_tag_is_refused(self):
        shipped = next(a for n, a in self.hop._SPLIT_ALPHABETS.items()
                       if n != G.GRID_SPLIT)
        # Disjoint window on purpose: sharing the grid's would let the *window*
        # rule fire and the test would pass without exercising the tag rule.
        clash = self.hop.SplitAlphabet(
            split="bogus", key_tag=shipped.key_tag,
            value_window=range(90_000, 90_008),
            data_seeds=G.DATA_SEEDS, seed_block=G.GRID_SEED_BLOCK)
        with self.assertRaises(ValueError) as cm:
            G.check_split_is_disjoint(self.hop, clash)
        self.assertIn("key tag", str(cm.exception))

    def test_a_colliding_value_window_is_refused(self):
        shipped = next(a for n, a in self.hop._SPLIT_ALPHABETS.items()
                       if n != G.GRID_SPLIT)
        clash = self.hop.SplitAlphabet(
            split="bogus", key_tag="zz9", value_window=shipped.value_window,
            data_seeds=G.DATA_SEEDS, seed_block=G.GRID_SEED_BLOCK)
        with self.assertRaises(ValueError) as cm:
            G.check_split_is_disjoint(self.hop, clash)
        self.assertIn("value window", str(cm.exception))

    def test_a_colliding_seed_block_is_refused(self):
        # The value window must be disjoint from *every* registered split,
        # including the grid's own -- otherwise the window check fires first and
        # this test would pass without ever reaching the seed-block rule.
        clash = self.hop.SplitAlphabet(
            split="bogus", key_tag="zz9", value_window=range(90_000, 90_008),
            data_seeds=G.DATA_SEEDS, seed_block=0)
        with self.assertRaises(ValueError) as cm:
            G.check_split_is_disjoint(self.hop, clash)
        self.assertIn("seed block", str(cm.exception))


class TestThePoolDemandGuard(_Ctx):
    def test_the_measured_demand_fits_the_pinned_pool_sizes(self):
        G.check_pool_demand_fits(self.hop)
        m = G.POOL_DEMAND_MEASURED
        self.assertEqual(self.hop.LABEL_POOL_SIZE, m["label_pool_size"])
        self.assertEqual(self.hop.VALUE_WINDOW_WIDTH, m["value_window_width"])

    def test_a_shrunken_pool_is_refused(self):
        # The permutation is only sound while no cell exhausts a pool, so a
        # generator bump that shrinks one must stop the build here rather than
        # surface as a mid-build refusal on cell 200-something.
        class Fake:
            LABEL_POOL_SIZE = 64
            VALUE_WINDOW_WIDTH = G.POOL_DEMAND_MEASURED["value_window_width"]
        with self.assertRaises(ValueError) as cm:
            G.check_pool_demand_fits(Fake)
        self.assertIn("label pool", str(cm.exception))

    def test_a_permuted_pool_with_no_seed_is_refused(self):
        # An unseeded permutation would be a non-reproducible build, which is
        # worse than a loud failure.
        factory = G._PermutingPoolFactory(self.hop._Pool)
        with self.assertRaises(ValueError) as cm:
            factory.pool(8, "keys")
        self.assertIn("reproducible", str(cm.exception))

    def test_a_permuted_pool_hands_out_the_same_number_of_indices(self):
        factory = G._PermutingPoolFactory(self.hop._Pool)
        factory.seed(7)
        permuted = factory.pool(16, "keys")
        got = [permuted.take() for _ in range(16)]
        self.assertEqual(sorted(got), list(range(16)))
        with self.assertRaises(self.hop.Refusal):
            permuted.take()


if __name__ == "__main__":
    unittest.main()
