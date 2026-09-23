"""E3 runner: re-derivation, grammar reuse, and the two statistics.

The load-bearing test is :meth:`TestRederivation.test_a_tampered_banked_hash_stops_the_run`.
Everything else could be right and the artifact still be worthless if a prompt that does
not re-derive were scored anyway, because the score would attach to the record's name
while belonging to a different prompt.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lrwkv_evidence.e3 import grid324 as G  # noqa: E402
from lrwkv_evidence.e3 import lane as L  # noqa: E402
from lrwkv_evidence.e3 import runner as R  # noqa: E402

EASY = G.Cell("associative_recall", 16384, 50, 8, "similar")


class _Ctx(unittest.TestCase):
    """Builds a real cell once per class: the generator import is the slow part."""

    @classmethod
    def setUpClass(cls):
        cls.hop, cls.tp, cls.registry = G._import_generator()
        G.register_grid_split(cls.hop)
        cls.enc = cls.registry.TrieEncoder.from_vocab(str(G.VOCAB))
        cls.pools = G._PermutingPoolFactory(cls.hop._Pool)

    def record(self, cell=EASY, n=2):
        return G.build_cell(self.hop, self.tp, self.enc, cell, n, self.pools)

    def first(self, cell=EASY, n=1):
        rec = self.record(cell, n)
        inst = L.instances_of(rec)[0]
        task, canvas = R.require_rederived(self.hop, self.tp, self.enc, cell,
                                           inst.data_seed, inst.instance_index,
                                           inst.input_ids_sha256, self.pools)
        return rec, inst, task, canvas


class TestTheGrammarIsTheGenerators(_Ctx):
    def test_the_grammar_comes_from_the_pinned_generator(self):
        # Not a paraphrase: `_grammar` returns the generator's own compiled pattern.
        hop, _, _ = G._import_generator()
        self.assertIs(R._grammar("associative_recall"),
                      hop.FAMILY_GRAMMAR["associative_recall"])

    def test_an_unknown_family_is_refused_rather_than_guessed(self):
        with self.assertRaises(SystemExit) as cm:
            R._grammar("not_a_family")
        self.assertIn("no answer grammar", str(cm.exception))

    def test_extraction_finds_the_familys_answer_shape(self):
        self.assertEqual(R.extract_answer("the answer is v3115.", "associative_recall"),
                         "v3115")
        self.assertEqual(R.extract_answer(" 42 ", "pointer_dataflow"), "42")
        # The pointer grammar is `\d+`, so it matches the digits *inside* a value
        # token. That is the generator's own definition and not a defect: a cell has
        # one family, so a pointer cell's gold is digits and its answers are digits.
        self.assertEqual(R.extract_answer("v3115", "pointer_dataflow"), "3115")

    def test_unparseable_output_is_none_not_empty_string(self):
        # "wrote something that is not an answer" and "wrote nothing" are different,
        # and the protocol wants the first counted rather than dropped.
        self.assertIsNone(R.extract_answer("I cannot answer that.", "associative_recall"))
        self.assertIsNone(R.extract_answer("", "associative_recall"))

    def test_the_metric_id_is_the_generators(self):
        hop, _, _ = G._import_generator()
        self.assertEqual(R.metric_id(), hop.METRIC)


class TestRederivation(_Ctx):
    def test_a_banked_prompt_re_derives(self):
        _rec, inst, task, canvas = self.first()
        self.assertEqual(G.ids_sha256(canvas.input_ids), inst.input_ids_sha256)

    def test_a_tampered_banked_hash_stops_the_run(self):
        rec = self.record()
        inst = L.instances_of(rec)[0]
        bogus = "0" * 64
        with self.assertRaises(ValueError) as cm:
            R.require_rederived(self.hop, self.tp, self.enc, EASY, inst.data_seed,
                                inst.instance_index, bogus, self.pools)
        msg = str(cm.exception)
        self.assertIn("re-derived", msg)
        self.assertIn(str(inst.data_seed), msg)

    def test_a_wrong_index_is_refused_as_a_mismatch(self):
        # The index is part of the rebuild key, so asking for a different instance of
        # the same seed must not silently return a prompt that hashes to the banked one.
        rec = self.record()
        inst = L.instances_of(rec)[1]
        with self.assertRaises(ValueError):
            R.require_rederived(self.hop, self.tp, self.enc, EASY, inst.data_seed,
                                inst.instance_index + 7, inst.input_ids_sha256,
                                self.pools)


class TestScoring(_Ctx):
    def test_a_decoder_that_returns_the_gold_scores_correct(self):
        _rec, inst, task, canvas = self.first()
        gold = inst.answers[0]
        out = R.score_instance(task, canvas, inst,
                               lambda ids, span: R.Decode(gold), mask_id=65535)
        self.assertTrue(out.correct)
        self.assertTrue(out.joint_exact)
        self.assertTrue(out.parse_ok)

    def test_the_mask_really_replaces_the_span(self):
        # If the gold were still on the canvas the task would be copy, not recall.
        _rec, inst, task, canvas = self.first()
        seen = {}
        lo, hi = inst.target_span

        def decode(ids, span):
            seen["span_ids"] = list(ids[lo:hi])
            return R.Decode("v0000")

        R.score_instance(task, canvas, inst, decode, mask_id=65535)
        self.assertEqual(set(seen["span_ids"]), {65535},
                         "the target span was not masked before decoding")

    def test_a_decoder_that_returns_a_grammatical_wrong_answer_is_incorrect(self):
        _rec, inst, task, canvas = self.first()
        out = R.score_instance(task, canvas, inst,
                               lambda ids, span: R.Decode("v0001"), mask_id=65535)
        self.assertFalse(out.correct)
        self.assertTrue(out.parse_ok)

    def test_an_unparseable_answer_is_counted_not_dropped(self):
        _rec, inst, task, canvas = self.first()
        out = R.score_instance(task, canvas, inst,
                               lambda ids, span: R.Decode("no idea"), mask_id=65535)
        self.assertFalse(out.correct)
        self.assertFalse(out.parse_ok, "an unparseable answer must be visible")

    def test_the_two_statistics_coincide_only_because_the_grid_has_one_query(self):
        # n_queries=1 makes correct == joint_exact; that is a property of the grid, and
        # the assertion exists so widening the grid fails here rather than averaging
        # two statistics into one cell.
        self.assertEqual(G.N_QUERIES, 1)
        R.require_single_query("associative_recall")

    def test_an_instance_without_a_target_span_is_refused(self):
        _rec, inst, task, canvas = self.first()
        import dataclasses
        no_span = dataclasses.replace(inst, target_span=None)
        with self.assertRaises(ValueError) as cm:
            R.score_instance(task, canvas, no_span, lambda i, s: R.Decode("x"), 1)
        self.assertIn("nothing to mask", str(cm.exception))


class TestTheCellRecordIsConsumed(_Ctx):
    def test_a_cell_records_into_the_runners_own_shape(self):
        # ``n`` is instances *per seed*: the cell spans every declared data seed, so
        # n=2 over 5 seeds is 10 items. Asserting the item count rather than only the
        # status is what makes "the cell was scored" mean the whole cell.
        rec = self.record(n=2)
        record = R.run_cell_record(rec, lambda ids, span: R.Decode("v0000"),
                                   mask_id=65535, pools=self.pools)
        # run_cell returns the runner package's CellRecord dataclass, not a dict.
        self.assertEqual(record.status, "complete")
        self.assertEqual(record.metrics["n_attempted"], 2 * len(G.DATA_SEEDS))
        self.assertEqual(record.metrics["coverage"], 1.0)

    def test_a_constant_predictor_scores_exactly_one_over_n(self):
        # A null, and a pointed one: an oracle that always emits one instance's gold
        # must score 1/n and nothing else. This is the instrument for the defect that
        # made this grid necessary in the first place -- the pinned generator answers
        # every instance of every cell with one constant value, so a constant
        # predictor is not a straw man here, it is the null the grid exists to break.
        rec = self.record(n=4)
        gold = rec["instances"][0]["answers"][0]
        record = R.run_cell_record(rec, lambda ids, span: R.Decode(gold),
                                   mask_id=65535, pools=self.pools)
        n = record.metrics["n_attempted"]
        hits = sum(1 for i in rec["instances"] if i["answers"][0] == gold)
        self.assertAlmostEqual(record.metrics["per_item_accuracy_percent"],
                               100.0 * hits / n, places=6)
        self.assertLess(record.metrics["per_item_accuracy_percent"], 100.0,
                        "a constant predictor scored the whole cell")

    def test_a_wrong_metric_in_the_record_is_refused(self):
        rec = self.record(n=1)
        rec["instances"][0]["metric"] = "something_else"
        with self.assertRaises(ValueError) as cm:
            R.run_cell_record(rec, lambda i, s: R.Decode("v0000"), mask_id=1,
                              pools=self.pools)
        self.assertIn("metric", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
