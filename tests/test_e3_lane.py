"""E3 lane: the family chain, the grid->CellSpec adapter, and the freeze refusal.

The load-bearing tests here are the ones about *filing*: three vocabularies name the
same three constructions, and a cell filed under the wrong family is averaged into the
wrong third of ``LC_N`` while the total still reads as a number.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qz"))

from lrwkv_evidence.e3 import grid324 as G  # noqa: E402
from lrwkv_evidence.e3 import lane as L  # noqa: E402


def _record(cell_id="associative_recall_L16384_p10_b1_none", *, status="ok",
            metric=L.METRIC_ID, spans=((16381, 16384),)):
    instances = [{
        "cell_id": cell_id, "data_seed": 101, "instance_index": i,
        "generator_seed": 3298534986752 + i, "input_ids_sha256": f"h{i}",
        "realized_tokens": 16384, "queries": ["g7k117"], "answers": ["v3099"],
        "metric": metric,
        "target_span": list(spans[i % len(spans)]) if spans else None,
    } for i in range(4)]
    return {"cell_id": cell_id, "family": "associative_recall",
            "history_length": 16384, "evidence_position_fraction": 0.1,
            "binding_load": 1, "distractors": "none", "status": status,
            "unsupported_reason": None if status == "ok" else "infeasible_cell: x",
            "total_instances": len(instances) if status == "ok" else 0,
            "instances": instances if status == "ok" else []}


class TestTheFamilyChain(unittest.TestCase):
    def test_the_chain_composes_onto_exactly_the_runners_families(self):
        # One-to-one and onto: a missing image would put a cell nowhere, and a
        # collision would put two constructions in one third of LC_N.
        composed = L.check_family_chain()
        self.assertEqual(composed, {"associative_recall": "recall",
                                    "overwrite_delayed_query": "overwrite",
                                    "code_dataflow": "dataflow"})
        runner = L._runner_module()
        self.assertEqual(sorted(composed.values()), sorted(runner.FAMILIES))

    def test_every_paper_family_in_the_shipped_grid_resolves(self):
        # Against the grid's own cell list, not against the map's own keys: a family
        # present in the CSV but absent from FAMILY_MAP would KeyError at filing time.
        import csv
        csv_path = (Path(__file__).resolve().parents[1] / "paper" / "protocol"
                    / "long_context_cells.csv")
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        composed = L.check_family_chain()
        for row in rows:
            self.assertIn(row["family"], composed, row["cell_id"])

    def test_a_generator_family_the_runner_cannot_name_is_refused(self):
        # The failure this protects against: a generator bump renames latest_write and
        # the composition silently drops a family instead of failing.
        saved = G.FAMILY_MAP
        G.FAMILY_MAP = {**saved, "overwrite_delayed_query": "some_new_family"}
        try:
            with self.assertRaises(SystemExit) as cm:
                L.check_family_chain()
            self.assertIn("does not know", str(cm.exception))
        finally:
            G.FAMILY_MAP = saved

    def test_a_cell_cannot_be_filed_under_an_unknown_family(self):
        with self.assertRaises(SystemExit):
            L.runner_distractor("teleport")


class TestTheGridAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, record) -> Path:
        path = Path(self.tmp.name) / f"{record['cell_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_a_cell_becomes_a_spec_with_the_paper_arm_not_the_track_id(self):
        # The runner keys its table on (arm, family, length, access_mode). Handing it
        # "a29_c6loop_s20500" would give every ladder step its own row for one arm.
        specs = L.cell_specs([_record()], arm="C6")
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].arm, "C6")
        self.assertEqual(specs[0].family, "recall")
        self.assertEqual(specs[0].length, 16384)
        self.assertEqual(specs[0].output_capacity_tokens, 3)

    def test_an_unsupported_cell_is_refused_rather_than_dropped(self):
        # 60 of the grid's 324 cells are unsupported. A panel of 264 labelled as 324
        # is the same number with a different meaning, so the loader refuses and the
        # collector reports the row as unsupported.
        path = self._write(_record(status="unsupported"))
        with self.assertRaises(SystemExit) as cm:
            L.load_cell_record(path)
        self.assertIn("unsupported", str(cm.exception))

    def test_a_cell_carrying_two_metrics_is_refused(self):
        record = _record()
        record["instances"][1]["metric"] = "some_other_metric"
        with self.assertRaises(SystemExit) as cm:
            L.load_cell_record(self._write(record))
        self.assertIn("averaged into one score", str(cm.exception))

    def test_instances_are_read_back_with_what_a_rederivation_needs(self):
        record = L.load_cell_record(self._write(_record()))
        insts = L.instances_of(record)
        self.assertEqual(len(insts), 4)
        self.assertEqual(insts[0].answers, ("v3099",))
        self.assertEqual(insts[0].target_span, (16381, 16384))
        self.assertEqual(insts[0].metric, L.METRIC_ID)
        self.assertTrue(insts[0].input_ids_sha256)

    def test_every_distractor_in_the_grid_round_trips(self):
        for name in G.DISTRACTORS:
            self.assertTrue(L.runner_distractor(name))


class TestTheDecoderMustBeFrozenFirst(unittest.TestCase):
    """The plan's Q2, enforced at the point of use rather than in prose."""

    def test_an_unpinned_decoder_is_refused_and_names_what_is_missing(self):
        for protocol in ({}, {"quality": {}}, {"quality": {"decoder": {}}},
                         {"quality": {"decoder": {"sampler": "matched_bernoulli"}}}):
            with self.assertRaises(SystemExit) as cm:
                L.require_frozen_decoder(protocol)
            self.assertIn("unpinned", str(cm.exception))

    def test_a_none_valued_key_counts_as_unpinned(self):
        # A key present but None is the shape a half-written protocol takes, and it
        # must not read as "pinned to nothing".
        decoder = {"sampler": "matched_bernoulli", "steps": None, "tau": 1.0,
                   "commit_order": "confidence"}
        with self.assertRaises(SystemExit) as cm:
            L.require_frozen_decoder({"quality": {"decoder": decoder}})
        self.assertIn("steps", str(cm.exception))

    def test_a_complete_decoder_is_returned_unchanged(self):
        decoder = {"sampler": "matched_bernoulli", "steps": 16, "tau": 1.0,
                   "commit_order": "confidence"}
        got = L.require_frozen_decoder({"quality": {"decoder": decoder}})
        self.assertEqual(got, decoder)

    def test_the_refusal_says_the_grid_is_the_confirmation_set(self):
        # The reason matters: choosing the sampler on these cells would be choosing it
        # on the test data, which is the thing the freeze exists to prevent.
        with self.assertRaises(SystemExit) as cm:
            L.require_frozen_decoder({})
        self.assertIn("confirmation set", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
