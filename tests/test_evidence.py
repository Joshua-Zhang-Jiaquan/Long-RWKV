"""The evidence collector: parsing, hazard scoping, and the check mode.

Two of these tests exist because the collector got them wrong on its first full run:
the arm hazard fired on 1,100 likelihood-panel rows that were correctly labelled, and
the file-level hazards were repeated on every reading of a file (2,941 and 1,302 rows).
A hazard report that fires on everything teaches its reader to skip it, so both are
pinned here rather than left to a comment.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lrwkv_evidence import evidence as EV  # noqa: E402


class TestRunDirParsing(unittest.TestCase):
    def test_a_run_dir_splits_into_tag_dataset_track(self):
        self.assertEqual(EV.split_run_dir("e1_mmlu_test_a29_c6loop_s9500"),
                         ("e1", "mmlu_test", "a29_c6loop_s9500"))
        self.assertEqual(EV.split_run_dir("e1b_hellaswag_a04_a1_s17"),
                         ("e1b", "hellaswag", "a04_a1_s17"))

    def test_the_longest_dataset_token_wins(self):
        # `mmlu_redux` must not parse as `mmlu` with a track of `redux_...`: the two are
        # different benchmarks and a mis-split would file one under the other's name.
        self.assertEqual(EV.split_run_dir("e1_mmlu_redux_rel_c1_rwkv7_0p4b"),
                         ("e1", "mmlu_redux", "rel_c1_rwkv7_0p4b"))
        self.assertEqual(EV.split_run_dir("e1_openbookqa_a04_a2_s17"),
                         ("e1", "openbookqa", "a04_a2_s17"))

    def test_a_non_run_dir_is_not_parsed(self):
        self.assertIsNone(EV.split_run_dir("cap_eval_0d_mmlu_validity_base"))
        self.assertIsNone(EV.split_run_dir("q1probe_mmlu_validation_x"))


class TestHazardScoping(unittest.TestCase):
    def _rec(self, **kw):
        base = {"source_class": "campaign_multichoice", "scoring_arm": "fwdce",
                "path": "/tmp/merged_x.json"}
        base.update(kw)
        return base

    def test_a_subset_is_flagged_with_both_numbers(self):
        hz = EV._hazards(self._rec(), {"record_count": 1000}, 10042)
        self.assertTrue(any("subset: 1000 of 10042" in h for h in hz), hz)

    def test_a_full_panel_is_not_flagged_as_a_subset(self):
        hz = EV._hazards(self._rec(), {"record_count": 14042}, 14042)
        self.assertFalse([h for h in hz if "subset" in h], hz)

    def test_the_arm_check_only_applies_to_multichoice(self):
        # The regression: `mc070`/`mc_elbo`/`iter16` are legitimate arms of the
        # likelihood panels, and flagging them produced ~1,100 false positives.
        for arm in ("mc070", "mc_elbo", "iter16"):
            # A likelihood-panel row legitimately carries one of these names.
            hz = EV._hazards(self._rec(source_class="banked_readings",
                                       scoring_arm=arm), {}, None)
            self.assertEqual(hz, [], f"{arm} flagged on a banked likelihood row")
        # But a *multichoice* row under an unexpected arm still is: the two names exist
        # because the statistic is one thing and the entry point is another.
        hz = EV._hazards(self._rec(scoring_arm="mc070"), {}, None)
        self.assertTrue(any("neither fwdce nor raw" in h for h in hz), hz)

    def test_the_option_pseudolikelihood_tree_is_flagged_as_ruled_out(self):
        # The manuscript's convention is one-token; this task_dir is the multi-token
        # option pseudolikelihood it explicitly rejects, so those rows are not merely
        # "a different encoding".
        stale = Path("/tmp/merged.json")
        self.assertIn("preprocessed_data/mmlu/validation",
                      "/inspire/hdd/global_user/x/research/DiffRwkv/preprocessed_data/"
                      "mmlu/validation")
        del stale


class TestFileLevelFactsAreLifted(unittest.TestCase):
    def test_a_file_level_hazard_is_reported_once_per_file_not_per_reading(self):
        readings = [
            {"path": "/a.json", "source_class": "campaign_multichoice",
             "hazards": ["registry_hash=unpinned", "subset: 100 of 200"]},
            {"path": "/a.json", "source_class": "campaign_multichoice",
             "hazards": ["registry_hash=unpinned"]},
            {"path": "/b.json", "source_class": "banked_readings",
             "hazards": ["source filename carries a probe_copy provenance marker"]},
        ]
        files = EV.source_facts(readings)
        self.assertEqual(files["/a.json"]["readings"], 2)
        self.assertEqual(files["/a.json"]["facts"], ["registry_hash=unpinned"])
        # The row-specific hazard survives on the row it belongs to.
        self.assertEqual(readings[0]["hazards"], ["subset: 100 of 200"])
        self.assertEqual(readings[1]["hazards"], [])
        self.assertEqual(files["/b.json"]["facts"],
                         ["source filename carries a probe_copy provenance marker"])


class TestTheRecordItself(unittest.TestCase):
    """One collection for the class: hashing every source file is the slow part."""

    @classmethod
    def setUpClass(cls):
        cls.rec = EV.collect()

    def test_collect_produces_the_declared_schema_and_counts(self):
        rec = self.rec
        self.assertEqual(rec["schema"], "lrwkv_readings_v1")
        self.assertTrue(rec["readings"], "the collector found nothing at all")
        self.assertEqual(sum(rec["counts"].values()), len(rec["readings"]))
        for r in rec["readings"]:
            self.assertIn("source_class", r)
            self.assertIn("path", r)
            self.assertIn("hazards", r)

    def test_every_campaign_reading_carries_a_hash_of_its_source(self):
        rec = self.rec
        camp = [r for r in rec["readings"]
                if r["source_class"] == "campaign_multichoice"]
        self.assertTrue(camp, "no campaign readings collected")
        for r in camp:
            self.assertRegex(r["sha256"], r"^[0-9a-f]{64}$")

    def test_the_record_is_written_and_check_verifies_it(self):
        rec = self.rec
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "readings.json"
            out.write_text(json.dumps(rec), encoding="utf-8")
            report = EV.check(json.loads(out.read_text(encoding="utf-8")))
        self.assertEqual(report["changed"], [])
        self.assertEqual(report["missing"], [])
        self.assertGreater(report["verified"], 0)

    def test_check_detects_a_changed_source(self):
        # The whole point of banking a hash: a record that no longer describes the bytes
        # on disk must fail rather than be trusted.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            p.write_text("{}", encoding="utf-8")
            rec = {"readings": [{"path": str(p), "sha256": "0" * 64,
                                 "source_class": "campaign_multichoice"}]}
            report = EV.check(rec)
        self.assertEqual(report["changed"], [str(p)])
        self.assertEqual(report["verified"], 0)


if __name__ == "__main__":
    unittest.main()
