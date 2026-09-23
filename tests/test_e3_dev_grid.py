"""The E3 dev panel and the pre-declared decoder freeze rule.

The property under test is not "the sweep runs" -- it is that the operating point
cannot be chosen on the confirmation data, and that the tie-break rule was fixed before
any measurement existed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lrwkv_evidence.e3 import dev_grid as D  # noqa: E402
from lrwkv_evidence.e3 import grid324 as G  # noqa: E402


def _row(sampler="matched_bernoulli", steps=16, tau=1.0, accuracy=0.5):
    return {"sampler": sampler, "steps": steps, "tau": tau,
            "commit_order": D.COMMIT_ORDER, "accuracy": accuracy}


class TestThePanelIsDevOnly(unittest.TestCase):
    def test_the_dev_seeds_are_the_generators_own_and_disjoint_from_the_grid(self):
        dev = set(D.dev_seeds())
        grid = set(G.DATA_SEEDS)
        self.assertEqual(dev, {201, 202, 203})
        self.assertEqual(dev & grid, set(),
                         "the calibration panel would draw from the confirmation split")

    def test_the_split_is_the_generators_dev_split_not_a_new_one(self):
        # A registered ``iclr2027_dev`` would be a fourth alphabet to keep disjoint;
        # the generator already owns a dev split and its seeds are already disjoint.
        self.assertEqual(D.DEV_SPLIT, "dev")
        self.assertNotEqual(D.DEV_SPLIT, G.GRID_SPLIT)

    def test_the_panel_has_one_cell_per_family(self):
        cells = D.panel_cells()
        self.assertEqual({c.family for c in cells}, set(G.PAPER_FAMILIES))
        self.assertEqual(len(cells), len(G.PAPER_FAMILIES) * len(D.DEV_SETTINGS))

    def test_the_panel_runs_at_the_smallest_primary_length(self):
        # Calibrating at 64K would cost 4x for an operating point applied at 16K.
        self.assertEqual(D.DEV_LENGTH, min(G.LENGTHS))
        self.assertIn(D.DEV_LENGTH, G.LENGTHS)


class TestTheSweepIsTheDeclaredGrid(unittest.TestCase):
    def test_the_sweep_covers_exactly_the_declared_product(self):
        configs = D.sweep_configs()
        n = (len(D.DECODER_GRID["sampler"]) * len(D.DECODER_GRID["steps"])
             * len(D.DECODER_GRID["tau"]))
        self.assertEqual(len(configs), n)
        self.assertEqual(len({D.config_id(c) for c in configs}), n)

    def test_commit_order_is_fixed_not_swept(self):
        # It is part of the frozen operating point (DECODER_KEYS) but not one of the
        # three axes the plan names, so sweeping it would report a different grid than
        # the one declared.
        self.assertIn("commit_order", D.DECODER_KEYS)
        self.assertEqual({c["commit_order"] for c in D.sweep_configs()},
                         {D.COMMIT_ORDER})

    def test_every_config_key_is_a_frozen_operating_point_key(self):
        for config in D.sweep_configs():
            self.assertEqual(set(config), set(D.DECODER_KEYS))


class TestTheSelectionRuleIsPreDeclared(unittest.TestCase):
    def test_the_highest_accuracy_wins(self):
        got = D.select_decoder([_row(accuracy=0.30), _row(steps=64, accuracy=0.55),
                                _row(steps=8, accuracy=0.41)])
        self.assertEqual(int(got["steps"]), 64)
        self.assertEqual(got["dev_accuracy"], 0.55)

    def test_a_tie_goes_to_the_fewer_steps(self):
        # Steps are the only axis that costs GPU time, and a tie is not evidence for
        # paying 8x for it.
        got = D.select_decoder([_row(steps=64, accuracy=0.5),
                                _row(steps=8, accuracy=0.5)])
        self.assertEqual(int(got["steps"]), 8)

    def test_a_tie_after_steps_goes_to_matched_bernoulli(self):
        got = D.select_decoder([_row(sampler="confidence", steps=16, accuracy=0.5),
                                _row(sampler="matched_bernoulli", steps=16,
                                     accuracy=0.5)])
        self.assertEqual(got["sampler"], "matched_bernoulli")

    def test_a_tie_after_the_sampler_goes_to_the_lower_tau(self):
        got = D.select_decoder([_row(tau=1.0, accuracy=0.5),
                                _row(tau=0.7, accuracy=0.5)])
        self.assertEqual(float(got["tau"]), 0.7)

    def test_the_margin_is_reported_so_a_coin_flip_is_visible(self):
        got = D.select_decoder([_row(steps=8, accuracy=0.5),
                                _row(steps=16, accuracy=0.5)])
        self.assertEqual(got["dev_margin"], 0.0)
        self.assertIn("s16", got["dev_runner_up"]["config"])

    def test_a_row_that_never_ran_is_refused_not_ranked_as_zero(self):
        # "did not run" and "scored zero" are different, and ranking them together
        # prefers whichever configuration happened to run at all.
        row = _row(accuracy=None)
        with self.assertRaises(SystemExit) as cm:
            D.select_decoder([_row(accuracy=0.4), row])
        self.assertIn("did not run", str(cm.exception))

    def test_a_row_missing_a_key_is_refused(self):
        bad = {"sampler": "matched_bernoulli", "steps": 16, "accuracy": 0.9}
        with self.assertRaises(SystemExit) as cm:
            D.select_decoder([bad])
        self.assertIn("tau", str(cm.exception))

    def test_an_empty_sweep_is_refused(self):
        with self.assertRaises(SystemExit):
            D.select_decoder([])


class TestTheFreezeIsWriteOnce(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.protocol = Path(self.tmp.name) / "protocol.json"
        self.protocol.write_text(json.dumps({"quality": {}}), encoding="utf-8")

    def test_the_freeze_writes_and_reads_back_from_the_file(self):
        decoder = _row()
        written = D.freeze(self.protocol, {k: decoder[k] for k in D.DECODER_KEYS})
        on_disk = json.loads(self.protocol.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["quality"]["decoder"], written)

    def test_a_second_freeze_is_refused(self):
        # The whole point: an operating point chosen after a confirmation run has been
        # scored is chosen on the test panel.
        D.freeze(self.protocol, {k: _row()[k] for k in D.DECODER_KEYS})
        with self.assertRaises(SystemExit) as cm:
            D.freeze(self.protocol, {**{k: _row()[k] for k in D.DECODER_KEYS},
                                     "steps": 64})
        self.assertIn("already carries", str(cm.exception))

    def test_the_selected_point_satisfies_the_lanes_own_freeze_check(self):
        # The two modules must agree: require_frozen_decoder is what the scoring path
        # consults, so a point written by select_decoder has to satisfy it.
        from lrwkv_evidence.e3 import lane as L
        selected = D.select_decoder([_row(accuracy=0.6)])
        point = {k: selected[k] for k in D.DECODER_KEYS}
        protocol = {"quality": {"decoder": point}}
        self.assertEqual(L.require_frozen_decoder(protocol), point)


if __name__ == "__main__":
    unittest.main()
