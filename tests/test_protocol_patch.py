"""CPU tests for the protocol patch.

The patch edits the *shipped* manuscript's ``protocol.json`` and
``training_run_matrix.csv``.  Two failure modes there are silent -- they produce a
validator that still exits 0 while the paper claims something it did not measure:

* a descriptive baseline arm placed in a phase makes ``validate_protocol.py:43-46``
  demand one matrix row per (arm, seed) for a released checkpoint nobody trained.
  Satisfying that demand means writing three fabricated rows; refusing it means the
  validator fails for a reason no one will connect back to this patch.
* a re-run that rewrites the 48 original rows.  They must stay ``not_run`` with no
  measurements, because that is exactly what scopes the paper's core claims.

Both are checked against a copy in a temp dir, never the real file.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lrwkv_evidence import protocol_patch as P  # noqa: E402
from lrwkv_evidence import tracks  # noqa: E402

REAL_PROTOCOL_DIR = ROOT / "paper" / "protocol"


class TestProtocolPatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dir = self.tmp / "protocol"
        shutil.copytree(REAL_PROTOCOL_DIR, self.dir)
        self.protocol = self.dir / "protocol.json"
        self.matrix = self.dir / "training_run_matrix.csv"

    def _run(self):
        return P.main(["--protocol-dir", str(self.dir)])

    def _protocol(self):
        return json.loads(self.protocol.read_text(encoding="utf-8"))

    def _rows(self):
        with self.matrix.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_every_baseline_arm_lands_in_the_arm_table(self):
        self._run()
        arms = self._protocol()["arms"]
        for arm, spec in tracks.BASELINE_ARMS.items():
            self.assertIn(arm, arms, f"{arm} would hit 'Unknown model arm'")
            for field in ("mixer", "objective", "direction"):
                self.assertEqual(arms[arm][field], spec[field])

    def test_no_baseline_arm_is_in_any_phase(self):
        # The load-bearing one: a phase arm needs one matrix row per training
        # seed.  A released checkpoint has no training runs, so a phase that
        # names it can only be satisfied with rows that were never run.
        self._run()
        p = self._protocol()
        for ph in p["phases"]:
            self.assertEqual(
                sorted(set(ph["arms"]) & set(tracks.BASELINE_ARMS)), [],
                f"phase {ph['id']} claims a descriptive baseline arm")

    def test_a_baseline_arm_in_a_phase_is_refused(self):
        # Prove the guard fires rather than trusting that it is written down.
        p = self._protocol()
        arm = sorted(tracks.BASELINE_ARMS)[0]
        p["phases"][0]["arms"] = [*p["phases"][0]["arms"], arm]
        self.protocol.write_text(json.dumps(p, indent=2), encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            P.patch_protocol(self.protocol, dry_run=True)
        self.assertIn(arm, str(cm.exception))
        self.assertIn("matrix row", str(cm.exception))

    def test_every_registry_arm_is_declared(self):
        # tracks.py may add an arm id at any time; if protocol.json never learns
        # about it, the refusal should come from here and not from a CSV row.
        self._run()
        arms = self._protocol()["arms"]
        for t in tracks.ALL_TRACKS:
            self.assertIn(t.paper_arm, arms,
                          f"{t.track_id} names arm {t.paper_arm}")

    def test_the_forty_eight_original_rows_stay_not_run(self):
        self._run()
        rows = self._rows()
        original = [r for r in rows if r["phase"] != P.PHASE["id"]]
        self.assertEqual(len(original), 48)
        for r in original:
            self.assertEqual(r["status"], "not_run", r["run_id"])
            self.assertFalse(r["actual_parameters"] or r["measured_gpu_hours"]
                             or r["checkpoint_sha256"],
                             f"{r['run_id']} gained a measurement")

    def test_running_it_twice_changes_nothing(self):
        self._run()
        first_p = self.protocol.read_bytes()
        first_m = self.matrix.read_bytes()
        self._run()
        self.assertEqual(self.protocol.read_bytes(), first_p)
        self.assertEqual(self.matrix.read_bytes(), first_m)

    def test_the_patched_protocol_passes_the_paper_validator(self):
        # The validator is the actual gate, so run it -- on the patched copy, with
        # the shipped script, not a paraphrase of its rules.
        self._run()
        shutil.copy(REAL_PROTOCOL_DIR / "validate_protocol.py", self.dir)
        proc = subprocess.run(
            [sys.executable, str(self.dir / "validate_protocol.py"), "--self-test"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        self.assertTrue(report["valid_design_arithmetic"])
        self.assertEqual(report["training_rows"], 60)

    def test_a_phase_arm_and_a_baseline_arm_cannot_be_the_same_id(self):
        clash = sorted(P.NEW_ARMS)[0]
        patched = {**tracks.BASELINE_ARMS, clash: dict(P.NEW_ARMS[clash])}
        original = tracks.BASELINE_ARMS
        tracks.BASELINE_ARMS = patched
        try:
            with self.assertRaises(ValueError) as cm:
                P.patch_protocol(self.protocol, dry_run=True)
            self.assertIn(clash, str(cm.exception))
        finally:
            tracks.BASELINE_ARMS = original


if __name__ == "__main__":
    unittest.main()
