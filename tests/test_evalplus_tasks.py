"""CPU tests for the EvalPlus task builders.

Both defects these pin were *measured* on real rows, and both had the same
signature: a plausible-looking 0 % pass@1 with nothing in the output saying the
grader never ran.

1. ``RLIMIT_NPROC (0, 0)`` (sandbox.py:124) makes OpenBLAS abort on import, so
   every numpy-importing test exits nonzero -- which ``_failure_tag`` records as
   ``tests_failed``, identical to a wrong answer. Canonical solutions scored
   0/40 before the thread-cap preamble and 542/542-minus-1 after.
2. Routing MBPP+ through the ``humaneval`` suite put the prose instruct prompt
   into the executable program, so all rows came back ``syntax_error``.

The tests therefore assert against the *real* scorer (``normalize_problem`` /
``build_candidate_code``) rather than restating the builders' own logic: a test
that re-implements the thing it checks cannot see a contract change.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SCALE = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222"
             "/qz_stage_traj4096_v7/scale")
sys.path.insert(0, str(SCALE))

from lrwkv_evidence.e1 import evalplus_tasks as EP  # noqa: E402

DATA_ROOT = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222"
                 "/capability_eval_data")
TASKS_DIR = DATA_ROOT / "tasks_evalplus"


def _built(name: str) -> list[dict]:
    path = TASKS_DIR / f"{name}.jsonl"
    if not path.is_file():
        raise unittest.SkipTest(f"{path} not built yet")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestRowCounts(unittest.TestCase):
    def test_item_counts_match_the_release(self):
        for name, expected in EP.EXPECTED_ROWS.items():
            self.assertEqual(len(_built(name)), expected,
                             f"{name} is not the {expected}-item release")

    def test_task_ids_unique(self):
        for name in EP.EXPECTED_ROWS:
            ids = [r["task_id"] for r in _built(name)]
            self.assertEqual(len(set(ids)), len(ids))


class TestThreadCap(unittest.TestCase):
    """The preamble must be on every test block, and only on the test block."""

    def test_every_test_block_carries_the_preamble(self):
        for name in EP.EXPECTED_ROWS:
            for row in _built(name):
                self.assertTrue(
                    row["test"].startswith(EP.THREAD_CAP_PREAMBLE),
                    f"{name}/{row['task_id']}: test block lacks the thread cap; "
                    f"OpenBLAS will abort under RLIMIT_NPROC(0,0) and the row "
                    f"will be scored as a wrong answer")

    def test_the_preamble_pins_every_blas_variable(self):
        # Missing one is enough: numpy picks whichever the build honours, and the
        # failure is a 64-thread spawn attempt, not a warning.
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            self.assertIn(var, EP.THREAD_CAP_PREAMBLE)

    def test_the_preamble_is_not_in_the_prompt_or_solution(self):
        # It must not reach the model, and must not alter the candidate program.
        for name in EP.EXPECTED_ROWS:
            for row in _built(name):
                self.assertNotIn("OPENBLAS_NUM_THREADS", row["prompt"])
                self.assertNotIn("OPENBLAS_NUM_THREADS", row["canonical_solution"])

    def test_the_preamble_parses_and_is_self_cleaning(self):
        import ast
        ast.parse(EP.THREAD_CAP_PREAMBLE)
        # `del _os, _v` keeps the names out of the test body's namespace, where
        # they could shadow a task's own variable.
        self.assertIn("del _os, _v", EP.THREAD_CAP_PREAMBLE)


class TestScorerRouting(unittest.TestCase):
    """Assert against the real code.py, not against a restatement of it."""

    def setUp(self):
        try:
            from eval.capability.code import build_candidate_code, normalize_problem
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest(f"code.py not importable: {exc}")
        self.normalize = normalize_problem
        self.assemble = build_candidate_code

    def test_run_as_suite_names_are_accepted_by_the_scorer(self):
        for name, suite in EP.RUN_AS_SUITE.items():
            row = _built(name)[0]
            prob = self.normalize(suite, row)   # must not raise ValueError
            self.assertTrue(prob["test_cases"][0])

    def test_mbppplus_program_is_the_completion_alone(self):
        # The regression that was measured: the humaneval rule prepends the prose
        # prompt, so the program starts with English and every row is a
        # syntax_error -- a silent 0% pass@1.
        row = _built("mbppplus")[0]
        prob = self.normalize("mbppplus", row)
        program = self.assemble("mbppplus", prob["prompt"], "def f():\n    return 1\n")
        self.assertEqual(program, "def f():\n    return 1\n")
        self.assertNotIn("You are an expert Python programmer", program)

    def test_humanevalplus_program_concatenates_the_prompt(self):
        row = _built("humanevalplus")[0]
        prob = self.normalize("humanevalplus", row)
        program = self.assemble("humanevalplus", prob["prompt"], "    return True\n")
        self.assertTrue(program.startswith(prob["prompt"]))

    def test_mbppplus_appends_no_check_call(self):
        # entry_point must stay empty: MBPP+ bodies are self-executing and there
        # is no single entry function to call.
        row = _built("mbppplus")[0]
        prob = self.normalize("mbppplus", row)
        self.assertEqual(prob["entry_point"], "")
        self.assertNotIn("\ncheck(", prob["test_cases"][0][-200:])

    def test_humanevalplus_appends_the_check_call(self):
        row = _built("humanevalplus")[0]
        prob = self.normalize("humanevalplus", row)
        self.assertEqual(prob["entry_point"], row["entry_point"])
        self.assertIn(f"check({row['entry_point']})", prob["test_cases"][0])

    def test_the_base_suites_are_untouched(self):
        # The pinned scorer produced the published GSM8K/HumanEval readings; if
        # these two branches moved, those numbers changed meaning.
        mbpp = {"task_id": 2, "prompt": "find shared elements",
                "test_imports": [], "test_list": ["assert f(a, b) == (4, 5)"]}
        prob = self.normalize("mbpp", mbpp)
        self.assertEqual(prob["test_cases"], ["assert f(a, b) == (4, 5)"])
        self.assertIn("Python code:", prob["prompt"])
        self.assertEqual(self.assemble("mbpp", "wrapper", "code\n"), "code\n")
        self.assertEqual(self.assemble("humaneval", "sig\n", "body\n"), "sig\nbody\n")


class TestPromptRoles(unittest.TestCase):
    def test_mbppplus_prompt_shows_the_base_assertions_only(self):
        # The visible spec is the three original MBPP asserts; the grader is the
        # expanded suite. If the expanded body reached the prompt, the hidden
        # tests would be leaked and the benchmark would be meaningless.
        for row in _built("mbppplus"):
            for case in row["mbpp_base_test_list"]:
                self.assertIn(case, row["prompt"])
            # A 9.5 KB body cannot fit in a prompt built from ~150 B of asserts.
            self.assertLess(len(row["prompt"]), len(row["test"]))

    def test_mbppplus_prompt_matches_the_pinned_mbpp_wrapper(self):
        # Character-for-character with code.py:134-140, so the model sees the
        # same instruction the historical MBPP runs used.
        row = _built("mbppplus")[0]
        self.assertTrue(row["prompt"].startswith(
            "You are an expert Python programmer. Write only the Python code "
            "for this task.\nTask: "))
        self.assertTrue(row["prompt"].endswith("\n\nPython code:\n"))

    def test_humanevalplus_prompt_is_the_raw_signature(self):
        for row in _built("humanevalplus")[:20]:
            self.assertNotIn("You are an expert", row["prompt"])


class TestRefusals(unittest.TestCase):
    def test_base_mbpp_parquet_is_refused(self):
        # A base-MBPP row has no expanded body; scoring it would report base
        # numbers under the plus name, which is invisible in a pass@1.
        with self.assertRaises(SystemExit):
            EP.build_mbppplus([{"task_id": 1, "prompt": "p", "code": "c",
                                "test_imports": [],
                                "test_list": ["assert f() == 1"],
                                "test": "assert f() == 1"}])

    def test_a_check_defining_mbppplus_body_is_refused(self):
        # If the suite shape changes to a check() wrapper, entry_point="" would
        # mean the tests are never invoked -- and every row would "pass".
        big = "def check(c):\n    assert c\n" + "#" * 4000
        with self.assertRaises(SystemExit):
            EP.build_mbppplus([{"task_id": 1, "prompt": "p", "code": "c",
                                "test_imports": [], "test_list": ["assert f()"],
                                "test": big}])

    def test_humanevalplus_without_check_is_refused(self):
        with self.assertRaises(SystemExit):
            EP.build_humanevalplus([{"task_id": "HumanEval/0", "prompt": "p",
                                     "entry_point": "f",
                                     "canonical_solution": "s",
                                     "test": "assert f(1) == 1"}])

    def test_a_wrong_item_count_is_refused(self):
        if not (DATA_ROOT / EP.SOURCES["mbppplus"]).is_dir():
            self.skipTest("parquet not present")
        original = EP.EXPECTED_ROWS["mbppplus"]
        try:
            EP.EXPECTED_ROWS["mbppplus"] = original - 1
            with self.assertRaises(SystemExit):
                EP.build("mbppplus", DATA_ROOT, TASKS_DIR)
        finally:
            EP.EXPECTED_ROWS["mbppplus"] = original


class TestManifest(unittest.TestCase):
    def test_manifest_records_the_run_suite_and_the_cap(self):
        for name in EP.EXPECTED_ROWS:
            path = TASKS_DIR / f"{name}.manifest.json"
            if not path.is_file():
                self.skipTest(f"{path} not built")
            man = json.loads(path.read_text())
            self.assertEqual(man["run_as_suite"], EP.RUN_AS_SUITE[name])
            self.assertEqual(man["items"], EP.EXPECTED_ROWS[name])
            self.assertEqual(len(man["jsonl_sha256"]), 64)
            self.assertTrue(man["source_parquet_sha256"])
            # The reason the cap exists must travel with the file: a later reader
            # seeing a plain preamble cannot otherwise tell it is load-bearing.
            self.assertIn("RLIMIT_NPROC", man["thread_cap_rationale"])


class TestSandboxCalibrationClaim(unittest.TestCase):
    """A build cannot certify a sandbox setting for the file it is writing.

    The manifest used to state ``sandbox_settings_validated: {256, 5.0}`` as a
    literal.  The calibration then measured that exact setting and found 4 rows
    SIGKILLed -- a resource kill that ``_failure_tag`` records as ``tests_failed``,
    which is indistinguishable from a wrong answer in the artifact.  So the
    build-time value is ``None`` and the ceiling arrives only with a receipt.
    """

    def test_a_fresh_build_claims_nothing(self):
        man = EP.manifest("mbppplus", [{"task_id": "Mbpp/2"}],
                          TASKS_DIR / "mbppplus.jsonl",
                          DATA_ROOT / EP.SOURCES["mbppplus"])
        self.assertIsNone(man["sandbox_settings_validated"])
        self.assertEqual(man["sandbox_calibration"]["status"], "not_attached")
        # Name the fix, not just the absence.
        self.assertIn("calibrate_evalplus_sandbox",
                      man["sandbox_calibration"]["how"])

    def test_an_attached_receipt_carries_the_measured_settings(self):
        for name in EP.EXPECTED_ROWS:
            path = TASKS_DIR / f"{name}.manifest.json"
            if not path.is_file():
                self.skipTest(f"{path} not built")
            cal = json.loads(path.read_text())["sandbox_calibration"]
            if cal["status"] != "attached":
                self.skipTest(f"{name} has no calibration attached yet")
            settings = json.loads(path.read_text())["sandbox_settings_validated"]
            self.assertEqual(settings["mem_mb"], 4096)
            self.assertEqual(settings["timeout_s"], 30.0)
            # An unexplained failure means some rows would be charged to the model.
            self.assertEqual(cal["unexplained_failures"], [])
            self.assertGreater(cal["ground_truth_pass_rate"], 0.99)
            # The mode the ceiling was taken under must be visible: the login node
            # reports socket_stub, pods run unshare, and the E1 shards have to
            # assert their own.
            self.assertIn(cal["isolation_mode_measured"],
                          ("socket_stub", "unshare", "none"))
            self.assertIn("unshare", cal["isolation_mode_note"])

    def test_a_receipt_for_another_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "r.json"
            receipt.write_text(json.dumps([{
                "suite_file": "mbppplus", "source": "/some/other/mbppplus.jsonl",
                "n_measured": 378, "mem_mb": 4096, "timeout_s": 30.0,
                "ground_truth_pass_rate": 1.0, "scorable_items": 378}]))
            with self.assertRaises(SystemExit) as cm:
                EP.attach_calibration(receipt, TASKS_DIR)
            self.assertIn("says nothing about this one", str(cm.exception))

    def test_a_partial_measurement_is_refused(self):
        # A ceiling from --limit 20 is a different statistic than a ceiling from
        # all 378 rows, and nothing downstream could tell them apart.
        if not (TASKS_DIR / "mbppplus.manifest.json").is_file():
            self.skipTest("mbppplus not built")
        src = json.loads((TASKS_DIR / "mbppplus.manifest.json").read_text())["path"]
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "r.json"
            receipt.write_text(json.dumps([{
                "suite_file": "mbppplus", "source": src,
                "n_measured": 20, "mem_mb": 4096, "timeout_s": 30.0,
                "ground_truth_pass_rate": 1.0, "scorable_items": 20}]))
            with self.assertRaises(SystemExit) as cm:
                EP.attach_calibration(receipt, TASKS_DIR)
            self.assertIn("subset", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
