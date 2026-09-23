"""CPU tests for the campaign ledger.

The ledger's job is to make two different situations distinguishable: "this name
has a live job" and "this name's job is over". Both look identical from a
``submitted`` row, so the block is correct and the *reconcile* is the repair. These
tests pin that split, because the failure modes are opposite and both expensive: a
missing block bills a duplicate 1-GPU job, and a stale block silently withholds a
resubmit that a bug fix requires.

Everything here runs against a temporary ledger; no test reads or writes the real
campaign file, and nothing here calls the scheduler.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

QZ_DIR = Path(__file__).resolve().parent.parent / "qz"
sys.path.insert(0, str(QZ_DIR))
sys.path.insert(0, str(QZ_DIR.parent))

import campaign as C  # noqa: E402
import emit_jobs as E  # noqa: E402
from lrwkv_evidence import tracks  # noqa: E402


class LedgerFixture(unittest.TestCase):
    """Redirect LEDGER at a temp file, and cut the network, for one test.

    ``list_all_jobs`` is stubbed to raise: no unit test may reach the scheduler.  A
    test that did would pass or fail according to what the live campaign happens to
    hold, and ``live_census`` is now on the drip's path, so the default has to be
    "no jobs" rather than "whatever is running". Tests that need a census set one.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self._tmp.name) / "campaign_ledger.jsonl"
        self._saved = C.LEDGER
        C.LEDGER = self.ledger
        self._saved_list = C.list_all_jobs

        def no_network():
            raise AssertionError(
                "a unit test called ListJobs; stub live_census in the test instead")

        C.list_all_jobs = no_network
        self.addCleanup(self._restore)

    def _restore(self):
        C.LEDGER = self._saved
        C.list_all_jobs = self._saved_list
        self._tmp.cleanup()

    def stub_census(self, submitted=0, running=0, jobs=None):
        """Pin the live-GPU census for this test."""
        saved = C.live_census
        C.live_census = lambda j=None: {
            "live_jobs": len(jobs or []), "live_gpus_submitted": submitted,
            "live_gpus_running": running, "jobs": jobs or []}
        self.addCleanup(lambda: setattr(C, "live_census", saved))

    def rows(self) -> list[dict]:
        return [json.loads(line) for line in
                self.ledger.read_text().splitlines() if line.strip()]


class TestBlocking(LedgerFixture):
    def test_submitted_blocks(self):
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        self.assertIsNotNone(C.already_submitted("n"))

    def test_attempted_blocks(self):
        # A CreateJob call may have reached the scheduler; the honest reading is
        # that a live job may exist, so the name is burnt until someone checks.
        C.append_ledger({"state": "attempted", "name": "n"})
        self.assertIsNotNone(C.already_submitted("n"))

    def test_a_later_failed_row_unblocks(self):
        # Last-state-wins: this is the whole mechanism by which a fixed job is
        # allowed back in, so it must not depend on row order in the file.
        C.append_ledger({"state": "attempted", "name": "n"})
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        C.append_ledger({"state": "failed", "name": "n", "job_id": "job-1"})
        self.assertIsNone(C.already_submitted("n"))

    def test_succeeded_still_blocks(self):
        # A succeeded name must never be resubmitted: the receipt exists, so the
        # only thing another job buys is a duplicate bill.
        C.append_ledger({"state": "succeeded", "name": "n", "job_id": "job-1"})
        self.assertIsNotNone(C.already_submitted("n"))

    def test_a_failed_row_for_another_name_does_not_unblock(self):
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        C.append_ledger({"state": "failed", "name": "other", "job_id": "job-2"})
        self.assertIsNotNone(C.already_submitted("n"))

    def test_genesis_has_no_name_and_blocks_nothing(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self.assertIsNone(C.already_submitted("n"))


class TestTerminalStatusTable(unittest.TestCase):
    """Which scheduler statuses are allowed to settle a row."""

    def test_restarting_is_not_terminal(self):
        # auto_fault_tolerance surfaces a crashed ROUND as job_restarting; the job
        # can still end job_succeeded on a later round. Settling it to `failed`
        # would unblock a resubmit while the original is still running -- two jobs
        # writing the same OUTDIR.
        self.assertNotIn("job_restarting", C.TERMINAL_STATUS)

    def test_running_and_pending_are_not_terminal(self):
        for status in ("job_running", "job_pending", "job_queuing", "job_creating"):
            self.assertNotIn(status, C.TERMINAL_STATUS)

    def test_every_terminal_status_maps_to_a_known_ledger_state(self):
        for status, state in C.TERMINAL_STATUS.items():
            self.assertIn(state, C.E_STATES, f"{status} -> unknown state {state}")

    def test_succeeded_maps_to_a_blocking_state_and_failed_does_not(self):
        # The direction matters: settling a failure must OPEN the name, and
        # settling a success must keep it closed.
        self.assertIn(C.TERMINAL_STATUS["job_succeeded"], C.BLOCKING_STATES)
        self.assertNotIn(C.TERMINAL_STATUS["job_failed"], C.BLOCKING_STATES)


class TestReconcile(LedgerFixture):
    """``--reconcile`` records what the scheduler says; it never guesses."""

    def _patch_status(self, table: dict[str, tuple[str | None, dict]]):
        saved = C.job_status
        C.job_status = lambda job_id: table.get(job_id, (None, {"error": "unknown"}))
        self.addCleanup(lambda: setattr(C, "job_status", saved))

    def test_a_dry_run_writes_nothing(self):
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        self._patch_status({"job-1": ("job_failed", {"current_running_round": 3})})
        C.reconcile(dry_run=True)
        self.assertEqual([r["state"] for r in self.rows()], ["submitted"])
        self.assertIsNotNone(C.already_submitted("n"))

    def test_a_failed_job_settles_and_unblocks(self):
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        self._patch_status({"job-1": ("job_failed", {"current_running_round": 3})})
        C.reconcile(dry_run=False)
        settled = self.rows()[-1]
        self.assertEqual(settled["state"], "failed")
        self.assertEqual(settled["scheduler_status"], "job_failed")
        self.assertEqual(settled["rounds_run"], 3)
        self.assertEqual(settled["source"], "reconcile")
        self.assertIsNone(C.already_submitted("n"))

    def test_a_running_job_is_left_blocking(self):
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        self._patch_status({"job-1": ("job_running", {"current_running_round": 1})})
        C.reconcile(dry_run=False)
        self.assertEqual([r["state"] for r in self.rows()], ["submitted"])
        self.assertIsNotNone(C.already_submitted("n"))

    def test_a_restarting_job_is_left_blocking(self):
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        self._patch_status({"job-1": ("job_restarting", {"current_running_round": 2})})
        C.reconcile(dry_run=False)
        self.assertEqual([r["state"] for r in self.rows()], ["submitted"])

    def test_an_unreadable_job_is_not_settled(self):
        # A single AccessForbidden (which is also what a mistyped id looks like)
        # must not be read as "the job is over".
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        self._patch_status({})
        C.reconcile(dry_run=False)
        self.assertEqual([r["state"] for r in self.rows()], ["submitted"])
        self.assertIsNotNone(C.already_submitted("n"))

    def test_reconcile_is_idempotent(self):
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        self._patch_status({"job-1": ("job_failed", {"current_running_round": 3})})
        C.reconcile(dry_run=False)
        n_after_first = len(self.rows())
        C.reconcile(dry_run=False)
        self.assertEqual(len(self.rows()), n_after_first,
                         "a settled name must not be re-settled on every tick")

    def test_a_succeeded_job_settles_but_stays_blocked(self):
        C.append_ledger({"state": "submitted", "name": "n", "job_id": "job-1"})
        self._patch_status({"job-1": ("job_succeeded", {"current_running_round": 1})})
        C.reconcile(dry_run=False)
        self.assertEqual(self.rows()[-1]["state"], "succeeded")
        self.assertIsNotNone(C.already_submitted("n"))

    def test_a_row_with_no_job_id_is_skipped(self):
        # An `attempted` row with no job_id means CreateJob's response was never
        # parsed. There is nothing to ask the scheduler about, so it must keep
        # blocking rather than be settled by omission.
        C.append_ledger({"state": "attempted", "name": "n"})
        self._patch_status({})
        C.reconcile(dry_run=False)
        self.assertEqual([r["state"] for r in self.rows()], ["attempted"])
        self.assertIsNotNone(C.already_submitted("n"))


class TestQ1Order(unittest.TestCase):
    def test_the_order_is_built_from_the_registry(self):
        ids = C.q1_track_ids()
        self.assertEqual(len(ids), len(set(ids)), "duplicate track in the Q1 order")
        registry = {t.track_id for t in tracks.ALL_TRACKS}
        self.assertTrue(set(ids) <= registry)

    def test_unschedulable_kinds_are_excluded(self):
        # Reads the derived set, not a copy: a hardcoded list here is how a new
        # kind (MISSING_RUNTIME_DEP, 2026-09-21) got into the Q1 order while the
        # emitter refused it.
        excluded = tracks.UNSCHEDULABLE_KINDS
        kinds = {t.model_kind for t in tracks.ALL_TRACKS
                 if t.track_id in set(C.q1_track_ids())}
        self.assertFalse(kinds & excluded)

    def test_every_ordered_track_builds_a_body(self):
        # The order and the emitter must agree: a track in the order that the
        # emitter refuses would fail at submit time, one job at a time.
        for track_id in C.q1_track_ids():
            C.E.probe_body(track_id)


class TestCapacityRefusalIsNotABodyDefect(LedgerFixture):
    """A full cluster and a wrong body must not be handled the same way.

    ``CreateJob`` returns nonzero for both.  Treating a capacity refusal as a body
    defect writes a ``failed`` row for every remaining name -- 390 of them in this
    campaign -- and each one is a real CreateJob call against a scheduler that has
    already said no.  Looping on the parent-project quota message is a standing
    operator rule, so the sweep stops at the first one and leaves the rest of the
    names *out* of the ledger, where a later re-run picks them up unchanged.
    """

    def test_the_quota_message_is_classified_as_capacity(self):
        # The LIVE refusal, verbatim from the 2026-09-20 ledger. The first version of
        # QUOTA_REFUSALS held only the 父 ("parent") variant, so this 总 ("total")
        # message fell through and 196 names were attempted after the first refusal.
        # Pinning the real string is the only thing that would have caught that.
        self.assertTrue(C.is_quota_refusal(
            "总项目配额不足：当前任务是 高优任务，运行配额限制为 16 GPU，总提交配额限制为 "
            "40 GPU，运行任务已使用 16 GPU，总提交任务已使用 47 GPU，此次申请资源 1 GPU，"
            "超出总提交配额限制，该任务提交失败"))
        self.assertTrue(C.is_quota_refusal("父项目配额不足"))
        self.assertTrue(C.is_quota_refusal("693 Insufficient cpu"))

    def test_the_recorded_quotas_match_the_refusal_that_reported_them(self):
        # There is no CLI verb that reads the quota, so these two constants are the
        # campaign's only model of its own ceiling. If they drift from the message,
        # the drip-feed batch size is sized against a fiction.
        self.assertEqual(C.RUNNING_GPU_QUOTA, 16)
        self.assertEqual(C.SUBMITTED_GPU_QUOTA, 40)
        self.assertLess(C.RUNNING_GPU_QUOTA, C.SUBMITTED_GPU_QUOTA)

    def test_a_capacity_refusal_is_not_a_body_defect_row(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self._fake_create([(0, "ResponseMetadata:\n  Action: CreateJob\n  Error:\n"
                              "    Code: InternalError\n    Message: 总项目配额不足："
                              "超出总提交配额限制\n")])
        row = C.submit_one({"name": "n", "description": "d"}, dry_run=False)
        # returncode 0 with an Error body: the rc alone would have read as success.
        self.assertEqual(row["returncode"], 0)
        self.assertEqual(row["state"], "failed")
        self.assertEqual(row["refusal_kind"], "capacity")

    def test_an_ordinary_error_is_not_capacity(self):
        # A bad spec_id or a malformed body must stay a body defect: stopping the
        # sweep would be right, but marking it "capacity" would suggest waiting
        # fixes it, and it never will.
        self.assertFalse(C.is_quota_refusal("InvalidParameter: spec_id not found"))
        self.assertFalse(C.is_quota_refusal("AccessForbidden"))

    def _fake_create(self, outcomes: list[tuple[int, str]]):
        """Patch subprocess.run so no CreateJob leaves this process."""
        calls: list[str] = []
        seq = list(outcomes)

        class R:
            def __init__(self, rc, out):
                self.returncode, self.stdout, self.stderr = rc, out, ""

        def fake(cmd, **kw):
            calls.append(cmd[2])
            rc, out = seq.pop(0) if seq else (0, "id: job-z")
            return R(rc, out)

        saved = C.subprocess.run
        C.subprocess.run = fake
        self.addCleanup(lambda: setattr(C.subprocess, "run", saved))
        return calls

    def _bodies(self, n: int) -> list[dict]:
        return [{"name": f"n{i}", "description": "d"} for i in range(n)]

    def test_the_sweep_stops_at_the_first_capacity_refusal(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        calls = self._fake_create([(0, "id: job-1"),
                                   (1, "Error: 父项目配额不足"),
                                   (0, "id: job-3")])
        saved = C.group_bodies
        C.group_bodies = lambda *a, **k: self._bodies(3)
        self.addCleanup(lambda: setattr(C, "group_bodies", saved))
        rc = C.main(["--group", "q1probe", "--submit"])
        self.assertEqual(rc, 0)
        states = [r["state"] for r in self.rows() if r.get("name")]
        self.assertEqual(states, ["attempted", "submitted", "attempted", "failed"])
        self.assertEqual(len(calls), 2, "a third CreateJob was issued after the "
                                        "cluster had already refused")
        self.assertIsNone(C.already_submitted("n2"),
                          "the unattempted name must stay out of the ledger")

    def test_the_refused_name_is_left_resubmittable(self):
        # No job exists, so the name must reopen -- otherwise a transient full
        # cluster permanently burns that shard.
        C.append_ledger({"state": "genesis", "at": "t"})
        self._fake_create([(1, "Error: 父项目配额不足")])
        C.submit_one({"name": "n", "description": "d"}, dry_run=False)
        row = self.rows()[-1]
        self.assertEqual(row["refusal_kind"], "capacity")
        self.assertIsNone(C.already_submitted("n"))

    def test_max_submit_counts_only_successful_creates(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        # An already-`succeeded` name is skipped; that skip must not consume the cap,
        # or a resumed sweep would make no progress at all.
        C.append_ledger({"state": "succeeded", "name": "n0", "job_id": "job-old"})
        calls = self._fake_create([(0, "id: job-1"), (0, "id: job-2")])
        saved = C.group_bodies
        C.group_bodies = lambda *a, **k: self._bodies(4)
        self.addCleanup(lambda: setattr(C, "group_bodies", saved))
        C.main(["--group", "q1probe", "--submit", "--max-submit", "2"])
        self.assertEqual(len(calls), 2)
        submitted = [r["name"] for r in self.rows() if r["state"] == "submitted"]
        self.assertEqual(submitted, ["n1", "n2"])
        self.assertIsNone(C.already_submitted("n3"),
                          "a name past the cap must stay out of the ledger")


class TestGroupBodies(LedgerFixture):
    """The group table must not be able to schedule an uncalibrated suite.

    A LedgerFixture even though nothing here writes rows: ``group_bodies`` now calls
    ``check_group_shape``, which READS the ledger, so against the real file these
    tests would pass or fail according to what the live campaign happens to have in
    flight rather than according to the code.
    """

    def test_every_gen_group_has_a_shard_count(self):
        for group, suite in C.GEN_GROUPS.items():
            self.assertIn(suite, C.GEN_SHARDS, f"{group} has no shard count")

    def test_the_gen_lane_tracks_are_registered_and_schedulable(self):
        # Reads the derived set, not a copy: a hardcoded list here is how a new
        # kind (MISSING_RUNTIME_DEP, 2026-09-21) got into the Q1 order while the
        # emitter refused it.
        excluded = tracks.UNSCHEDULABLE_KINDS
        by_id = {t.track_id: t for t in tracks.ALL_TRACKS}
        for suite in C.GEN_SHARDS:
            for track_id in C.gen_track_ids(suite):
                self.assertNotIn(by_id[track_id].model_kind, excluded)

    def test_the_gen_lane_is_a_subset_of_the_probed_tracks(self):
        # Every generative body calls require_probe; a track outside the Q1 order
        # has no receipt and would refuse one job at a time at submit time.
        probed = set(C.q1_track_ids())
        for suite in C.GEN_SHARDS:
            self.assertTrue(set(C.gen_track_ids(suite)) <= probed)

    def test_an_explicit_shard_count_overrides_the_per_suite_default(self):
        bodies = C.group_bodies("e1-humanevalplus", "a29_c6loop_s20500", 2, "test")
        self.assertEqual(len(bodies), 2)

    def test_the_default_sentinel_selects_the_per_suite_count(self):
        bodies = C.group_bodies("e1-gsm8k", "a29_c6loop_s20500",
                                C.DEFAULT_SHARDS, "test")
        self.assertEqual(len(bodies), C.GEN_SHARDS["gsm8k"])


class TestDrip(LedgerFixture):
    """A tick must be resumable and must not mistake "queue full" for "done"."""

    def setUp(self):
        super().setUp()
        # A clear allowance: these tests are about resume/refusal behaviour, and an
        # allowance stop would look identical to a capacity stop from the outside.
        self.stub_census(submitted=0)
        self._bodies = [{"name": f"n{i}", "description": "d"} for i in range(5)]
        # Patch the LANE builder, not group_bodies: a lane now builds per track, so a
        # fake at group_bodies would return these same five names once per track and
        # the duplicate names would mask the resume behaviour under test.
        saved_g = C.lane_bodies
        C.lane_bodies = lambda group, gpus_per_pod: (
            self._bodies if (group == C.DRIP_ORDER[0] and gpus_per_pod == 1) else [])
        self.addCleanup(lambda: setattr(C, "lane_bodies", saved_g))
        saved_r = C.reconcile
        C.reconcile = lambda dry_run=True: 0
        self.addCleanup(lambda: setattr(C, "reconcile", saved_r))

    def _fake_create(self, outcomes):
        seq, calls = list(outcomes), []

        class R:
            def __init__(self, rc, out):
                self.returncode, self.stdout, self.stderr = rc, out, ""

        def fake(cmd, **kw):
            calls.append(cmd[2])
            rc, out = seq.pop(0) if seq else (0, "id: job-z")
            return R(rc, out)

        saved = C.subprocess.run
        C.subprocess.run = fake
        self.addCleanup(lambda: setattr(C.subprocess, "run", saved))
        return calls

    def test_a_dry_tick_submits_nothing_and_counts_the_backlog(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        calls = self._fake_create([])
        C.drip(dry_run=True)
        self.assertEqual(calls, [], "a dry tick called CreateJob")

    def test_a_tick_stops_on_capacity_and_leaves_the_rest_pending(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        calls = self._fake_create([(0, "id: job-1"), (0, "id: job-2"),
                                   (0, "Error:\n Message: 总项目配额不足\n")])
        C.drip(dry_run=False)
        self.assertEqual(len(calls), 3)
        for name in ("n3", "n4"):
            self.assertIsNone(C.already_submitted(name),
                              f"{name} was burnt by a capacity stop")

    def test_the_next_tick_resumes_where_the_last_one_stopped(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self._fake_create([(0, "id: job-1"),
                           (0, "Error:\n Message: 总项目配额不足\n")])
        C.drip(dry_run=False)
        calls = self._fake_create([(0, "id: job-3"), (0, "id: job-4"),
                                   (0, "id: job-5"), (0, "id: job-6")])
        C.drip(dry_run=False)
        # n0 is submitted and must be skipped. n1 was REFUSED, and a refusal is not
        # a submission -- no job exists -- so it reopens and is retried first. Four
        # names remain pending, not three.
        self.assertEqual(len(calls), 4)
        submitted = {r["name"] for r in self.rows() if r["state"] == "submitted"}
        self.assertEqual(submitted, {"n0", "n1", "n2", "n3", "n4"})

    def test_a_tick_reconciles_before_it_submits(self):
        # Reconciling is what frees the submitted-GPU quota, so a tick that submits
        # first would refill against a stale picture and refuse immediately.
        order = []
        C.reconcile = lambda dry_run=True: order.append("reconcile") or 0
        self._fake_create([(0, "id: job-1")])
        saved = C.submit_one
        C.submit_one = lambda body, dry_run: (order.append("submit")
                                              or {"name": body["name"],
                                                  "state": "submitted"})
        self.addCleanup(lambda: setattr(C, "submit_one", saved))
        C.append_ledger({"state": "genesis", "at": "t"})
        C.drip(dry_run=False)
        self.assertEqual(order[0], "reconcile")

    def test_the_drip_order_covers_every_schedulable_group(self):
        # A group missing from the order would silently never be submitted, and the
        # campaign would look finished with a whole lane at not_run.  An omission is
        # therefore only legitimate when it is *stated* in DRIP_EXCLUDED with a
        # reason, which is what makes "we dropped this suite" distinguishable from
        # "someone edited a tuple".
        self.assertEqual(set(C.DRIP_ORDER) | set(C.DRIP_EXCLUDED),
                         {"e1-mmlu", C.AUX_GROUP, *C.GEN_GROUPS})
        self.assertEqual(set(C.DRIP_ORDER) & set(C.DRIP_EXCLUDED), set(),
                         "a group is both scheduled and excluded")
        for group, reason in C.DRIP_EXCLUDED.items():
            self.assertTrue(reason.strip(), f"{group} is excluded with no reason")

    def test_an_excluded_group_is_never_offered_by_the_drip(self):
        # Asserted through drip() rather than by reading DRIP_ORDER, because the
        # thing that matters is which bodies reach submit_one.
        offered = []
        C.lane_bodies = lambda group, gpus_per_pod: (offered.append(group) or [])
        C.reconcile = lambda dry_run=True: 0
        C.append_ledger({"state": "genesis", "at": "t"})
        C.drip(dry_run=False)
        for group in C.DRIP_EXCLUDED:
            self.assertNotIn(group, offered)
        self.assertIn("e1-mmlu", offered)

    def test_stop_excluded_targets_only_the_excluded_suites(self):
        # The hazard is a prefix match that also catches a suite we still want:
        # "lrwkv-e1-mmlu-..." must survive while "lrwkv-e1-gsm8k-..." is stopped.
        jobs = [
            {"name": "lrwkv-e1-gsm8k-a04-a1-s17-s0of8", "job_id": "j1",
             "gpu_count": 1, "status": "job_running"},
            {"name": "lrwkv-e1-mmlu-test-a04-a1-s17-s0of8", "job_id": "j2",
             "gpu_count": 1, "status": "job_running"},
        ]
        C.live_census = lambda: {"jobs": jobs, "live_gpus_submitted": 2,
                                 "live_gpus_running": 2}
        stopped = []
        C.stop_job = lambda job_id: (stopped.append(job_id) or (True, ""))
        C.append_ledger({"state": "genesis", "at": "t"})
        C.stop_excluded(dry_run=False)
        self.assertEqual(stopped, ["j1"])
        rows = [r for r in C.read_ledger() if r.get("state") == "stopped"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "stop_excluded")
        self.assertIn("gsm8k", rows[0]["reason"])

    def test_stop_excluded_writes_nothing_on_a_dry_run(self):
        jobs = [{"name": "lrwkv-e1-gsm8k-a04-a1-s17-s0of8", "job_id": "j1",
                 "gpu_count": 1, "status": "job_running"}]
        C.live_census = lambda: {"jobs": jobs, "live_gpus_submitted": 1,
                                 "live_gpus_running": 1}
        stopped = []
        C.stop_job = lambda job_id: (stopped.append(job_id) or (True, ""))
        C.append_ledger({"state": "genesis", "at": "t"})
        C.stop_excluded(dry_run=True)
        self.assertEqual(stopped, [])
        self.assertEqual([r for r in C.read_ledger()
                          if r.get("state") == "stopped"], [])

    def test_a_capacity_stop_in_one_lane_does_not_stop_the_other(self):
        # The two specs bill DIFFERENT projects, and the 16/40 quota is per project.
        # Inferring "full" for project-160ccb20 from a refusal by project-632c8db8
        # would leave the whole second lane's sanctioned capacity idle -- which is
        # exactly the capacity the operator asked us to use "for more".
        two = [{"name": f"m{i}", "description": "d"} for i in range(3)]
        C.lane_bodies = lambda group, gpus_per_pod: (
            [] if group != C.DRIP_ORDER[0] else
            (self._bodies if gpus_per_pod == 1 else two))
        C.append_ledger({"state": "genesis", "at": "t"})
        calls = self._fake_create([(0, "Error:\n Message: 总项目配额不足\n")])
        C.drip(dry_run=False)
        # 1 refused call in lane 1, then all three of lane 2 still attempted.
        self.assertEqual(len(calls), 4, "the second lane was skipped")
        submitted = {r["name"] for r in self.rows() if r["state"] == "submitted"}
        self.assertEqual(submitted, {"m0", "m1", "m2"})

    def test_the_dedicated_lane_is_filled_before_the_shared_lane(self):
        # Operator order 2026-09-20: project-632c8db8 first, project-160ccb20 "for
        # more". Reversing it would spend the shared project while the dedicated one
        # sat idle. The lane list is spelled out rather than read from LANE_SHAPES so
        # that a silent reorder of the constant fails here.
        self.assertEqual(C.LANE_SHAPES, (1, 8))
        order = []
        C.lane_bodies = lambda group, gpus_per_pod: (
            [] if group != C.DRIP_ORDER[0] else
            [{"name": f"lane{gpus_per_pod}", "description": "d"}])
        saved = C.submit_one
        C.submit_one = lambda body, dry_run: (order.append(body["name"])
                                              or {"name": body["name"],
                                                  "state": "submitted"})
        self.addCleanup(lambda: setattr(C, "submit_one", saved))
        C.append_ledger({"state": "genesis", "at": "t"})
        C.drip(dry_run=False)
        self.assertEqual(order, ["lane1", "lane8"])


    def test_the_second_lane_is_built_after_the_first_one_submits(self):
        # The shape lock only protects the wide lane if that lane's pending list is
        # built AFTER the narrow lane submitted: the lock reads the ledger, and the
        # narrow rows are what it reads. Computing both lists up front would offer
        # group X as 8-GPU pods after X's shards went out as 1-GPU jobs -- the same
        # shard billed twice under two names no name check can relate.
        order = []
        C.lane_bodies = lambda group, gpus_per_pod: (
            [] if group != C.DRIP_ORDER[0] else
            (order.append(f"build{gpus_per_pod}") or
             [{"name": f"lane{gpus_per_pod}-s0of8", "description": "d"}]))
        saved = C.submit_one
        C.submit_one = lambda body, dry_run: (order.append(f"submit:{body['name']}")
                                              or {"name": body["name"],
                                                  "state": "submitted"})
        self.addCleanup(lambda: setattr(C, "submit_one", saved))
        C.append_ledger({"state": "genesis", "at": "t"})
        C.drip(dry_run=False)
        self.assertEqual(order, ["build1", "submit:lane1-s0of8",
                                 "build8", "submit:lane8-s0of8"])


class TestGroupShapeLock(LedgerFixture):
    """A shard group must not change pod shape while any of its rows is live.

    Name de-duplication cannot see this hazard: the same shard is covered under
    ``-s1of8`` at 1 GPU and under ``-p0of4x2`` at 2 GPU, so both names are "new".
    """

    def _row(self, name, state):
        C.append_ledger({"state": state, "name": name, "at": "t"})

    def test_a_live_one_gpu_group_pins_the_shape(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self._row("lrwkv-e1-mmlu-test-x-s0of8", "submitted")
        self.assertEqual(C.shard_shape_in_ledger("lrwkv-e1-mmlu-test-x"), 1)

    def test_an_all_failed_group_is_free_to_be_reshaped(self):
        # A refusal holds no job, so nothing would be duplicated. Pinning the shape
        # off a failed row would strand the group in a lane that had no room.
        C.append_ledger({"state": "genesis", "at": "t"})
        for s in range(8):
            self._row(f"lrwkv-e1-mmlu-test-y-s{s}of8", "failed")
        self.assertIsNone(C.shard_shape_in_ledger("lrwkv-e1-mmlu-test-y"))

    def test_a_partially_submitted_group_still_pins(self):
        # The live case measured 2026-09-20: shards 0-2 submitted, 3-7 refused.
        C.append_ledger({"state": "genesis", "at": "t"})
        for s in range(3):
            self._row(f"lrwkv-e1-mmlu-test-z-s{s}of8", "submitted")
        for s in range(3, 8):
            self._row(f"lrwkv-e1-mmlu-test-z-s{s}of8", "failed")
        self.assertEqual(C.shard_shape_in_ledger("lrwkv-e1-mmlu-test-z"), 1)

    def test_check_group_shape_refuses_the_change(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self._row("lrwkv-e1-mmlu-test-z-s0of8", "submitted")
        bodies = [{"name": "lrwkv-e1-mmlu-test-z-p0of4x2"}]
        with self.assertRaises(SystemExit) as cm:
            C.check_group_shape(bodies, 2)
        self.assertIn("bill twice", str(cm.exception))

    def test_check_group_shape_allows_the_same_shape(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self._row("lrwkv-e1-mmlu-test-z-s0of8", "submitted")
        C.check_group_shape([{"name": "lrwkv-e1-mmlu-test-z-s1of8"}], 1)

    def test_a_mixed_shape_history_refuses_rather_than_guesses(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self._row("lrwkv-e1-mmlu-test-z-s0of8", "submitted")
        self._row("lrwkv-e1-mmlu-test-z-p1of4x2", "submitted")
        with self.assertRaises(SystemExit) as cm:
            C.shard_shape_in_ledger("lrwkv-e1-mmlu-test-z")
        self.assertIn("more than one pod shape", str(cm.exception))

    def test_a_name_without_a_topology_suffix_is_refused(self):
        # A body whose group cannot be identified cannot be shape-checked, so
        # accepting it would silently disable the gate for that body.
        C.append_ledger({"state": "genesis", "at": "t"})
        with self.assertRaises(SystemExit) as cm:
            C.check_group_shape([{"name": "lrwkv-q1probe-a29-c6loop"}], 1)
        self.assertIn("no topology suffix", str(cm.exception))

    def test_the_suffix_pattern_matches_what_the_emitter_writes(self):
        # The gate reads names the emitter produces; a drifted pattern would match
        # nothing and the gate would pass permissively on every group.
        for gpus in (1, 2):
            for body in E.mmlu_group("a29_c6loop_s20500", num_shards=8,
                                     gpus_per_pod=gpus):
                self.assertIsNotNone(C.TOPOLOGY_SUFFIX.match(body["name"]),
                                     body["name"])

    def test_a_lane_skips_a_shape_locked_track_instead_of_aborting(self):
        # Inside a tick, "this group belongs to the other lane" must not abort the
        # tick: that would stop every later track on one locked group.
        saved = C.group_bodies
        seen_tag = []

        def fake(group, track, shards, split, gpus_per_pod=1, tag="e1"):
            seen_tag.append(tag)
            if track == "locked":
                raise SystemExit("shape lock")
            return [{"name": f"{track}-p0of1x8", "description": "d"}]

        C.group_bodies = fake
        self.addCleanup(lambda: setattr(C, "group_bodies", saved))
        saved_ids = C.q1_track_ids
        C.q1_track_ids = lambda: ["locked", "ok"]
        self.addCleanup(lambda: setattr(C, "q1_track_ids", saved_ids))
        saved_merged = E.merged_present
        E.merged_present = lambda track, *a, **k: False
        self.addCleanup(lambda: setattr(E, "merged_present", saved_merged))
        saved_probe = E.require_probe
        E.require_probe = lambda track, *a, **k: "stub-receipt"
        self.addCleanup(lambda: setattr(E, "require_probe", saved_probe))
        got = C.lane_bodies("e1-mmlu", 8)
        self.assertEqual([b["name"] for b in got], ["ok-p0of1x8"])
        # And the wide lane really is the re-tagged one -- a body built under `e1`
        # would collide with the shape-locked groups this lane exists to avoid.
        self.assertEqual(set(seen_tag), {E.TAG_8GPU})

    def test_the_narrow_mmlu_lane_is_retired(self):
        # The 1-GPU MMLU lane would re-run, on project-632c8db8, the same shards the
        # wide lane now covers on project-160ccb20 -- and that first project is the
        # one the excluded GSM8K suite is saturating. Nothing, not even a track with
        # no reading, is offered there.
        self.assertEqual(C.lane_bodies("e1-mmlu", 1), [])

    def test_the_wide_mmlu_lane_covers_only_tracks_without_a_merged_reading(self):
        # Re-running a merged track spends ~8 GPU-h to re-derive an identical number,
        # so the selector is the merge receipt on disk, not "every track".
        saved = C.group_bodies
        C.group_bodies = lambda group, track, shards, split, gpus_per_pod=1, tag="e1": (
            [{"name": f"{track}-p0of1x8", "description": "d"}])
        self.addCleanup(lambda: setattr(C, "group_bodies", saved))
        saved_ids = C.q1_track_ids
        C.q1_track_ids = lambda: ["merged", "unmerged"]
        self.addCleanup(lambda: setattr(C, "q1_track_ids", saved_ids))
        saved_merged = E.merged_present
        E.merged_present = lambda track, *a, **k: track == "merged"
        self.addCleanup(lambda: setattr(E, "merged_present", saved_merged))
        saved_probe = E.require_probe
        E.require_probe = lambda track, *a, **k: "stub-receipt"
        self.addCleanup(lambda: setattr(E, "require_probe", saved_probe))
        got = C.lane_bodies("e1-mmlu", 8)
        self.assertEqual([b["name"] for b in got], ["unmerged-p0of1x8"])


class TestOwnAllowanceIsEnforcedBeforeCreateJob(LedgerFixture):
    """The operator's 64-GPU ceiling is ours to enforce; the scheduler's is not.

    Measured 2026-09-20: a tick whose only brake was the capacity refusal put 390
    GPUs in flight, because ``PROJECT_SHARED`` has a much larger quota than
    ``PROJECT_1GPU``'s 16/40 and accepted all 172 pods without complaint.
    """

    def setUp(self):
        super().setUp()
        self._saved_census = C.live_census
        saved_r = C.reconcile
        C.reconcile = lambda dry_run=True: 0
        self.addCleanup(lambda: setattr(C, "reconcile", saved_r))
        self._submitted = []
        saved_s = C.submit_one
        C.submit_one = lambda body, dry_run: (
            self._submitted.append(body["name"])
            or {"name": body["name"], "state": "submitted"})
        self.addCleanup(lambda: setattr(C, "submit_one", saved_s))

    def _lane(self, per_lane):
        saved = C.lane_bodies
        C.lane_bodies = lambda group, gpus_per_pod: (
            [] if group != C.DRIP_ORDER[0] else
            [{"name": f"g{gpus_per_pod}-{i}-s0of8", "description": "d"}
             for i in range(per_lane)])
        self.addCleanup(lambda: setattr(C, "lane_bodies", saved))

    def test_a_full_allowance_submits_nothing(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self.stub_census(submitted=C.OWN_LIVE_GPU_ALLOWANCE)
        self._lane(10)
        C.drip(dry_run=False)
        self.assertEqual(self._submitted, [],
                         "submitted past the operator's own ceiling")

    def test_submission_stops_exactly_at_the_allowance(self):
        C.append_ledger({"state": "genesis", "at": "t"})
        self.stub_census(submitted=C.OWN_LIVE_GPU_ALLOWANCE - 3)
        self._lane(10)
        C.drip(dry_run=False)
        # Three 1-GPU pods fit in lane 1; lane 2 needs 2 GPUs and gets none.
        self.assertEqual(len(self._submitted), 3, self._submitted)

    def test_a_wide_pod_is_charged_all_eight_gpus_against_the_allowance(self):
        # Charging per JOB instead of per GPU is how a wide lane multiplies the real
        # spend while the counter still reads under budget. Expressed at the 8-GPU
        # lane (the only wide lane since 2026-09-21): with 12 GPUs of headroom exactly
        # ONE 8-GPU pod fits, and a per-job counter would have admitted twelve.
        C.append_ledger({"state": "genesis", "at": "t"})
        self.stub_census(submitted=C.OWN_LIVE_GPU_ALLOWANCE - 12)
        saved = C.lane_bodies
        C.lane_bodies = lambda group, gpus_per_pod: (
            [] if group != C.DRIP_ORDER[0] or gpus_per_pod != 8 else
            [{"name": f"t-{i}-p0of1x8", "description": "d"} for i in range(10)])
        self.addCleanup(lambda: setattr(C, "lane_bodies", saved))
        C.drip(dry_run=False)
        self.assertEqual(len(self._submitted), 1, self._submitted)

    def test_headroom_never_goes_negative(self):
        self.assertEqual(C.allowance_headroom(
            {"live_gpus_submitted": C.OWN_LIVE_GPU_ALLOWANCE + 300}), 0)

    def test_the_allowance_is_the_users_number(self):
        # 32 is the updated theory-MVP allowance granted 2026-09-21.
        # It is deliberately NOT derived from the scheduler quota: the two are
        # different claims and the smaller one is the one we must respect.
        self.assertEqual(C.OWN_LIVE_GPU_ALLOWANCE, 32)
        self.assertGreater(C.OWN_LIVE_GPU_ALLOWANCE, C.RUNNING_GPU_QUOTA)

    def test_queuing_jobs_count_as_live(self):
        # A queuing job consumes submitted quota and will consume cards with no
        # further action, so a census that counted only `job_running` would report
        # 128 where the truth was 390 and the allowance check would pass wrongly.
        for status in ("job_queuing", "job_creating", "job_pending",
                       "job_restarting", "job_running"):
            self.assertIn(status, C.LIVE_STATUSES)
        jobs = [{"name": "lrwkv-a-s0of8", "status": "job_queuing", "gpu_count": 2,
                 "job_id": "job-1", "project_id": "p"},
                {"name": "lrwkv-a-s1of8", "status": "job_running", "gpu_count": 2,
                 "job_id": "job-2", "project_id": "p"}]
        got = self._saved_census(jobs)
        self.assertEqual(got["live_gpus_submitted"], 4)
        self.assertEqual(got["live_gpus_running"], 2)

    def test_the_census_counts_granted_gpus_not_the_requested_spec(self):
        # gpu_count is what the scheduler GRANTED. This project has already seen a
        # task declare 1 GPU and be granted 8, so re-deriving the count from the
        # spec id would under-report exactly when it matters.
        jobs = [{"name": "lrwkv-a-s0of8", "status": "job_running", "gpu_count": 8,
                 "job_id": "job-1", "project_id": "p"}]
        self.assertEqual(self._saved_census(jobs)["live_gpus_submitted"], 8)

    def test_foreign_jobs_are_not_counted_against_our_allowance(self):
        # Other campaigns consume the same project quota but are not ours to stop or
        # to be charged for.
        jobs = [{"name": "ablation-new-2", "status": "job_running", "gpu_count": 8,
                 "job_id": "job-x", "project_id": "p"}]
        self.assertEqual(self._saved_census(jobs)["live_gpus_submitted"], 0)


class TestStopExcessDropsWholeGroups(unittest.TestCase):
    """A partially stopped group spends its survivors' GPU-hours and never merges."""

    def _jobs(self, spec):
        out = []
        for group, (n, gpus, running) in spec.items():
            for i in range(n):
                out.append({"name": f"{group}-s{i}of{n}", "job_id": f"job-{group}-{i}",
                            "status": "job_running" if i < running else "job_queuing",
                            "gpu_count": gpus, "project_id": "p"})
        return out

    def test_the_kept_set_never_exceeds_the_allowance(self):
        for total in (10, 40, 60, 100):
            spec = {f"g{i}": (4, 2, 0) for i in range(total)}
            kept = self._plan(self._jobs(spec))["gpus_kept"]
            self.assertLessEqual(kept, C.OWN_LIVE_GPU_ALLOWANCE, f"{total} groups")

    def _plan(self, jobs):
        saved = C.live_census
        C.live_census = lambda j=None: {
            "live_jobs": len(jobs),
            "live_gpus_submitted": sum(x["gpu_count"] for x in jobs),
            "live_gpus_running": sum(x["gpu_count"] for x in jobs
                                     if x["status"] == "job_running"),
            "jobs": jobs}
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                C.stop_excess(dry_run=True)
            return json.loads(buf.getvalue())
        finally:
            C.live_census = saved

    def test_a_group_is_kept_or_dropped_entirely(self):
        # The unit is the group because the merge gate waits for NUM_SHARDS files.
        jobs = self._jobs({f"g{i}": (8, 1, 0) for i in range(20)})
        plan = self._plan(jobs)
        kept = {g["group"] for g in plan["groups_kept"]}
        # Every kept group contributes all 8 of its GPUs, so the total is a multiple
        # of 8 -- a partially-kept group would break that.
        self.assertEqual(plan["gpus_kept"] % 8, 0)
        self.assertEqual(plan["gpus_kept"], 8 * len(kept))

    def test_the_paper_priority_groups_are_kept_first(self):
        spec = {g: (8, 1, 0) for g in C.KEEP_PRIORITY[:4]}
        spec.update({f"filler{i}": (8, 1, 8) for i in range(20)})
        plan = self._plan(self._jobs(spec))
        kept = {g["group"] for g in plan["groups_kept"]}
        for g in C.KEEP_PRIORITY[:4]:
            self.assertIn(g, kept, "a headline E1 group was dropped for filler")

    def test_a_running_group_outranks_a_queuing_one_among_the_rest(self):
        # A queuing job has spent nothing, so it is the cheapest thing to give back.
        spec = {"runner": (8, 1, 8), "queuer": (8, 1, 0)}
        plan = self._plan(self._jobs(spec))
        self.assertLess(C.keep_rank("runner", 8), C.keep_rank("queuer", 0))
        self.assertEqual(plan["gpus_kept"], 16)   # both fit under 64

    def test_nothing_is_dropped_when_already_under_the_allowance(self):
        plan = self._plan(self._jobs({"g0": (4, 1, 4)}))
        self.assertEqual(plan["groups_dropped"], 0)
        self.assertEqual(plan["gpus_freed"], 0)

    def test_a_dry_plan_stops_nothing(self):
        plan = self._plan(self._jobs({f"g{i}": (8, 1, 0) for i in range(20)}))
        self.assertGreater(plan["groups_dropped"], 0)
        self.assertEqual(plan["jobs_stopped"], 0, "a dry plan called StopJob")

    def test_every_keep_priority_name_is_a_real_group_name(self):
        # A typo here silently demotes a headline group to filler, and the cut would
        # drop the arm the paper's table is built on.
        for name in C.KEEP_PRIORITY:
            self.assertIsNotNone(C.TOPOLOGY_SUFFIX.match(f"{name}-s0of8"), name)
            track = name.replace("lrwkv-e1-mmlu-test-", "").replace("-", "_")
            self.assertTrue(
                any(t.track_id == track for t in tracks.ALL_TRACKS),
                f"{name} maps to unknown track {track!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
