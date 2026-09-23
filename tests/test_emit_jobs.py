"""CPU tests for the qz emitter.

These target the failure modes that are *silent on the cluster*: a body that
schedules a model with no loader, a shard group that scores one shard k times, a
spec paired with the wrong project, and a command that mangles in the pod.  Each
of those returns numbers rather than an error, which is why they have to be caught
here.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

QZ_DIR = Path(__file__).resolve().parent.parent / "qz"
sys.path.insert(0, str(QZ_DIR))
sys.path.insert(0, str(QZ_DIR.parent))

import emit_jobs as E  # noqa: E402
from lrwkv_evidence import tracks  # noqa: E402

#: A BiRWKV track whose tokenizer tree exists -- the common case.
BIRWKV = "a29_c6loop_s20500"


class TestFrameworkConfig(unittest.TestCase):
    def test_exactly_five_keys(self):
        body = E.probe_body(BIRWKV)
        self.assertEqual(set(body["framework_config"][0]),
                         set(E.ACCEPTED_FRAMEWORK_KEYS))

    def test_spec_derives_shm_project_and_wall(self):
        body = E.probe_body(BIRWKV)
        self.assertEqual(body["framework_config"][0]["spec_id"], E.SPEC_1GPU)
        self.assertEqual(body["framework_config"][0]["shm_gi"], 400)
        self.assertEqual(body["project_id"], E.PROJECT_1GPU)
        # A string, not an int: the API rejects the int and --dry-run cannot see it.
        self.assertIsInstance(body["max_running_time_ms"], str)
        self.assertEqual(body["max_running_time_ms"], str(E.TIMEOUT_1GPU_MS))

    def test_unknown_spec_is_refused(self):
        with self.assertRaises(SystemExit):
            E.make_body("n", "bash -lc 'true'", "d", spec_id="some-other-spec")

    def test_gpu_cap_is_enforced(self):
        # The cap is the operator's per-job number, 8 as of 2026-09-21 ("use 8h100 for
        # each benchmark / for each job"), raised from 3. Asserted explicitly so that a
        # regression to the old value fails here rather than at the next CreateJob.
        self.assertEqual(E.MAX_GPUS_PER_JOB, 8)
        # 8 GPU/pod x 2 pods = 16, over the cap.
        with self.assertRaises(SystemExit):
            E.make_body("n", "bash -lc 'true'", "d",
                        spec_id=E.SPEC_8GPU, instance_count=2)
        # Exactly the cap is allowed: the cap is a real bound, not an accident of
        # every caller passing 1 pod.
        body = E.make_body("n", "bash -lc 'true'", "d",
                           spec_id=E.SPEC_8GPU, instance_count=1)
        self.assertEqual(body["framework_config"][0]["spec_id"], E.SPEC_8GPU)


class TestCommandValidation(unittest.TestCase):
    def test_accepts_the_real_bodies(self):
        for body in E.mmlu_group(BIRWKV, num_shards=4) + [E.probe_body(BIRWKV)]:
            E.validate_command(body["command"])

    def test_refuses_unbalanced_quote(self):
        with self.assertRaises(SystemExit):
            E.validate_command("bash -lc 'echo it's broken'")

    def test_refuses_double_dollar(self):
        # $$ reaches bash as a bare $ after the launcher's own expansion; no
        # `bash -lc` drill on the login node reproduces it.
        with self.assertRaises(SystemExit):
            E.validate_command("bash -lc 'echo $$'")

    def test_refuses_non_bash_lc(self):
        with self.assertRaises(SystemExit):
            E.validate_command("python train.py")


class TestLoaderGating(unittest.TestCase):
    def test_no_loader_tracks_are_refused(self):
        absent = [t.track_id for t in tracks.ALL_TRACKS
                  if t.model_kind == tracks.NO_LOADER]
        self.assertTrue(absent, "the registry should still mark the two absent baselines")
        for track_id in absent:
            with self.assertRaises(SystemExit):
                E.mmlu_body(track_id, 0, 1)

    def test_llada_is_routed_away_from_run_eval(self):
        llada = [t.track_id for t in tracks.ALL_TRACKS
                 if t.model_kind == tracks.LLADA_MODEL_KIND]
        for track_id in llada:
            with self.assertRaises(SystemExit):
                E.mmlu_body(track_id, 0, 1)

    def test_unknown_tokenizer_tree_is_refused(self):
        # Scoring a checkpoint against another tokenizer's label ids would score
        # four arbitrary tokens and return a plausible-looking accuracy.
        bad = tracks.Track(
            track_id="fake", track="released", paper_arm="C0",
            local_label="fake", rel_path="models/nonexistent",
            model_dir="models/nonexistent", sha256="", params_stored=1,
            params_executed=1, tokens_seen=-1, training_seed=None, step=None,
            loop_reps=0, variant="", model_kind="hf_causal")
        with self.assertRaises(SystemExit):
            E.mmlu_task_dir(bad, "test")

    def test_every_gated_track_has_a_tree(self):
        # NOT gated on UNSCHEDULABLE_KINDS: a MISSING_RUNTIME_DEP row still needs
        # its tree registered, because the only thing standing between it and a
        # submission is an environment fix -- and discovering the missing tree then
        # would mean re-deriving which tokenizer's label ids it needs. The two
        # kinds skipped here are skipped for reasons no tree can repair: NO_LOADER
        # has no code path at all, and LLaDA scores multichoice through
        # llada_batch_eval.py, which does not read these trees.
        for t in tracks.ALL_TRACKS:
            if t.model_kind in (tracks.NO_LOADER, tracks.LLADA_MODEL_KIND):
                continue
            self.assertIn(t.model_dir, E.TOKENIZER_TREE,
                          f"{t.track_id} is schedulable but has no MMLU tree")


class TestShardCoverage(unittest.TestCase):
    def test_group_tiles_the_shard_range(self):
        for n in (1, 4, 8):
            bodies = E.mmlu_group(BIRWKV, num_shards=n)
            self.assertEqual(len(bodies), n)
            ranks = sorted(int(E.exported(b["command"], "NODE_RANK")) for b in bodies)
            self.assertEqual(ranks, list(range(n)))

    def test_nnodes_equals_shard_count(self):
        # At NNODES=1 the launcher ignores NODE_RANK and hard-assigns 0, so a
        # group that got this wrong would run shard 0 in every pod.
        for b in E.mmlu_group(BIRWKV, num_shards=8):
            self.assertEqual(E.exported(b["command"], "NNODES"), "8")
            self.assertEqual(E.exported(b["command"], "NGPUS"), "1")

    def test_duplicate_rank_is_refused(self):
        bodies = E.mmlu_group(BIRWKV, num_shards=4)
        bodies[3]["command"] = bodies[3]["command"].replace(
            'export NODE_RANK="3"', 'export NODE_RANK="0"')
        with self.assertRaises(SystemExit):
            E.check_shard_coverage(bodies, 4)

    def test_missing_rank_export_is_refused(self):
        bodies = E.mmlu_group(BIRWKV, num_shards=2)
        bodies[1]["command"] = bodies[1]["command"].replace(
            'export NODE_RANK="1"; ', "")
        with self.assertRaises(SystemExit):
            E.check_shard_coverage(bodies, 2)

    def test_mismatched_num_shards_is_refused(self):
        bodies = E.mmlu_group(BIRWKV, num_shards=4)
        for b in bodies:
            b["command"] = b["command"].replace(
                'export NUM_SHARDS="4"', 'export NUM_SHARDS="5"')
        with self.assertRaises(SystemExit):
            E.check_shard_coverage(bodies, 4)

    def test_names_are_distinct(self):
        names = [b["name"] for b in E.mmlu_group(BIRWKV, num_shards=8)]
        self.assertEqual(len(set(names)), 8)


class TestTheWideLoadSlots(unittest.TestCase):
    """The load gate, and the shape whose measured cgroup justifies lifting it."""

    def test_only_the_eight_gpu_shape_lifts_the_gate(self):
        # Measured in-pod 2026-09-21: the 8-GPU pod's cgroup is 1.93 TB, which is the
        # case load_gate.py names as the one to raise the slots for. The 2-GPU shape is
        # a different quota row with an unmeasured cgroup, so it must NOT inherit this --
        # hence an exact set rather than a `gpus_per_pod > 1` predicate.
        self.assertEqual(E.WIDE_SHAPES, frozenset({E.SPEC_8GPU}))
        self.assertEqual(E.WIDE_LOAD_SLOTS, 8)

    def test_the_wide_body_exports_the_slot_count_and_the_narrow_ones_do_not(self):
        builders = [
            ("aux", lambda gp: E.aux_body("hellaswag", "a29_c6loop_s20500", 0, 8, gp)),
            ("mmlu", lambda gp: E.mmlu_body("a29_c6loop_s20500", 0, 8, tag="e1b",
                                            gpus_per_pod=gp)),
        ]
        for name, builder in builders:
            self.assertEqual(
                E.exported(builder(8)["command"], "BIRWKV_LOAD_SLOTS"), "8",
                f"the 8-GPU {name} body would serialize loads at 3 and idle its cards")
            for narrow in (1, 2):
                self.assertIsNone(
                    E.exported(builder(narrow)["command"], "BIRWKV_LOAD_SLOTS"),
                    f"the {narrow}-GPU {name} body inherited a wider pod's load budget")

    def test_the_body_carries_the_slot_count_in_its_env_exports(self):
        # The launcher reads BIRWKV_LOAD_SLOTS from the environment it runs under, so it
        # has to be an `export` in the command -- a body-level envs[] entry would be a
        # different mechanism, and the gate would silently keep its default.
        body = E.aux_body("hellaswag", "a29_c6loop_s20500", 0, 8, 8)
        self.assertIn('export BIRWKV_LOAD_SLOTS="8"', body["command"])


class TestTwoGpuPods(unittest.TestCase):
    """The 2-GPU shape must cover the same shards, in a different project.

    Its point is capacity, not pod size: the 16-running/40-submitted quota is
    per-project and the two specs route to different projects, so these bodies are
    additional concurrency rather than a rearrangement of the 1-GPU lane.
    """

    def test_a_two_gpu_group_still_covers_every_shard(self):
        # The launcher runs NGPUS workers per pod, worker w = NODE_RANK*NGPUS+g,
        # stride NNODES*NGPUS. So 4 pods x 2 GPU must cover shards 0..7 exactly
        # once -- the thing that would silently break is coverage, not the API call.
        for n in (2, 4, 8):
            bodies = E.mmlu_group(BIRWKV, num_shards=n, gpus_per_pod=2)
            self.assertEqual(len(bodies), n // 2)
            covered = []
            for b in bodies:
                rank = int(E.exported(b["command"], "NODE_RANK"))
                ngpus = int(E.exported(b["command"], "NGPUS"))
                nnodes = int(E.exported(b["command"], "NNODES"))
                world = nnodes * ngpus
                for g in range(ngpus):
                    w = rank * ngpus + g
                    covered.extend(range(w, n, world))
            self.assertEqual(sorted(covered), list(range(n)),
                             f"{n} shards at 2 GPU/pod do not tile exactly once")

    def test_num_shards_still_counts_shards_not_pods(self):
        # NUM_SHARDS is also the count the merge gate waits for. A group that wrote
        # the pod count here would produce real numbers over half the benchmark and
        # never merge.
        for b in E.mmlu_group(BIRWKV, num_shards=8, gpus_per_pod=2):
            self.assertEqual(E.exported(b["command"], "NUM_SHARDS"), "8")
            self.assertEqual(E.exported(b["command"], "NNODES"), "4")

    def test_the_two_gpu_shape_routes_to_the_other_project(self):
        one = E.mmlu_group(BIRWKV, num_shards=8, gpus_per_pod=1)[0]
        two = E.mmlu_group(BIRWKV, num_shards=8, gpus_per_pod=2)[0]
        self.assertEqual(one["framework_config"][0]["spec_id"], E.SPEC_1GPU)
        self.assertEqual(two["framework_config"][0]["spec_id"], E.SPEC_2GPU)
        self.assertEqual(one["project_id"], E.PROJECT_1GPU)
        self.assertEqual(two["project_id"], E.PROJECT_SHARED)
        # If both shapes billed the same project the 2-GPU lane would buy nothing:
        # it would consume two of the same 16 running GPUs per job.
        self.assertNotEqual(one["project_id"], two["project_id"])

    def test_one_gpu_names_are_unchanged(self):
        # 27 MMLU shards are already in flight under the s<k>of<n> names. Renaming
        # them would make every one read as unsubmitted and bill a duplicate.
        names = [b["name"] for b in E.mmlu_group(BIRWKV, num_shards=8)]
        self.assertTrue(all(nm.endswith(f"-s{i}of8") for i, nm in enumerate(names)),
                        names)

    def test_a_two_gpu_name_cannot_collide_with_a_one_gpu_name(self):
        one = {b["name"] for b in E.mmlu_group(BIRWKV, num_shards=8)}
        two = {b["name"] for b in E.mmlu_group(BIRWKV, num_shards=8, gpus_per_pod=2)}
        self.assertEqual(one & two, set())

    def test_an_indivisible_shard_count_is_refused(self):
        # 5 shards over 2-GPU pods cannot tile: the launcher's stride would leave
        # the tail unscored, and the failure is silent (real numbers, wrong subset).
        with self.assertRaises(SystemExit) as cm:
            E.mmlu_group(BIRWKV, num_shards=5, gpus_per_pod=2)
        self.assertIn("do not divide", str(cm.exception))

    def test_an_unsanctioned_pod_width_is_refused(self):
        # 4 and 8 left this tuple on 2026-09-21: 4 rides the 8-GPU quota and 8 IS the
        # sanctioned wide shape.  The widths that remain unsanctioned are the ones with
        # no quota behind them -- the scheduler grants a whole shape, so a 5-card ask
        # books eight and computes on five.
        for width in (3, 5, 6, 7):
            with self.assertRaises(SystemExit):
                E.mmlu_body(BIRWKV, 0, 8, gpus_per_pod=width)
        # And the wide shape is now accepted, so this cannot pass by refusing
        # everything.
        self.assertEqual(
            E.mmlu_body(BIRWKV, 0, 8, gpus_per_pod=8)["framework_config"][0]
            ["spec_id"], E.SPEC_8GPU)

    def test_coverage_check_accepts_the_two_gpu_group(self):
        # The pre-change check asserted the ranks tiled range(num_shards), which is
        # true only at 1 GPU/pod and would have rejected every correct 2-GPU group.
        bodies = E.mmlu_group(BIRWKV, num_shards=8, gpus_per_pod=2)
        E.check_shard_coverage(bodies, 8)   # must not raise

    def test_a_short_two_gpu_group_is_refused(self):
        # Dropping a pod leaves NNODES claiming more pods than were submitted, so
        # the tail shards are never run and the merge never fires.
        bodies = E.mmlu_group(BIRWKV, num_shards=8, gpus_per_pod=2)
        with self.assertRaises(SystemExit):
            E.check_shard_coverage(bodies[:-1], 8)

    def test_gen_bodies_also_take_the_two_gpu_shape(self):
        for suite, n in (("gsm8k", 8), ("humanevalplus", 4), ("mbppplus", 4)):
            bodies = E.gen_group(BIRWKV, suite, num_shards=n, gpus_per_pod=2)
            self.assertEqual(len(bodies), n // 2)
            self.assertEqual(bodies[0]["project_id"], E.PROJECT_SHARED)
            self.assertEqual(E.exported(bodies[0]["command"], "NUM_SHARDS"), str(n))
            # The code exports must survive the topology change.
            if suite != "gsm8k":
                self.assertEqual(E.exported(bodies[0]["command"], "CODE_MEM_MB"),
                                 str(E.CODE_MEM_MB))

    def test_a_two_gpu_body_is_still_under_the_per_job_ceiling(self):
        for b in E.mmlu_group(BIRWKV, num_shards=8, gpus_per_pod=2):
            fc = b["framework_config"][0]
            self.assertEqual(fc["instance_count"], 1)
            # The standing user constraint is fewer than 4 H100 per job; 2x1 = 2.
            self.assertLessEqual(2 * fc["instance_count"], E.MAX_GPUS_PER_JOB)


class TestCheckpointLayout(unittest.TestCase):
    """The loader reads <ckpt_dir>/model.pt; model_kind does not promise that exists."""

    #: A bundle-shaped track declared ``birwkv_diffusion``, which is what the
    #: registry held until the r04 arch mismatch was measured (2026-09-20).  The
    #: registry now marks those rows :data:`tracks.FOREIGN_ARCH`, so the container
    #: guards below have no live registry user -- but they are still the guards
    #: that stop a *future* bundle-shaped row from reaching the loader, so they are
    #: exercised against a synthetic track rather than deleted.  Deleting them
    #: because nothing currently trips them is how a guard silently stops working.
    @staticmethod
    def _bundle_tracks():
        return [dataclasses.replace(t, model_kind="birwkv_diffusion")
                for t in tracks.ALL_TRACKS
                if t.model_kind == tracks.FOREIGN_ARCH and t.path.is_file()]

    def test_flat_step_dirs_are_accepted(self):
        n = 0
        for t in tracks.ALL_TRACKS:
            if t.model_kind != "birwkv_diffusion":
                continue
            if not (t.path / "model.pt").is_file():
                continue
            E.check_loadable_layout(t)   # must not raise
            n += 1
        self.assertGreaterEqual(n, 22, "expected the 2.9B + task7 step dirs")

    def test_bundled_pt_files_route_to_a_verified_derived_dir(self):
        bundles = self._bundle_tracks()
        self.assertTrue(bundles, "the r04 rows are the only bundle-shaped ones on disk")
        for t in bundles:
            ckpt = E.eval_ckpt_dir(t)
            # The bundle itself is never handed to the loader: it would ask for
            # <file>.pt/model.pt, which cannot exist.
            self.assertNotEqual(ckpt, t.ckpt_arg)
            self.assertTrue(Path(ckpt, "model.pt").is_file(), f"{ckpt} has no model.pt")
            meta = json.loads(Path(ckpt, "meta.json").read_text())
            # The derived dir must descend from THIS track: these runs differ only
            # by seed, so a mixed-up dir would load, score, and report normally.
            self.assertEqual(meta["derived_from"], str(t.path))
            self.assertEqual(meta["derived_from_sha256"], t.sha256)

    def test_a_mislineaged_derived_dir_is_refused(self):
        bundles = self._bundle_tracks()
        if len(bundles) < 2:
            self.skipTest("need two bundles to swap")
        victim = bundles[0]
        meta_path = Path(E.eval_ckpt_dir(victim), "meta.json")
        original = meta_path.read_text()
        try:
            bad = json.loads(original)
            bad["derived_from"] = str(bundles[1].path)
            meta_path.write_text(json.dumps(bad))
            with self.assertRaises(SystemExit) as cm:
                E.eval_ckpt_dir(victim)
            # Name the other checkpoint, so the reader can see which two got
            # crossed rather than only that something was wrong.
            self.assertIn(str(bundles[1].path), str(cm.exception))
        finally:
            meta_path.write_text(original)

    def test_a_stale_source_sha_is_refused(self):
        bundles = self._bundle_tracks()
        if not bundles:
            self.skipTest("no bundles")
        victim = bundles[0]
        meta_path = Path(E.eval_ckpt_dir(victim), "meta.json")
        original = meta_path.read_text()
        try:
            bad = json.loads(original)
            bad["derived_from_sha256"] = "0" * 64
            meta_path.write_text(json.dumps(bad))
            with self.assertRaises(SystemExit):
                E.eval_ckpt_dir(victim)
        finally:
            meta_path.write_text(original)

    def test_a_converted_bundle_reaches_a_loadable_ckpt_dir(self):
        # The conversion works: a bundle-shaped birwkv_diffusion track resolves to
        # a dir the loader can open. This is about the CONTAINER only -- the r04
        # rows are refused one level up for their key layout, which a container
        # check cannot see (see TestForeignArch).
        for t in self._bundle_tracks():
            ckpt = E.eval_ckpt_dir(t)
            E.check_loadable_layout(t)
            self.assertTrue(Path(ckpt, "model.pt").is_file())

    def test_the_refusal_names_the_conversion_when_nothing_is_derived(self):
        # A bundle with no derived dir must refuse, and point at the fix rather
        # than merely reporting a missing path.
        fake = tracks.Track(
            track_id="r04_unconverted", track="adaptation_0p4b", paper_arm="C5",
            local_label="fake", rel_path="outputs_rwkv04b/runs/M6_adapt_s17/step_00001000.pt",
            model_dir="models/rwkv7-0.4B-world", sha256="", params_stored=-1,
            params_executed=453_938_176, tokens_seen=1_000_000, training_seed=17,
            step=1000, loop_reps=0, variant="", model_kind="birwkv_diffusion")
        if not fake.path.is_file():
            self.skipTest("the step_00001000 bundle is not on disk")
        with self.assertRaises(SystemExit) as cm:
            E.check_loadable_layout(fake)
        self.assertIn("convert_rwkv04b", str(cm.exception))

    def test_hf_causal_needs_no_step_dir(self):
        # hf_causal loads from --model_dir; --ckpt_dir is provenance only
        # (run_eval.py:250), so a layout requirement there would be a false refusal.
        hf = [t for t in tracks.ALL_TRACKS if t.model_kind == "hf_causal"]
        self.assertTrue(hf)
        for t in hf:
            E.check_loadable_layout(t)


class TestForeignArch(unittest.TestCase):
    """A loadable container is not a loadable model.

    Measured 2026-09-20: five r04 probes passed every submit-time check and then
    died in-pod on ``checkpoint/model mismatch: ['layers.0.attn_fwd.x_r', ...]``,
    one full 0.4B load each.  The container was right (``convert_rwkv04b`` had
    already fixed it) and the *key layout* was wrong, which no container check can
    see: the r04 trainer's ``TiedBiRWKV7Block`` stores one mixer per block as
    ``layers.N.attn.*``, while this harness builds untied ``attn_fwd``/``attn_bwd``.
    """

    def test_the_registry_marks_the_r04_lineage_foreign(self):
        foreign = [t for t in tracks.ALL_TRACKS
                   if t.model_kind == tracks.FOREIGN_ARCH]
        self.assertEqual(len(foreign), 5, "the five rwkv04b runs")
        for t in foreign:
            self.assertTrue(t.track_id.startswith("r04_"))

    def test_a_body_for_a_foreign_arch_track_is_refused(self):
        for t in tracks.ALL_TRACKS:
            if t.model_kind != tracks.FOREIGN_ARCH:
                continue
            for build in (lambda i=t.track_id: E.probe_body(i),
                          lambda i=t.track_id: E.mmlu_body(i, 0, 1)):
                with self.assertRaises(SystemExit) as cm:
                    build()
                msg = str(cm.exception)
                # The refusal must name the mechanism and the harness that CAN run
                # it: "unsupported" would read as "these weights are unusable",
                # which is false -- they have 16K/32K/64K long-context results.
                self.assertIn("attn_fwd", msg)
                self.assertIn("rwkv04b/longrwkv/eval", msg)

    def test_no_foreign_track_is_in_the_q1_order(self):
        import campaign  # noqa: PLC0415 -- same sys.path insert as E
        foreign = {t.track_id for t in tracks.ALL_TRACKS
                   if t.model_kind == tracks.FOREIGN_ARCH}
        self.assertFalse(foreign & set(campaign.q1_track_ids()))


class TestProbeGate(unittest.TestCase):
    def test_probe_only_is_set(self):
        self.assertEqual(E.exported(E.probe_body(BIRWKV)["command"], "PROBE_ONLY"), "1")

    def test_sweep_body_does_not_set_probe_only(self):
        self.assertEqual(E.exported(E.mmlu_body(BIRWKV, 0, 4)["command"],
                                    "PROBE_ONLY"), "0")

    def test_require_probe_refuses_without_a_receipt(self):
        # Existing production receipts must not turn this negative check into
        # a skip. Exercise the filesystem guard against isolated empty evidence.
        with tempfile.TemporaryDirectory() as output, patch.object(E, "OUTPUTS", output):
            probe = Path(E.probe_outdir(BIRWKV))
            for state in ("missing_directory", "empty_directory", "empty_receipt"):
                if state == "empty_directory":
                    probe.mkdir(parents=True)
                if state == "empty_receipt":
                    (probe / "test.probe.json").touch()
                with self.subTest(state=state), self.assertRaisesRegex(
                        SystemExit, "has no Q1 qualification receipt"):
                    E.require_probe(BIRWKV)


class TestTaskDirs(unittest.TestCase):
    def test_task_dirs_exist_on_disk(self):
        for tree in sorted(set(E.TOKENIZER_TREE.values())):
            for split in ("test", "validation"):
                d = Path(f"{E.MMLU_TASKS}/{tree}/{split}")
                self.assertTrue(d.is_dir(), f"missing task dir {d}")
                self.assertTrue(any(d.glob("*.npz")), f"no npz under {d}")

    def test_body_points_at_an_existing_dir(self):
        for b in E.mmlu_group(BIRWKV, num_shards=2):
            self.assertTrue(Path(E.exported(b["command"], "TASK_DIR")).is_dir())

    def test_conditions_is_a_scorer_the_harness_knows(self):
        # fwdce is parsed by multichoice.parse_condition; a typo would be an
        # unknown-condition crash per shard after the model has loaded.
        self.assertEqual(E.exported(E.mmlu_body(BIRWKV, 0, 1)["command"],
                                    "CONDITIONS"), "fwdce")


class TestConditionRouting(unittest.TestCase):
    """The scoring condition is an entry point, not a label.

    Measured 2026-09-20: five released-HF probes died with ``'bool' object has no
    attribute 'shape'`` because the emitter exported ``fwdce`` for every track.
    ``_score_fwdce`` (multichoice.py:211) calls ``model(ids, True)``, where the
    second positional is ``BiRWKV7ForMaskedDiffusion.forward``'s ``force_forward``
    flag -- on a plain HF module it binds to ``attention_mask``.  ``_score_raw``
    (:194-199) goes through ``backbone(input_ids=..., attention_mask=...)``, which
    is the same statistic through the entry point an HF model has.

    The dangerous half is the other direction: ``maskce`` parses for both kinds and
    computes a *different* statistic (masked-span, position-aligned, no shift), so a
    wrong-but-parseable condition returns plausible accuracies instead of crashing.
    Hence the condition is derived from ``model_kind`` and pinned here.
    """

    @staticmethod
    def _schedulable(kind):
        return [t for t in tracks.ALL_TRACKS
                if t.model_kind == kind and t.model_dir in E.TOKENIZER_TREE]

    def test_every_hf_causal_track_renders_raw(self):
        hf = self._schedulable("hf_causal")
        self.assertTrue(hf, "the released AR baselines should be schedulable")
        for t in hf:
            for body in (E.probe_body(t.track_id), E.mmlu_body(t.track_id, 0, 1)):
                self.assertEqual(E.exported(body["command"], "CONDITIONS"), "raw",
                                 f"{t.track_id} would pass True as attention_mask")

    def test_every_birwkv_track_renders_fwdce(self):
        bi = self._schedulable("birwkv_diffusion")
        self.assertTrue(bi)
        for t in bi:
            body = E.mmlu_body(t.track_id, 0, 1)
            self.assertEqual(E.exported(body["command"], "CONDITIONS"), "fwdce",
                             f"{t.track_id} is the causal-span comparability control")

    def test_the_two_kinds_do_not_share_a_condition(self):
        # If both mapped to the same string, the derivation would be decorative and
        # the defect above would be back without any test changing.
        self.assertNotEqual(E.CAUSAL_SPAN_CONDITION["hf_causal"],
                            E.CAUSAL_SPAN_CONDITION["birwkv_diffusion"])

    def test_an_unmapped_model_kind_is_refused_not_defaulted(self):
        unmapped = tracks.MODEL_KINDS - set(E.CAUSAL_SPAN_CONDITION)
        self.assertTrue(unmapped, "there should be kinds with no causal-span path")
        for kind in sorted(unmapped):
            bogus = tracks.Track(
                track_id="fake", track="released", paper_arm="C0",
                local_label="fake", rel_path="models/RWKV7-Goose-World3-2.9B-HF",
                model_dir="models/RWKV7-Goose-World3-2.9B-HF", sha256="",
                params_stored=1, params_executed=1, tokens_seen=-1,
                training_seed=None, step=None, loop_reps=0, variant="",
                model_kind=kind)
            with self.assertRaises(SystemExit) as cm:
                E.causal_span_condition(bogus)
            self.assertIn(kind, str(cm.exception))

    def test_the_rendered_condition_is_one_the_harness_parses(self):
        # parse_condition is the real gate; read it rather than restating its
        # vocabulary here, so a rename upstream fails this test instead of passing.
        src = Path(E.SCALE_DIR, "eval/capability/multichoice.py").read_text()
        for cond in set(E.CAUSAL_SPAN_CONDITION.values()):
            self.assertIn(f'"{cond}"', src, f"{cond} is not named in multichoice.py")


class TestGenerativeSuites(unittest.TestCase):
    """The math/code lane's own failure modes, none of which raise in the pod.

    Three that were measured, not imagined:

    * a resource kill is recorded as ``tests_failed`` (``_run_single_test`` returns
      ``(False, stderr, False)`` for any nonzero exit), so 4 EvalPlus rows are
      silently wrong at ``eval_code``'s 256 MiB / 5 s defaults;
    * MBPP+ through the ``humaneval`` assembly rule put prose into the executable
      program -- 100 % ``syntax_error``, which reads as a model that cannot code;
    * without the thread-cap preamble every numpy-importing test aborts at import,
      which scored the canonical solutions 0/40 and would be reported as 0 % pass@1.

    So the suite file, its item count, its assembly rule and its calibrated sandbox
    budget are all checked before a shard is scheduled.
    """

    def test_code_bodies_export_the_calibrated_budget(self):
        for suite in ("humanevalplus", "mbppplus"):
            cmd = E.gen_body(BIRWKV, suite, 0, 4)["command"]
            self.assertEqual(E.exported(cmd, "CODE_MEM_MB"), str(E.CODE_MEM_MB))
            self.assertEqual(E.exported(cmd, "CODE_TIMEOUT_S"), str(E.CODE_TIMEOUT_S))
            # The launcher's own defaults are 256/5.0; exporting nothing would
            # silently run at the setting the calibration refuted.
            self.assertNotEqual(E.exported(cmd, "CODE_MEM_MB"), "256")

    def test_code_bodies_export_the_assembly_rule(self):
        for suite in ("humanevalplus", "mbppplus"):
            cmd = E.gen_body(BIRWKV, suite, 0, 4)["command"]
            self.assertEqual(E.exported(cmd, "SUITE"), suite)
            self.assertEqual(E.exported(cmd, "TASK"), "code")

    def test_math_bodies_carry_no_sandbox_knobs(self):
        cmd = E.gen_body(BIRWKV, "gsm8k", 0, 4)["command"]
        self.assertEqual(E.exported(cmd, "TASK"), "math")
        self.assertIsNone(E.exported(cmd, "SUITE"))
        self.assertIsNone(E.exported(cmd, "CODE_MEM_MB"))

    def test_no_generative_body_exports_a_multichoice_condition(self):
        # For math/code, CONDITIONS carries the DENOISE STEP COUNT (iter<N>), not a
        # scoring arm. Reusing the fwdce/raw derivation here would silently pass a
        # scorer name where the launcher expects a step count.
        #
        # This asserted `is None` until 2026-09-21, which is how it green-lit the
        # 55-pod failure it was written to prevent: unset does not mean "no scorer
        # name reaches the launcher", it means the launcher supplies its own
        # multichoice default `raw,ddpm100`. The real invariant is that no
        # *multichoice* condition name appears -- see
        # TestGenerativeConditionIsSetNotDefaulted for the rest.
        multichoice_only = set(E.CAUSAL_SPAN_CONDITION.values()) | {"maskce"}
        for suite in sorted(E.GEN_SUITES):
            cmd = E.gen_body(BIRWKV, suite, 0, 4)["command"]
            cond = E.exported(cmd, "CONDITIONS")
            self.assertIsNotNone(cond, f"{suite} would inherit 'raw,ddpm100'")
            for part in cond.split(","):
                self.assertNotIn(
                    part, multichoice_only,
                    f"{suite} passes the multichoice scorer {part!r} where the "
                    f"launcher expects a denoise step count")

    def test_every_suite_file_has_its_cited_item_count(self):
        for suite in sorted(E.GEN_SUITES):
            checked = E.check_gen_suite(suite)
            self.assertEqual(checked["items"], E.GEN_SUITES[suite]["expected_items"])

    def test_a_short_suite_file_is_refused(self):
        original = E.GEN_SUITES["gsm8k"]["expected_items"]
        try:
            E.GEN_SUITES["gsm8k"]["expected_items"] = original + 1
            with self.assertRaises(SystemExit) as cm:
                E.check_gen_suite("gsm8k")
            self.assertIn("same cell", str(cm.exception))
        finally:
            E.GEN_SUITES["gsm8k"]["expected_items"] = original

    def test_an_uncalibrated_code_suite_is_refused(self):
        # A 0 % arm and a sandbox that never ran the grader are the same artifact,
        # so the ceiling must exist before any shard is scheduled.
        saved = E.suite_manifest
        try:
            E.suite_manifest = lambda suite: {
                "jsonl_sha256": "x", "sandbox_calibration": {"status": "not_attached"}}
            with self.assertRaises(SystemExit) as cm:
                E.check_gen_suite("mbppplus")
            self.assertIn("0/40", str(cm.exception))
        finally:
            E.suite_manifest = saved

    def test_a_calibration_at_another_budget_is_refused(self):
        saved = E.suite_manifest
        try:
            E.suite_manifest = lambda suite: {
                "jsonl_sha256": "x",
                "sandbox_settings_validated": {"mem_mb": 256, "timeout_s": 5.0},
                "sandbox_calibration": {"status": "attached",
                                        "unexplained_failures": []}}
            with self.assertRaises(SystemExit) as cm:
                E.check_gen_suite("humanevalplus")
            self.assertIn("uncalibrated budget", str(cm.exception))
        finally:
            E.suite_manifest = saved

    def test_an_unexplained_ground_truth_failure_is_refused(self):
        # Those rows would be charged to the model; a classified upstream-broken
        # row is fine, an unclassified one is not.
        saved = E.suite_manifest
        try:
            E.suite_manifest = lambda suite: {
                "jsonl_sha256": "x",
                "sandbox_settings_validated": {"mem_mb": E.CODE_MEM_MB,
                                               "timeout_s": float(E.CODE_TIMEOUT_S)},
                "sandbox_calibration": {"status": "attached",
                                        "unexplained_failures": ["Mbpp/11"]}}
            with self.assertRaises(SystemExit) as cm:
                E.check_gen_suite("mbppplus")
            self.assertIn("charged to", str(cm.exception))
        finally:
            E.suite_manifest = saved

    def test_a_rebuilt_suite_file_invalidates_the_ceiling(self):
        saved = E.suite_manifest
        try:
            E.suite_manifest = lambda suite: {
                "jsonl_sha256": "0" * 64,
                "sandbox_settings_validated": {"mem_mb": E.CODE_MEM_MB,
                                               "timeout_s": float(E.CODE_TIMEOUT_S)},
                "sandbox_calibration": {"status": "attached",
                                        "unexplained_failures": []}}
            with self.assertRaises(SystemExit) as cm:
                E.check_gen_suite("mbppplus")
            self.assertIn("belongs to the old one", str(cm.exception))
        finally:
            E.suite_manifest = saved

    def test_the_description_carries_the_ceiling(self):
        # A code cell's score is only readable against its ceiling; putting it in
        # the job description means the number travels with the run.
        desc = E.gen_body(BIRWKV, "humanevalplus", 0, 4)["description"]
        self.assertIn("ceiling", desc)

    def test_groups_tile_and_are_named_distinctly(self):
        for suite in sorted(E.GEN_SUITES):
            bodies = E.gen_group(BIRWKV, suite, num_shards=4)
            self.assertEqual(len(bodies), 4)
            E.check_shard_coverage(bodies, 4)

    def test_unschedulable_kinds_are_refused_in_the_generative_lane(self):
        for kind in sorted(tracks.UNSCHEDULABLE_KINDS):
            victims = [t.track_id for t in tracks.ALL_TRACKS if t.model_kind == kind]
            self.assertTrue(victims, f"no registry row has kind {kind}")
            for track_id in victims:
                with self.assertRaises(SystemExit):
                    E.gen_body(track_id, "gsm8k", 0, 1)


class TestGenerativeConditionIsSetNotDefaulted(unittest.TestCase):
    """An unset ``CONDITIONS`` inherits a default meant for a different task.

    Measured 2026-09-21, after 55 pods failed: ``gen_body`` exported no
    ``CONDITIONS``, so every math/code shard ran under the launcher's *multichoice*
    default ``raw,ddpm100`` (launch_capability_eval.sh:52).  That broke the two model
    kinds in opposite directions, and only one of them was loud:

    * ``birwkv_diffusion`` refuses any non-``iter`` method (run_eval.py:338,481), so
      the sweep died with ``supports iter<N> conditions only, got 'raw'`` -- but only
      *after* the load was paid for, and only in the per-GPU log.  The probe passed
      every time, because the math/code probe runs ``--arm math_probe``, which
      bypasses the condition parse entirely.  A green probe is not evidence about the
      sweep's conditions.
    * ``hf_causal`` accepted it and produced numbers.  ``raw`` and ``greedy`` and
      ``ddpm100`` all map to method ``raw`` for an HF model and build the same
      ``HFCausalGenerator``, so each released baseline scored one statistic twice
      under two arm labels: Qwen2.5-3B GSM8K em ``0.8061`` under both arms, 143/165
      completions byte-identical.  Double the GPU-hours, and an ``n`` that reads as
      doubled to anything counting records rather than documents.

    The steps/group/order are pinned to the banked protocol for a separate reason:
    comparability.  Every generative number already on disk
    (``cap_eval_s9500_{math,code}``, ``cap_eval_b3_gsm8k_g4_*``, ``cap_eval_t2_*``,
    ``cap_eval_wl1_*``, the W-L0 battery) was produced at ``iter16`` / ``g=4`` /
    ``confidence``, and g=0 is a different decode path, not a milder setting.
    """

    @staticmethod
    def _schedulable(kind):
        return [t for t in tracks.ALL_TRACKS
                if t.model_kind == kind and t.model_dir in E.TOKENIZER_TREE]

    def test_every_generative_body_sets_conditions(self):
        for kind in ("birwkv_diffusion", "hf_causal"):
            for t in self._schedulable(kind):
                for suite in ("gsm8k", "humanevalplus", "mbppplus"):
                    cmd = E.gen_body(t.track_id, suite, 0, 4)["command"]
                    self.assertIsNotNone(
                        E.exported(cmd, "CONDITIONS"),
                        f"{t.track_id}/{suite} would inherit 'raw,ddpm100'")

    def test_diffusion_tracks_get_an_iter_condition(self):
        bi = self._schedulable("birwkv_diffusion")
        self.assertTrue(bi)
        for t in bi:
            cmd = E.gen_body(t.track_id, "gsm8k", 0, 4)["command"]
            self.assertEqual(E.exported(cmd, "CONDITIONS"), f"iter{E.GEN_STEPS}")

    def test_no_generative_condition_names_a_method_the_denoiser_refuses(self):
        # run_eval raises for any method != "iter" on birwkv_diffusion. Read its
        # own parser rather than restating the grammar, so an upstream rename
        # fails here instead of in a pod.
        run_eval = Path(E.SCALE_DIR, "eval/capability/run_eval.py").read_text()
        self.assertIn('supports iter<N> conditions only', run_eval,
                      "the refusal this test is about is gone; re-read the parser")
        cond = E.GEN_CONDITION["birwkv_diffusion"]
        for part in cond.split(","):
            self.assertTrue(part.startswith("iter"),
                            f"{part!r} would raise on the card after the load")

    def test_hf_causal_does_not_score_the_same_statistic_twice(self):
        # The measured defect: two arm labels, one computation. Any hf_causal
        # condition list whose parts all reduce to method "raw" must be a single
        # part, or the shard pays twice for one number.
        cond = E.GEN_CONDITION["hf_causal"]
        parts = cond.split(",")
        self.assertEqual(len(parts), 1,
                         f"{cond!r}: every part maps to method 'raw' for an HF "
                         f"model, so >1 part is the same statistic billed twice")

    def test_the_hf_causal_label_does_not_claim_denoise_steps(self):
        # `ddpm100`/`iter16` on an AR model would put a step count on a model with
        # no denoise loop; the merged artifact groups by this label.
        cond = E.GEN_CONDITION["hf_causal"]
        for method in ("ddpm", "ddim", "flow", "iter"):
            self.assertFalse(cond.startswith(method),
                             f"{cond!r} claims a {method} step count for an AR model")

    def test_generative_conditions_are_parsed_by_the_harness(self):
        run_eval = Path(E.SCALE_DIR, "eval/capability/run_eval.py").read_text()
        for cond in E.GEN_CONDITION.values():
            for part in cond.split(","):
                stem = part.rstrip("0123456789") or part
                self.assertIn(f'"{stem}"', run_eval,
                              f"{part} is not in parse_generative_condition")

    def test_an_unmapped_kind_is_refused_in_the_generative_lane(self):
        unmapped = tracks.MODEL_KINDS - set(E.GEN_CONDITION)
        self.assertTrue(unmapped)
        for kind in sorted(unmapped):
            bogus = tracks.Track(
                track_id="fake", track="released", paper_arm="C0",
                local_label="fake", rel_path="models/RWKV7-Goose-World3-2.9B-HF",
                model_dir="models/RWKV7-Goose-World3-2.9B-HF", sha256="",
                params_stored=1, params_executed=1, tokens_seen=-1,
                training_seed=None, step=None, loop_reps=0, variant="",
                model_kind=kind)
            with self.assertRaises(SystemExit) as cm:
                E.generative_condition(bogus)
            self.assertIn(kind, str(cm.exception))

    def test_the_decoder_knobs_match_the_banked_protocol(self):
        # Not style: g=0 is the legacy single-forward multi-commit path and g=4
        # bounds commits per forward with a canvas refresh between groups. The rows
        # this table joins were all run at g=4/confidence.
        for suite in ("gsm8k", "humanevalplus"):
            cmd = E.gen_body(BIRWKV, suite, 0, 4)["command"]
            self.assertEqual(E.exported(cmd, "COMMIT_GROUP"), "4")
            self.assertEqual(E.exported(cmd, "COMMIT_ORDER"), "confidence")

    def test_every_generative_body_pins_the_banked_temperature(self):
        # TEMPERATURE lives in the launcher's COMMON_ARGS, so it reaches math too.
        # Gating this export on task == "code" left gsm8k on the launcher default.
        for suite in sorted(E.GEN_SUITES):
            cmd = E.gen_body(BIRWKV, suite, 0, 4)["command"]
            self.assertEqual(E.exported(cmd, "TEMPERATURE"), "0.2",
                             f"{suite} would inherit the launcher default")

    def test_an_explicit_condition_still_overrides(self):
        cmd = E.gen_body(BIRWKV, "gsm8k", 0, 4, conditions="iter8")["command"]
        self.assertEqual(E.exported(cmd, "CONDITIONS"), "iter8")

    def test_the_group_passes_the_condition_to_every_pod(self):
        for body in E.gen_group(BIRWKV, "gsm8k", num_shards=4, conditions="iter8"):
            self.assertEqual(E.exported(body["command"], "CONDITIONS"), "iter8")


class TestCodeIsolationIsReadFromTheArtifact(unittest.TestCase):
    """The pod's isolation mode is a post-hoc reading, not a preflight.

    The calibration receipt records ``socket_stub`` because it ran on the login
    node, where ``unshare`` is unavailable.  That certifies the ceiling at
    4096 MiB / 30 s and says nothing about how a pod sandboxed its tests -- two
    different claims, and only the pod can settle the second.  ``run_eval`` writes
    ``isolation_mode`` onto every code record, so the artifact answers it if asked.
    """

    def test_the_required_mode_is_not_the_login_nodes(self):
        self.assertEqual(E.REQUIRED_POD_ISOLATION, "unshare")
        self.assertNotEqual(E.REQUIRED_POD_ISOLATION, "socket_stub")

    def _shard(self, tmp: Path, modes: list[str]):
        (tmp / "code-humanevalplus.raw.shard0of1.json").write_text(json.dumps(
            {"records": [{"document_id": f"i{i}", "isolation_mode": m}
                         for i, m in enumerate(modes)]}))

    def test_an_unshare_shard_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._shard(Path(tmp), ["unshare"] * 3)
            observed = E.check_code_isolation(tmp)
            self.assertEqual(list(observed.values()), [{"unshare": 3}])

    def test_a_socket_stub_shard_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._shard(Path(tmp), ["socket_stub"] * 3)
            with self.assertRaises(SystemExit) as cm:
                E.check_code_isolation(tmp)
            self.assertIn("socket_stub", str(cm.exception))

    def test_a_mixed_shard_is_refused(self):
        # Partial isolation is the dangerous case: most rows look right, so a
        # spot check passes and the aggregate is still a different protocol.
        with tempfile.TemporaryDirectory() as tmp:
            self._shard(Path(tmp), ["unshare", "unshare", "socket_stub"])
            with self.assertRaises(SystemExit):
                E.check_code_isolation(tmp)

    def test_an_empty_lane_cannot_be_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                E.check_code_isolation(tmp)
            self.assertIn("must not", str(cm.exception))

    def test_records_without_the_field_cannot_be_certified(self):
        # "no isolation_mode key" must not read as "isolation was fine".
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "code-x.json").write_text(json.dumps({"records": []}))
            with self.assertRaises(SystemExit):
                E.check_code_isolation(tmp)


    def test_the_drip_subset_is_a_subset_and_the_full_sweep_survives(self):
        # Operator direction 2026-09-21: get results sooner. The lever is model coverage,
        # not benchmark size -- every panel cell that IS reported stays measured over the
        # full item set. The full 25-track list must remain reachable.
        full = E.aux_track_ids()
        keep = [t for t in E.AUX_TRACKS_DRIP if t in full]
        self.assertTrue(keep, "the drip subset contains no schedulable track")
        self.assertLess(len(keep), len(full),
                        "the subset is the whole sweep, so nothing was reduced")
        self.assertEqual(set(keep) - set(full), set())
        # The paper's own comparison rows must all be in it.
        for tid in ("a29_c6_f2_s14000", "rel_c1_rwkv7_2p9b", "rel_c1_rwkv7_0p4b"):
            self.assertIn(tid, keep, f"{tid} is a headline row and must be kept")

    def test_the_drip_subset_covers_every_arm_of_the_0p4b_factorial(self):
        # The direction x loop table needs all four cells at one seed; a subset that
        # dropped one would leave the table with a hole where its contrast is.
        for arm in ("a04_a1_s17", "a04_a2_s17", "a04_a3_s17", "a04_a5_s17"):
            self.assertIn(arm, E.AUX_TRACKS_DRIP)


class TestTheAuxFillOrder(unittest.TestCase):
    """The fill order decides which panels become usable first, so it is a claim."""

    def test_priority_is_a_reordering_and_never_a_filter(self):
        # The set must be exactly the schedulable rwkv_world tracks. A priority list
        # that dropped a track would silently shrink every panel by one column.
        expected = {t.track_id for t in E.tracks.ALL_TRACKS
                    if t.model_kind not in E.tracks.UNSCHEDULABLE_KINDS
                    and t.model_dir in E.TOKENIZER_TREE
                    and E.TOKENIZER_TREE.get(t.model_dir) == "rwkv_world"}
        self.assertEqual(set(E.aux_track_ids()), expected)

    def test_the_studys_own_rows_lead_every_dataset(self):
        ids = E.aux_track_ids()
        rank = {tid: i for i, tid in enumerate(E.AUX_TRACK_PRIORITY)}
        # Every listed-and-schedulable track precedes every unlisted one.
        placed = [t for t in ids if t in rank]
        unplaced = [t for t in ids if t not in rank]
        self.assertTrue(placed, "no priority track is schedulable at all")
        if unplaced:
            self.assertLess(max(rank[t] for t in placed),
                            min(rank.get(t, len(rank)) for t in unplaced),
                            "a priority track is ordered after an unlisted one")
        # And F2 itself, the object of the study, is first.
        self.assertEqual(ids[0], "a29_c6_f2_s14000")

    def test_a_priority_track_that_is_not_schedulable_is_dropped_not_fatal(self):
        # rel_c7ar_mamba_2p8b is in the priority list but tokenizes with GPT-NeoX,
        # so it cannot use an RWKV-world tree. That must not raise: the ordering
        # preference for one track is not a reason to refuse the whole lane.
        self.assertIn("rel_c7ar_mamba_2p8b", E.AUX_TRACK_PRIORITY)
        self.assertNotIn("rel_c7ar_mamba_2p8b", E.aux_track_ids())

    def test_the_named_panels_are_ordered_before_the_supporting_breadth(self):
        # The protocol names MMLU, OpenBookQA and RACE; OpenBookQA is cheap and was
        # previously ninth, so a named panel is bought for a fraction of RACE's cost.
        order = list(E.AUX_ORDER)
        self.assertLess(order.index("openbookqa"), order.index("gpqa_diamond"))
        self.assertLess(order.index("race"), order.index("siqa"))
        self.assertEqual(set(order), set(E.AUX_TREES),
                         "the fill order and the registry disagree about the datasets")


class TestTheStateBaselines(unittest.TestCase):
    """The long-context claim needs another constant-state family member.

    RWKV-7 is one member of the state/linear-attention family.  Compared only to
    softmax attention, "constant state wins at long context" cannot be separated
    from "our checkpoint wins": the mixer and the training run move together.  So
    three released family members on this disk were registered on 2026-09-21 --
    and probed rather than assumed, which is the whole content of these tests: the
    three probes gave three *different* answers, and the registry records each.

    The trap this class exists to hold shut: ``check_hf_causal_is_loadable`` reads
    ``config.json`` only.  Both blocked models *have* a valid dispatch target, so
    that check passes them, and calling them ``hf_causal`` on its word would have
    booked two baselines that fail on an in-pod ImportError after the queue wait.
    """

    MAMBA = "rel_c7ar_mamba_2p8b"
    GLA = "rel_c3ar_gla_1p3b"
    RWKV6 = "rel_c8ar_rwkv6_3b"

    def _track(self, track_id):
        return tracks.track_by_id(track_id) if hasattr(tracks, "track_by_id") else \
            next(t for t in tracks.ALL_TRACKS if t.track_id == track_id)

    def test_all_three_are_in_the_registry(self):
        ids = {t.track_id for t in tracks.ALL_TRACKS}
        for tid in (self.MAMBA, self.GLA, self.RWKV6):
            self.assertIn(tid, ids)

    def test_the_loadable_ones_are_schedulable_and_the_third_is_not(self):
        # Measured, not assumed: mamba-2.8b built MambaForCausalLM and ran a
        # forward on CPU. GLA's blocker was one missing import and now has a loader
        # kind of its own; rwkv6 still raises before any weight is read, for a
        # different reason (bitsandbytes at module scope in its remote modeling
        # file), and stays MISSING_RUNTIME_DEP.
        self.assertEqual(self._track(self.MAMBA).model_kind, "hf_causal")
        self.assertEqual(self._track(self.GLA).model_kind, "fla_causal")
        self.assertEqual(self._track(self.RWKV6).model_kind,
                         tracks.MISSING_RUNTIME_DEP)

    def test_the_fla_loader_is_gated_by_the_probe_that_now_passes(self):
        # The loader kind makes the row *schedulable*; the Q1 probe is what makes it
        # *scheduled*. On 2026-09-21 that distinction was live -- the row had no receipt
        # and `require_probe` refused, which is what kept an unverified loader out of a
        # full 8-GPU MMLU group. The probe then PASSED (job-051095cb: 339 tensors loaded,
        # 4/4 scored), after three successive blockers -- architecture registration, a
        # missing sentencepiece backend, and an HF-cache wrapper directory that would not
        # resolve. The assertion now records the outcome rather than the refusal.
        ok, detail = E.probe_passed(self.GLA)
        self.assertTrue(ok, f"GLA has no Q1 receipt: {detail}")
        self.assertEqual(E.require_probe(self.GLA), detail)
        # And it is genuinely offered by the wide MMLU lane now, not merely unrefused.
        self.assertIn(self.GLA, E.schedulable_tracks())
        self.assertFalse(E.merged_present(self.GLA),
                         "GLA already has a merged MMLU reading")

    def test_a_blocked_baseline_is_refused_in_both_lanes(self):
        # Both lanes, because a kind added to one refusal and not the other is a
        # submission waiting to happen.
        with self.assertRaises(SystemExit) as cm:
            E.mmlu_body(self.RWKV6, 0, 1)
        self.assertIn("relay2:v2", str(cm.exception))
        with self.assertRaises(SystemExit):
            E.gen_body(self.RWKV6, "gsm8k", 0, 1)

    def test_the_schedulable_baseline_builds_a_body_with_its_own_tree(self):
        cmd = E.mmlu_body(self.MAMBA, 0, 1)["command"]
        # Scoring a mamba checkpoint against the rwkv_world tree would score four
        # arbitrary token ids: the label ids are 329/378/330/399 there and
        # 300-303 here.
        self.assertIn("tasks_mmlu_onetoken/mamba/", cmd)
        self.assertNotIn("tasks_mmlu_onetoken/rwkv_world/", cmd)

    def test_every_baseline_has_its_own_tokenizer_tree(self):
        seen = {}
        for tid in (self.MAMBA, self.GLA, self.RWKV6):
            t = self._track(tid)
            self.assertIn(t.model_dir, E.TOKENIZER_TREE,
                          f"{tid} has no MMLU tree, so its label ids are unknown")
            tree = E.TOKENIZER_TREE[t.model_dir]
            self.assertNotIn(tree, seen,
                             f"{tid} shares tree {tree!r} with {seen.get(tree)}; "
                             f"these are four different vocabularies")
            seen[tree] = tid

    def test_a_blocked_row_names_the_missing_dependency(self):
        # Otherwise MISSING_RUNTIME_DEP is indistinguishable from NO_LOADER, and
        # the paper reports "absent" where the truth is "one import away".
        tracks.check_runtime_dep_rows_state_the_dependency()
        self.assertIn("fla", self._track(self.GLA).notes)
        self.assertIn("bitsandbytes", self._track(self.RWKV6).notes)

    def test_the_arms_are_autoregressive_not_diffusion(self):
        # A released Mamba/GLA/RWKV-6 checkpoint is AR. Filing it under C7/C3
        # (absorbing_diffusion, bidirectional) would assert an objective and a
        # direction it does not have -- the rwkv04b "A0" error in reverse.
        for tid in (self.MAMBA, self.GLA, self.RWKV6):
            arm = self._track(tid).paper_arm
            mixer, objective, direction = tracks.PAPER_ARMS[arm]
            self.assertEqual(objective, "autoregressive", f"{tid} -> {arm}")
            self.assertEqual(direction, "causal", f"{tid} -> {arm}")

    def test_the_new_arms_are_declared_for_the_protocol_patch(self):
        # validate_protocol.py:50 refuses a matrix row whose arm is not in
        # protocol.json["arms"], so an arm this registry invents must arrive with
        # the fields that patch needs.
        for tid in (self.MAMBA, self.GLA, self.RWKV6):
            arm = self._track(tid).paper_arm
            self.assertIn(arm, tracks.BASELINE_ARMS,
                          f"{arm} is in PAPER_ARMS but has no protocol entry")
            entry = tracks.BASELINE_ARMS[arm]
            self.assertEqual(
                (entry["mixer"], entry["objective"], entry["direction"]),
                tracks.PAPER_ARMS[arm],
                f"{arm} describes a different mixer in the two tables")
            self.assertTrue(entry["deviation"],
                            f"{arm} deviates from the protocol silently")

    def test_the_mamba_row_does_not_claim_to_be_mamba2(self):
        # config.json is model_type=mamba / MambaForCausalLM: this is Mamba-1
        # (S6). The protocol's C7 mixer label reads "mamba2", so the row has to
        # say so or the paper reports an SSD result it never measured.
        t = self._track(self.MAMBA)
        self.assertIn("Mamba-1", t.variant)
        self.assertIn("not Mamba-2", t.variant.replace("/SSD", ""))

    def test_the_mamba_row_records_that_it_has_no_fast_path(self):
        # mamba_ssm/causal_conv1d are absent from relay2:v2, so the SSM recurrence
        # is a per-timestep Python loop. E1 at 4K is fine; pricing a 32K cell off
        # the RWKV arms' cost would be wrong by an unmeasured factor.
        self.assertIn("slow_forward", self._track(self.MAMBA).notes)

    def test_the_unschedulable_set_is_derived_not_listed(self):
        # This set had four hardcoded copies; adding a kind broke exactly one of
        # them, and the disagreement would have surfaced one submitted job at a
        # time. Deriving it means a new kind is excluded by construction.
        self.assertEqual(
            tracks.UNSCHEDULABLE_KINDS,
            frozenset(tracks.MODEL_KINDS - tracks.RUN_EVAL_MODEL_KINDS))
        self.assertIn(tracks.MISSING_RUNTIME_DEP, tracks.UNSCHEDULABLE_KINDS)
        self.assertNotIn("hf_causal", tracks.UNSCHEDULABLE_KINDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
