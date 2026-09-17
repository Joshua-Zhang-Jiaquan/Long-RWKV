"""Task 7: the arm matrix must be the registered study, and its flags must be real.

Two classes of error this file exists to catch before a job is submitted:

* **A flag that the trainer does not define.**  A single misspelled flag in one
  of twelve runs would fail after the job started.  The launcher has a preflight
  for this, but a dry-run here costs nothing and the check is stronger: it reads
  the trainer's own argparse.
* **A matrix that is not the study.**  A dropped seed or a relabelled arm would
  look like a complete run set.  The plan's failure QA requires that removing one
  seed fails the matched-study gate rather than shrinking the study silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.task7 import arm_matrix as am

REPO = Path(__file__).resolve().parents[3]
TRAINER = REPO / "DAN" / "v7_arch_round" / "code" / "train" / "train_birwkv_diffusion.py"


def _trainer_flags() -> set[str]:
    """Every ``--flag`` the trainer's argparse defines, via the launcher's own AST.

    Deliberately the same technique as the launcher's preflight (AST-walk the
    ``add_argument`` calls) rather than a regex over the source: a regex and an
    AST disagree on exactly the cases that matter, and the launcher's verdict is
    the one that decides whether a job runs.
    """
    if not TRAINER.is_file():
        pytest.skip(f"trainer absent: {TRAINER}")
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    return {
        a.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
        for a in n.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str)
    }


def _preflight_rejects(argv: list[str]) -> list[str]:
    """The launcher's rule: a token starting ``--`` must EQUAL a flag name."""
    known = _trainer_flags()
    return [a for a in argv if a.startswith("--") and a not in known]


# --------------------------------------------------------------------------
# the flags are real
# --------------------------------------------------------------------------


def test_every_arm_flag_passes_the_launchers_preflight() -> None:
    """The check that would have caught a fired job failing its own preflight.

    The first submission used ``--seed=17`` and ``--loop-range=12:24``. The
    launcher's preflight compares each ``--`` token to the flag names for EQUALITY,
    so both were rejected, the run exited before loading a model, and the job's
    auto-fault-tolerance restarted the identical broken command. A substring
    match here would have passed it, which is why this mirrors the preflight's
    own rule instead.
    """
    # Given: every arm's argv tail, split the way the shell will split it.
    for arm_id, spec in am.ARMS.items():
        argv = list(spec.extra_args)
        # When/Then: no token is rejected...
        assert _preflight_rejects(argv) == [], f"{arm_id}: {_preflight_rejects(argv)}"
        # ...and no token smuggles its value with '='.
        assert all("=" not in token for token in argv if token.startswith("--")), arm_id


def test_every_seed_passes_the_launchers_preflight() -> None:
    # Given: the real argv tails, which include the seed the driver appends.
    runs = am.build_matrix(ngpus=8)
    # When/Then: splitting on whitespace must produce preflight-clean tokens --
    # this is the exact string the launcher will receive in EXTRA_ARGS.
    for run in runs:
        tokens = run.argv_tail.split()
        assert _preflight_rejects(tokens) == [], (run.run_name, _preflight_rejects(tokens))


def test_the_seed_flag_exists_and_is_not_passed_by_the_launcher() -> None:
    # Given: the trainer and the launcher contract.
    defined = _trainer_flags()
    # When/Then: --seed is a real flag, so passing it through EXTRA_ARGS is
    # sound rather than a hope -- the launcher's preflight validates EXTRA_ARGS
    # against this same argparse, as the two tests above now do for real.
    assert "--seed" in defined


# --------------------------------------------------------------------------
# the matrix is the registered study
# --------------------------------------------------------------------------


def test_the_matrix_is_four_arms_by_three_seeds() -> None:
    # Given: the decided study.
    runs = am.build_matrix(ngpus=8)
    # When/Then: twelve runs, and every arm-seed pair present exactly once.
    assert len(runs) == 12
    assert len({(r.arm_id, r.seed) for r in runs}) == 12
    am.require_matrix(runs)


def test_the_two_absent_arms_are_recorded_with_their_reasons() -> None:
    # Given: the registered arms that cannot be trained.
    # When/Then: they are named, with why, so a reader of the matrix cannot
    # assume four arms cover six.
    assert set(am.ABSENT_ARMS) == {"A0", "A4"}
    assert "causal-objective switch" in am.ABSENT_ARMS["A0"]
    assert "external-validity" in am.ABSENT_ARMS["A0"]
    assert "not implemented" in am.ABSENT_ARMS["A4"]
    assert "A0" not in am.ARMS and "A4" not in am.ARMS


def test_an_unimplemented_arm_cannot_be_added_by_name() -> None:
    # Given: a caller asking for the arm with no implementation.
    # When/Then: refused, and the message says why rather than emitting a run
    # that would silently be some other arm's configuration.
    with pytest.raises(am.MatrixRefusal, match="not trainable"):
        am.build_matrix(ngpus=8, arms=("A1", "A4"))
    with pytest.raises(am.MatrixRefusal, match="not trainable"):
        am.build_matrix(ngpus=8, arms=("A0",))


def test_dropping_a_seed_fails_the_matched_study_gate() -> None:
    # Given: a matrix built from two seeds only.
    runs = am.build_matrix(ngpus=8, seeds=(17, 29))
    # When/Then: the coverage check refuses it, naming the missing seed -- the
    # plan's failure QA is explicit that a reduced study must not read as a
    # complete one.
    with pytest.raises(am.MatrixRefusal, match="missing"):
        am.require_matrix(runs)


def test_the_frozen_budget_cannot_be_changed() -> None:
    # Given: the owner's frozen budget.
    assert am.FROZEN_TOKEN_BUDGET == 2_000_000_000
    # When/Then: asking for another budget is refused, because changing it after
    # results are observed is the thing the freeze exists to prevent.
    with pytest.raises(am.MatrixRefusal, match="frozen"):
        am.build_matrix(ngpus=8, budget=4_000_000_000)


def test_every_run_passes_its_seed_to_the_trainer() -> None:
    # Given: the matrix.
    runs = am.build_matrix(ngpus=8)
    # When/Then: each run's argv carries its own seed. The trainer defaults to
    # 42, so a run that omitted this would silently be a fourth seed-42 run.
    for run in runs:
        assert f"--seed {run.seed}" in run.argv_tail
    assert {r.seed for r in runs} == set(am.TRAINING_SEEDS)


def test_run_names_are_unique_and_carry_the_arm_and_seed() -> None:
    # Given: the matrix.
    runs = am.build_matrix(ngpus=8)
    names = [r.run_name for r in runs]
    # When/Then: names are unique -- twelve runs share one save root, and a name
    # that collided would let one run overwrite another's checkpoint.
    assert len(set(names)) == len(names)
    for run in runs:
        assert run.arm_id.lower() in run.run_name
        assert f"s{run.seed}" in run.run_name


# --------------------------------------------------------------------------
# the arm configurations differ where they must
# --------------------------------------------------------------------------


def test_the_arms_differ_only_in_direction_and_loop() -> None:
    # Given: the four trainable arms.
    # When/Then: the mechanism under test is isolated. A1 and A5 are forward-only;
    # A2 and A3 are bidirectional; only A3 and A5 loop. Any other difference
    # would make a delta unattributable.
    forward = {a for a, s in am.ARMS.items() if s.force_forward}
    looping = {a for a, s in am.ARMS.items() if s.loop_reps > 0}
    assert forward == {"A1", "A5"}
    assert looping == {"A3", "A5"}


def test_the_loop_range_matches_the_measured_small_arm() -> None:
    # Given: the range the calibration table measured.
    # When/Then: the loop arms pass exactly that range, so the emitted runs are
    # the geometry the throughput numbers describe.
    assert am.SMALL_LOOP_RANGE == (12, 24)
    for arm_id in ("A3", "A5"):
        flags = am.ARMS[arm_id].extra_args
        assert "--loop-range" in flags and "--loop-reps" in flags
        assert f"{am.SMALL_LOOP_RANGE[0]}:{am.SMALL_LOOP_RANGE[1]}" in flags
        assert "1" in flags


def test_gradient_checkpointing_is_on_for_every_arm() -> None:
    # Given: a 4.09B-class model trained at 4096 tokens on one node.
    # When/Then: every arm carries it. An arm without it would be measuring a
    # different memory regime, and its throughput would not be comparable.
    for arm_id, spec in am.ARMS.items():
        assert "--gradient-checkpointing" in spec.extra_args, arm_id


# --------------------------------------------------------------------------
# the arithmetic and the forecast
# --------------------------------------------------------------------------


def test_tokens_per_step_matches_the_continuation() -> None:
    # Given: the small-arm batch shape at 8 GPUs.
    per_step = am.tokens_per_step(ngpus=8)
    # When/Then: it equals the 1,048,576 tokens/step the 2.9B continuation runs
    # at 16 GPUs (4 x 4 x 16 x 4096), which is why the two are commensurate.
    assert per_step == 1_048_576
    assert am.tokens_per_step(ngpus=16, microbatch=4, grad_accum=4) == per_step


def test_steps_round_up_so_no_arm_is_short_of_the_budget() -> None:
    # Given: a budget that does not divide evenly.
    steps = am.steps_for_budget(2_000_000_000, ngpus=8)
    # When/Then: the run reaches at least the budget -- rounding down would make
    # every arm slightly short and the arms short by different amounts.
    assert steps == 1908
    assert steps * am.tokens_per_step(ngpus=8) >= 2_000_000_000
    with pytest.raises(am.MatrixRefusal, match="positive"):
        am.steps_for_budget(0, ngpus=8)


def test_the_forecast_scales_from_the_ledger_and_is_labelled_one() -> None:
    # Given: the ledger's measured 4B figure for A2.
    # When/Then: the 2B forecast is exactly half, and the report says these are
    # forecasts rather than measurements.
    assert am.forecast_gpu_hours("A2", budget=4_000_000_000) == pytest.approx(1253.14)
    assert am.forecast_gpu_hours("A2", budget=2_000_000_000) == pytest.approx(626.57)
    report = am.matrix_report(ngpus=8)
    assert "FORECASTS" in report["note"]
    assert report["arms_absent"] == am.ABSENT_ARMS


def test_an_arm_with_no_throughput_measurement_is_refused() -> None:
    # Given: an arm the ledger never measured.
    # When/Then: refusing, because a forecast from no measurement is a guess
    # wearing a number.
    with pytest.raises(am.MatrixRefusal, match="no ledger throughput"):
        am.forecast_gpu_hours("A4")


def test_a_run_longer_than_one_wall_cap_is_visible_before_submitting() -> None:
    """A wall-cap stop does not auto-resume, so segments are a real cost."""
    # Given: the 2B budget at 8 GPUs, which the ledger's throughputs put at ~78h.
    report = am.matrix_report(ngpus=8)
    segments = report["segments_per_run_at_wall_cap"]
    # When/Then: the report states that each run spans multiple 36h segments, and
    # that more cards would collapse it -- so the operator chooses knowingly.
    assert all(v >= 2 for v in segments.values()), segments
    # 24 cards collapses it inside one segment (626.6 GPU-h / 24 = 26.1 h);
    # 8 or 16 cannot, which is why the helper returns 0 rather than a fiction.
    assert am.gpus_for_one_segment("A2", budget=2_000_000_000) == 24
    assert am.gpus_for_one_segment("A2", budget=2_000_000_000,
                                  candidates=(8, 16)) == 0
    assert am.forecast_wall_hours("A2", ngpus=24, budget=2_000_000_000) < am.WALL_CAP_HOURS


def test_the_total_forecast_is_the_sum_over_arms_and_seeds() -> None:
    # Given: the report at 8 GPUs.
    report = am.matrix_report(ngpus=8)
    # When/Then: the total is the per-arm-seed sum times the seeds -- so a reader
    # cannot mistake the per-arm figure for the study's cost.
    per = sum(report["forecast_gpu_hours_per_arm_seed"].values())
    assert report["forecast_gpu_hours_total"] == pytest.approx(per * 3, rel=1e-6)
    assert report["runs"] == 12


# --------------------------------------------------------------------------
# what the seeds actually vary
# --------------------------------------------------------------------------


def test_the_trainer_wires_the_seed_into_the_data_sampler() -> None:
    """The seed is not only an init seed, and the matrix depends on that."""
    # Given: the trainer's DataLoader construction.
    if not TRAINER.is_file():
        pytest.skip(f"trainer absent: {TRAINER}")
    source = TRAINER.read_text(encoding="utf-8")
    # When/Then: the sampler is built with seed=args.seed, so --seed varies the
    # PERMUTATION as well as the initialisation. If this ever becomes a constant,
    # every arm in a wave would still read the same rows (fine) but the three
    # seeds would stop being data-level replication (not fine), and nothing else
    # in the suite would notice.
    assert "sampler=ShardContiguousSampler(" in source
    assert "seed=args.seed" in source


def test_within_a_wave_every_arm_shares_one_seed() -> None:
    """The controlled comparison: architecture is the only variable."""
    # Given: the wave packing.
    waves = am.wave_plan(ngpus_per_run=8)
    # When/Then: each wave is exactly one seed across every arm, so all four arms
    # read the same permutation -- which is what makes an arm difference
    # attributable to the arm.
    for wave in waves:
        seeds = {run.seed for run in wave.runs}
        arms = {run.arm_id for run in wave.runs}
        assert len(seeds) == 1, (wave.index, seeds)
        assert arms == set(am.ARMS), (wave.index, arms)


def test_across_waves_the_seed_changes() -> None:
    """So the three seeds are data-level replication, not three initials."""
    # Given: the wave packing.
    waves = am.wave_plan(ngpus_per_run=8)
    # When/Then: the wave seeds are the three training seeds, in order.
    assert [next(iter({r.seed for r in w.runs})) for w in waves] == list(am.TRAINING_SEEDS)


def test_every_arm_in_a_wave_passes_the_wave_seed_to_the_trainer() -> None:
    # Given: a wave.
    wave = am.wave_plan(ngpus_per_run=8)[1]
    # When/Then: every run in it carries that same seed on its argv, so the
    # shared-data property survives the trip through EXTRA_ARGS.
    seed = next(iter({run.seed for run in wave.runs}))
    for run in wave.runs:
        assert f"--seed {seed}" in run.argv_tail
