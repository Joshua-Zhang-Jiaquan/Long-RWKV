"""The supervisor must continue a run and must never continue a bug.

Same two guards as the resubmitter it replaces, kept for the reasons they earned: a
barren streak stops the chain (the first attempt of this study died in its own preflight
and the platform restarted the identical broken command three times), and an unknown
submission outcome stops rather than risking two jobs on one checkpoint directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.task7 import arm_matrix as am
from scale.experiments.nonlatent_iclr.task7 import supervise_matrix_jobs as sup


def _state(step: int | None = 0, status: str = "job_stopped", target: int = 1908) -> sup.RunState:
    return sup.RunState(name="task7-a1-s17", job_id="job-x", status=status,
                        best_step=step, target_steps=target)


# --------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------


def test_a_run_that_reached_the_target_is_done() -> None:
    decision, reason = sup.decide_run(_state(step=1908), resubmits_used=0,
                                      max_resubmits=12, barren_streak=0, max_barren=2)
    assert decision is sup.Decision.DONE
    assert "1908" in reason


def test_a_running_run_waits() -> None:
    for status in ("job_running", "job_queuing", "job_restarting"):
        decision, _ = sup.decide_run(_state(step=100, status=status), resubmits_used=0,
                                     max_resubmits=12, barren_streak=0, max_barren=2)
        assert decision is sup.Decision.WAIT, status


def test_a_stopped_run_short_of_its_target_is_resubmitted() -> None:
    decision, reason = sup.decide_run(_state(step=1400), resubmits_used=1,
                                      max_resubmits=12, barren_streak=0, max_barren=2)
    assert decision is sup.Decision.RESUBMIT
    assert "1400" in reason


def test_a_barren_streak_stops_the_run_before_its_budget() -> None:
    """The preflight case: 8 H100s billing to run nothing."""
    decision, reason = sup.decide_run(_state(step=None), resubmits_used=1,
                                      max_resubmits=12, barren_streak=2, max_barren=2)
    assert decision is sup.Decision.STOP
    assert "no checkpoint" in reason


def test_a_run_with_no_checkpoint_at_all_has_not_produced() -> None:
    # Given: a run that never wrote a step directory.
    # When/Then: it has not produced, so a resubmission after it counts as barren.
    assert sup.RunState("r", "j", "job_failed", None, 1908).produced is False
    assert sup.RunState("r", "j", "job_failed", 500, 1908).produced is True
    # and a run at step 0 has not produced either
    assert sup.RunState("r", "j", "job_failed", 0, 1908).produced is False


def test_the_budget_is_a_ceiling() -> None:
    decision, reason = sup.decide_run(_state(step=900), resubmits_used=12,
                                      max_resubmits=12, barren_streak=0, max_barren=2)
    assert decision is sup.Decision.STOP
    assert "budget exhausted" in reason


# --------------------------------------------------------------------------
# reading progress from the run's own directory
# --------------------------------------------------------------------------


def test_the_newest_step_is_read_from_the_run_directory(tmp_path: Path) -> None:
    # Given: three checkpoints, written out of order.
    run = tmp_path / "task7-a1-s17"
    for step in (500, 1500, 1000):
        (run / f"step_{step:08d}").mkdir(parents=True)
        (run / f"step_{step:08d}" / "model.pt").write_text("x")
    # When/Then: the HIGHEST is returned, because that is what the launcher resumes from.
    assert sup.newest_step(run) == 1500


def test_a_run_with_no_directory_has_no_step(tmp_path: Path) -> None:
    assert sup.newest_step(tmp_path / "absent") is None


def test_step_directories_with_no_weights_are_refused(tmp_path: Path) -> None:
    """A step directory without model.pt is a partial write, not progress.

    Counting it would resume from a checkpoint that does not exist, and the run would
    silently restart from an earlier one while the supervisor believed it had advanced.
    """
    run = tmp_path / "task7-a1-s17"
    (run / "step_00001000").mkdir(parents=True)
    (run / "step_00001000" / "meta.json").write_text("{}")
    with pytest.raises(sup.SupervisorRefusal, match="partial write"):
        sup.newest_step(run)


# --------------------------------------------------------------------------
# the waves
# --------------------------------------------------------------------------


def test_each_wave_is_four_runs_sharing_one_seed() -> None:
    for wave in range(3):
        runs = sup.wave_runs(wave)
        assert len(runs) == 4
        assert {r.seed for r in runs} == {am.TRAINING_SEEDS[wave]}
        assert len({r.arm_id for r in runs}) == 4


def test_an_out_of_range_wave_is_refused() -> None:
    for wave in (-1, 3):
        with pytest.raises(sup.SupervisorRefusal, match="outside"):
            sup.wave_runs(wave)


# --------------------------------------------------------------------------
# the submission guard
# --------------------------------------------------------------------------


def test_an_unknown_submission_outcome_is_a_refusal() -> None:
    def runner(*_a, **_k):
        class _R:
            returncode = 124
            stdout = ""
            stderr = ""
        return _R()

    with pytest.raises(sup.SupervisorRefusal, match="SUBMISSION_UNKNOWN"):
        sup.submit({"command": "x"}, runner=runner)


def test_a_successful_submission_returns_the_id() -> None:
    def runner(*_a, **_k):
        class _R:
            returncode = 0
            stdout = "Result:\n  job_id: job-abc\n"
            stderr = ""
        return _R()

    assert sup.submit({"command": "x"}, runner=runner) == "job-abc"


def test_the_submitted_record_names_every_wave_zero_run() -> None:
    """The record must cover the four runs actually submitted, by name."""
    repo = Path(__file__).resolve().parents[3]
    record = repo / ".omo" / "evidence" / "task7-arm-matrix-20260917" / "wave0-submitted.json"
    if not record.is_file():
        pytest.skip("no submission record")
    document = json.loads(record.read_text())
    assert set(document["wave_0"]) == {r.run_name for r in sup.wave_runs(0)}
    for name, job_id in document["wave_0"].items():
        assert job_id.startswith("job-"), name
        assert len(job_id) == 40, name
