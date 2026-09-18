"""The ablation's analysis must not manufacture a result from seed noise.

A1's own three seeds land 0.53 apart in masked cross-entropy. Any between-arm difference
smaller than that is not a finding, and the failure this file guards against is a comparison
that reports one anyway -- which is exactly the "extra compute or a warm-start effect rather
than demonstrated architecture advantage" interpretation the plan names in advance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.task7 import analyse_matrix as am


def _log(tmp_path: Path, name: str, steps_and_ce: list[tuple[int, float]]) -> Path:
    lines = [f"2026-09-18 00:00:{i:02d}  [step {s}] mask_ce={ce} ce_low=1.0 grad=2.0"
             for i, (s, ce) in enumerate(steps_and_ce)]
    path = tmp_path / f"{name}.run.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# reading a run
# --------------------------------------------------------------------------


def test_the_final_step_is_the_one_read(tmp_path: Path) -> None:
    path = _log(tmp_path, "task7-a1-s17", [(20, 7.0), (100, 6.0), (1908, 5.6091)])
    run = am.read_run_log(path)
    assert run.final_step == 1908
    assert run.metrics["mask_ce"] == 5.6091
    assert run.arm == "A1" and run.seed == 17
    assert run.complete is True


def test_a_restarted_log_does_not_read_a_replayed_step_as_the_final_one(
        tmp_path: Path) -> None:
    """The launcher APPENDS across segments, so a resumed run replays earlier steps.

    A log is [.. 1908] then [20 .. 1908] after a restart. Taking the last LINE would read
    step 1908 either way, but taking a later line blindly would read the resumed run's
    step 20 as its final one and file a finished arm as short.
    """
    path = _log(tmp_path, "task7-a1-s17",
                [(1800, 5.7), (1908, 5.6091), (20, 7.0), (500, 6.0)])
    run = am.read_run_log(path)
    # the replayed steps are LOWER, so the highest is still the real final one
    assert run.final_step == 1908
    assert run.metrics["mask_ce"] == 5.6091


def test_a_run_name_without_an_arm_and_seed_is_refused() -> None:
    with pytest.raises(am.AnalysisRefusal, match="does not encode an arm and a seed"):
        am.parse_run_name("task7-matrix")


def test_a_log_with_no_metric_line_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "task7-a1-s17.run.log"
    path.write_text("nothing to see\n", encoding="utf-8")
    with pytest.raises(am.AnalysisRefusal, match="no \\[mask_ce\\] step line"):
        am.read_run_log(path)


def test_the_arm_is_read_off_the_name_not_assumed(tmp_path: Path) -> None:
    for name, arm, seed in (("task7-a3-s29", "A3", 29), ("task7-a5-s43", "A5", 43)):
        run = am.read_run_log(_log(tmp_path, name, [(1908, 5.0)]))
        assert (run.arm, run.seed) == (arm, seed)


# --------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------


def test_the_spread_is_the_range_across_seeds(tmp_path: Path) -> None:
    runs = [am.read_run_log(_log(tmp_path, f"task7-a1-s{s}", [(1908, ce)]))
            for s, ce in ((17, 5.6091), (29, 5.1118), (43, 5.6413))]
    summary = am.summarise(runs)
    assert summary["A1"].n == 3
    assert summary["A1"].spread == pytest.approx(5.6413 - 5.1118, abs=1e-9)
    assert summary["A1"].mean == pytest.approx((5.6091 + 5.1118 + 5.6413) / 3, abs=1e-9)


def test_an_unfinished_run_is_refused_not_silently_included(tmp_path: Path) -> None:
    """A run short of its budget is a cross-entropy from earlier in its own schedule."""
    runs = [am.read_run_log(_log(tmp_path, "task7-a1-s17", [(1908, 5.6)])),
            am.read_run_log(_log(tmp_path, "task7-a1-s29", [(1200, 5.0)]))]
    with pytest.raises(am.AnalysisRefusal, match="stopped short"):
        am.summarise(runs)


# --------------------------------------------------------------------------
# the contrasts
# --------------------------------------------------------------------------


def _summary(**by_arm: dict[int, float]) -> dict[str, am.ArmSummary]:
    return {arm: am.ArmSummary(arm=arm, values=values) for arm, values in by_arm.items()}


def test_a_difference_inside_the_seed_spread_is_not_separable() -> None:
    """The finding this module exists to refuse."""
    # Given: two arms whose difference (0.10) is far inside their own seed spread (0.53).
    summary = _summary(A1={17: 5.6091, 29: 5.1118, 43: 5.6413},
                       A2={17: 5.5091, 29: 5.0118, 43: 5.5413})
    rows = {c["contrast"]: c for c in am.contrasts(summary)}
    directionality = rows["directionality"]
    # When/Then: the difference is reported, and it is NOT separable.
    assert directionality["difference"] == pytest.approx(-0.1, abs=1e-9)
    assert directionality["separable"] is False
    assert "must clear" in directionality["note"] or "seed spreads" in directionality["note"]


def test_a_difference_larger_than_the_spread_is_separable() -> None:
    summary = _summary(A1={17: 5.6091, 29: 5.1118, 43: 5.6413},
                       A2={17: 4.1091, 29: 3.6118, 43: 4.1413})
    rows = {c["contrast"]: c for c in am.contrasts(summary)}
    assert rows["directionality"]["separable"] is True


def test_an_arm_that_was_not_measured_says_so_rather_than_being_skipped() -> None:
    # Given: only A1 has data.
    summary = _summary(A1={17: 5.6, 29: 5.1, 43: 5.6})
    rows = am.contrasts(summary)
    # Then: every declared contrast appears, and the unmeasurable ones are NAMED.
    assert len(rows) == len(am.DECLARED_CONTRASTS)
    for row in rows:
        assert row["status"] == "not_measured"
        assert row["separable"] is None
        assert row["difference"] is None


def test_every_declared_contrast_names_real_arms() -> None:
    """The comparisons are pre-registered, and they must name arms that exist.

    This assertion was written first as `... if hasattr(am, "am") else True`, which is
    decoration: `analyse_matrix` does not import `arm_matrix` under that name, so the
    condition was always false and the assertion always passed. Importing the arms registry
    directly is what makes it able to fail.
    """
    from scale.experiments.nonlatent_iclr.task7 import arm_matrix
    names = {c[0] for c in am.DECLARED_CONTRASTS}
    assert names == {"directionality", "loop_on_bidirectional", "loop_on_forward_only"}
    for _, arm, reference in am.DECLARED_CONTRASTS:
        assert arm in arm_matrix.ARMS, arm
        assert reference in arm_matrix.ARMS, reference


def test_completion_is_read_from_the_checkpoint_not_the_log(tmp_path: Path) -> None:
    r"""The trainer logs every 20 steps, so a finished 1908 run's last line is 1900.

    Reading completion off the log filed every FINISHED run as short, and the module
    refused on that evidence -- which is the same defect class as accepting bad evidence,
    one direction over: a check that blocks what it should pass destroys a correct result.
    """
    from dataclasses import replace
    # Given: a run whose log ends at 1900 but whose checkpoint is at 1908.
    run = am.read_run_log(_log(tmp_path, "task7-a1-s17", [(1880, 5.62), (1900, 5.6091)]))
    assert run.complete is False, "the log alone cannot establish completion"
    # When: the checkpoint step is supplied.
    completed = replace(run, checkpoint_step=1908)
    # Then: it is complete, and the metric still comes from the log.
    assert completed.complete is True
    assert completed.metrics["mask_ce"] == 5.6091
    # and a genuinely short run is still short even with a checkpoint
    short = replace(am.read_run_log(_log(tmp_path, "task7-a2-s17", [(1000, 6.0)])),
                    checkpoint_step=1000)
    assert short.complete is False
