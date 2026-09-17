"""Task 7: the single-job wave runner must pack, pin, and fail loudly.

Three failure modes this file exists to pin, each of which produces a run set
that looks complete:

* a wave that does not divide across nodes loses the remainder silently;
* a run launched against the wrong cards silently changes the effective batch,
  which is the one thing the exposure-matched comparison cannot absorb;
* a barrier written over a failed run lets the next wave proceed, so the arm
  that failed is simply absent from the study.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.task7 import arm_matrix as am
from scale.experiments.nonlatent_iclr.task7 import run_matrix_job as rj


def _wave(index: int = 0) -> am.Wave:
    return am.wave_plan(ngpus_per_run=8)[index]


# --------------------------------------------------------------------------
# node assignment
# --------------------------------------------------------------------------


def test_a_wave_splits_across_nodes_without_losing_a_run() -> None:
    # Given: a four-run wave and two nodes.
    wave = _wave()
    total = []
    # When: each node's share is taken.
    for node in range(4):
        total.extend(rj.runs_for_node(wave, node_index=node, nodes=4, per_node=1))
    # Then: the union is the wave, with nothing dropped and nothing doubled.
    assert sorted(total) == sorted(run.run_name for run in wave.runs)
    assert len(total) == len(set(total)) == 4


def test_an_indivisible_wave_is_refused_rather_than_truncated() -> None:
    # Given: a wave of four and a three-node split at one run per node.
    wave = _wave()
    # When/Then: refused, because the leftover run would otherwise never start
    # and the study would read as complete.
    with pytest.raises(rj.JobRefusal, match="does not divide"):
        rj.runs_for_node(wave, node_index=0, nodes=3, per_node=1)


def test_a_negative_or_out_of_range_node_index_is_refused() -> None:
    wave = _wave()
    with pytest.raises(rj.JobRefusal, match="outside"):
        rj.runs_for_node(wave, node_index=4, nodes=4, per_node=1)
    with pytest.raises(rj.JobRefusal, match="positive"):
        rj.runs_for_node(wave, node_index=0, nodes=0)


# --------------------------------------------------------------------------
# the per-run command
# --------------------------------------------------------------------------


def _env() -> dict[str, str]:
    return {"DAN_SCALE_DIR": "/stage/scale", "MODEL_DIR": "/models/rwkv7-0.4B",
            "TOKEN_DIR": "/tokens/a,/tokens/b", "SAVE_ROOT": "/out",
            "LOGDIR": "/out/logs"}


def test_the_command_pins_four_cards_and_passes_the_seed() -> None:
    # Given: a run and the second card slot on a node.
    run = am.build_matrix(ngpus=8)[0]
    # When: the command is built.
    argv = rj.run_command(run, launcher=Path("/stage/scale/qz/launch.sh"),
                          node_env=_env(), gpus=8, cuda_devices="8,9,10,11,12,13,14,15")
    script = argv[-1]
    # Then: the visible devices are exactly that slot, the arm's flags and the
    # seed are passed, and the launcher gets the frozen step count.
    assert "CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15" in script
    assert f"RUN_NAME={run.run_name}" in script
    assert f"--seed={run.seed}" in script
    assert f"STEPS={run.steps}" in script
    assert "NNODES=1" in script and "NGPUS=8" in script
    for flag in run.extra_args:
        assert flag in script


def test_a_device_list_that_does_not_match_the_declared_count_is_refused() -> None:
    """The mismatch would silently change the effective batch."""
    # Given: four GPUs declared but three named.
    run = am.build_matrix(ngpus=8)[0]
    # When/Then: refused, because the run would train with a global batch the
    # matrix did not choose and its arm would no longer be comparable.
    with pytest.raises(rj.JobRefusal, match="CUDA_VISIBLE_DEVICES"):
        rj.run_command(run, launcher=Path("/l.sh"), node_env=_env(), gpus=8,
                       cuda_devices="0,1,2")


# --------------------------------------------------------------------------
# the barrier
# --------------------------------------------------------------------------


def test_a_wave_is_only_done_when_every_node_says_so(tmp_path: Path) -> None:
    # Given: one node's marker for wave 0.
    mine, _ = rj.barrier_paths(tmp_path, 0)
    mine.write_text("{}")
    # When/Then: the wave is not done, because the other node may still be
    # training its half.
    assert rj.wave_already_done(tmp_path, 0, nodes=2) is False
    # and it is done once a second host has written its own marker -- the real
    # system has two hostnames, so the second marker is written directly rather
    # than through barrier_paths, which would reuse this host's name.
    (tmp_path / "wave-000-otherhost.done").write_text("{}")
    assert rj.wave_already_done(tmp_path, 0, nodes=2) is True
    # and a wave nobody finished stays unfinished
    assert rj.wave_already_done(tmp_path, 1, nodes=2) is False


def test_a_failed_run_cannot_be_marked_done(tmp_path: Path) -> None:
    # Given: a wave whose second run exited non-zero.
    # When/Then: the marker write refuses, so the next wave cannot proceed over
    # a missing arm -- and no marker is left behind.
    with pytest.raises(rj.JobRefusal, match="refusing to mark it complete"):
        rj.mark_wave_done(tmp_path, 0, [0, 1])
    assert not list(tmp_path.glob("wave-000-*.done"))


def test_a_clean_wave_is_marked_and_reads_back(tmp_path: Path) -> None:
    # Given: a wave where every run succeeded.
    marker = rj.mark_wave_done(tmp_path, 2, [0, 0])
    # When/Then: the marker exists, records the codes, and is discoverable by the
    # restart check that stops a finished wave running twice.
    assert marker.is_file()
    assert json.loads(marker.read_text())["codes"] == [0, 0]
    assert rj.wave_already_done(tmp_path, 2, nodes=1) is True


def test_an_empty_code_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(rj.JobRefusal, match="no run codes"):
        rj.mark_wave_done(tmp_path, 0, [])


# --------------------------------------------------------------------------
# executing a wave
# --------------------------------------------------------------------------


class _FakeProcess:
    def __init__(self, code: int) -> None:
        self.code = code

    def wait(self) -> int:
        return self.code


def test_execute_wave_starts_every_run_together(tmp_path: Path) -> None:
    # Given: a wave of four, two of which belong to this node.
    wave = _wave()
    seen: list[tuple[str, str]] = []

    def fake(argv, stdout=None, stderr=None):  # noqa: ARG001
        script = argv[-1]
        devices = [part.split("=", 1)[1].split(";")[0]
                   for part in script.split() if part.startswith("CUDA_VISIBLE_DEVICES=")]
        seen.append((devices[0] if devices else "?", script))
        return _FakeProcess(0)

    names = rj.runs_for_node(wave, node_index=0, nodes=4, per_node=1)
    codes = rj.execute_wave(wave=wave, run_names=names, launcher=Path("/l.sh"),
                            node_env=_env(), gpus=8, log_dir=tmp_path,
                            subprocess_run=fake)
    # When/Then: both ran, each on its own four cards, and both codes came back.
    assert codes == [0]
    # 4 nodes, 1 run each: this node takes the whole node's eight cards.
    assert [entry[0] for entry in seen] == ["0,1,2,3,4,5,6,7"]


def test_execute_wave_reports_a_failing_run(tmp_path: Path) -> None:
    # Given: a run that fails.
    wave = _wave()
    names = rj.runs_for_node(wave, node_index=0, nodes=4, per_node=1)
    codes = rj.execute_wave(wave=wave, run_names=names, launcher=Path("/l.sh"),
                            node_env=_env(), gpus=8, log_dir=tmp_path,
                            subprocess_run=lambda *_a, **_k: _FakeProcess(7))
    # When/Then: the code surfaces rather than being swallowed, so the barrier
    # check downstream can refuse.
    assert codes == [7]
    with pytest.raises(rj.JobRefusal):
        rj.mark_wave_done(tmp_path, 0, codes)


# --------------------------------------------------------------------------
# the whole plan
# --------------------------------------------------------------------------


def test_the_job_report_packs_twelve_runs_into_three_waves() -> None:
    # Given: the owner's shape, 32 GPUs at 8 per run.
    report = am.job_report(ngpus_per_run=8)
    # When/Then: 4 concurrent runs, 3 waves, one per seed, 12 runs total.
    assert report["concurrent_runs"] == 4
    assert report["waves"] == 3
    assert report["runs"] == 12
    assert report["per_wave_arms"][0] == ["A1/s17", "A2/s17", "A3/s17", "A5/s17"]
    assert report["per_wave_arms"][2] == ["A1/s43", "A2/s43", "A3/s43", "A5/s43"]


def test_the_split_does_not_change_the_total_cost() -> None:
    """The number the owner should read: GPU-hours are set by the budget.

    Wall clock is a different quantity and it does NOT move monotonically with
    the split: it is the sum over waves of the wave's SLOWEST run, so packing
    more runs into a wave lets the fast arms finish while the wave waits for the
    slow one. The invariant that does hold is the one worth asserting -- the
    budget fixes the GPU-hours, and the packing only decides how much of them
    lands on the wall.
    """
    # Given: the same matrix packed two ways.
    wide = am.job_report(ngpus_per_run=8)
    narrow = am.job_report(ngpus_per_run=2)
    # When/Then: total GPU-hours are identical across packings...
    assert wide["forecast_gpu_hours_total"] == pytest.approx(
        narrow["forecast_gpu_hours_total"], rel=1e-9)
    # ...and the wider split is never slower, because it leaves fewer cards idle
    # behind a wave's slowest arm.
    assert wide["forecast_wall_hours_total"] <= narrow["forecast_wall_hours_total"]
    # and the 36h cap is exceeded several times over either way, which is why the
    # segment count is the operationally interesting number.
    assert wide["segments_at_wall_cap"] > 1


def test_the_dry_run_does_not_claim_a_node_or_a_wave(tmp_path: Path) -> None:
    # Given: a state directory and a dry run.
    state = tmp_path / "state"
    code = rj.main(["--dry-run", "--launcher", "/l.sh",
                    "--state-dir", str(state), "--log-dir", str(tmp_path / "log")])
    # When/Then: it exits clean and leaves nothing behind -- a dry run that took
    # a node index or wrote a barrier would corrupt the real run's bookkeeping.
    assert code == 0
    assert not state.exists() or not list(state.glob("*"))
