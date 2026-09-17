"""The matrix runs as twelve 1-node jobs, which is the shape that is known to work.

These tests pin the two decisions that came out of three failed 32-GPU submissions:
every job is ONE node, and every command carries an explicit rendezvous endpoint
and an explicit attempt-free resume. The first is because the multiplexed 4-node
form died in NCCL after a clean preflight and a successful rendezvous, while a
1-node smoke of the same launcher completed 20 steps; the second because
``--standalone`` was what failed before that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.task7 import arm_matrix as am
from scale.experiments.nonlatent_iclr.task7 import emit_matrix_jobs as emj


def _kwargs() -> dict:
    return {"staged_scale_dir": "/S", "launcher": "/L.sh",
            "model_dir": am.SMALL_MODEL_DIR, "token_dir": "/T",
            "save_root": "/R", "tb_path": "/TB"}


# --------------------------------------------------------------------------
# the shape, which is the whole point
# --------------------------------------------------------------------------


def test_every_job_is_exactly_one_node() -> None:
    """One node per job: the shape the smoke proved, not the shape that failed.

    The multiplexed design ran four 8-GPU process groups inside one job and every
    one of them died in NCCL checkTimeout. A single-node job of the same launcher
    completed. This asserts the emitted shape is the one that worked.
    """
    # Given: wave 0's bodies.
    bodies = emj.wave_bodies(wave=0, **_kwargs())
    # When/Then: each asks for one instance, and says so in its config.
    assert len(bodies) == 4
    for body in bodies:
        config = body["framework_config"][0]
        assert config["instance_count"] == 1, body["name"]
        assert emj.NODES_PER_JOB == 1
        assert "NNODES=1 " in str(body["command"])


def test_every_command_sets_the_explicit_rendezvous_endpoint() -> None:
    """--standalone was the first failure; an explicit endpoint fixed it."""
    for body in emj.wave_bodies(wave=0, **_kwargs()):
        command = str(body["command"])
        assert f"MASTER_PORT={emj.BASE_MASTER_PORT}" in command, body["name"]
        # and the port is a real one
        assert 1024 <= emj.BASE_MASTER_PORT <= 65535


def test_the_port_differs_from_the_multiplexed_design() -> None:
    # Given: the port the failed 4-node design used (29511).
    # When/Then: the new base is elsewhere, so a stray process group from an
    # earlier attempt cannot be mistaken for this design's.
    assert emj.BASE_MASTER_PORT != 29511


def test_platform_restart_is_off() -> None:
    """A restart re-runs the same command; the checkpoint, not the platform, resumes."""
    for body in emj.wave_bodies(wave=0, **_kwargs()):
        assert body["auto_fault_tolerance"] is False
        assert body["fault_tolerance_max_retry"] == 0


def test_a_job_carries_no_attempt_or_state_directory() -> None:
    # Given: the multiplexed design's claim/wave machinery.
    # When/Then: none of it survives, because there is nothing to claim -- one job
    # is one node, and the platform places it.
    for body in emj.wave_bodies(wave=0, **_kwargs()):
        command = str(body["command"])
        assert "--attempt" not in command
        assert "--state-dir" not in command
        assert "run_matrix_job" not in command


# --------------------------------------------------------------------------
# the run itself
# --------------------------------------------------------------------------


def test_each_job_names_its_arm_and_seed() -> None:
    bodies = emj.wave_bodies(wave=0, **_kwargs())
    names = {b["name"] for b in bodies}
    assert names == {"task7-a1-s17", "task7-a2-s17", "task7-a3-s17", "task7-a5-s17"}
    for body in bodies:
        assert f"RUN_NAME={body['name']} " in str(body["command"])


def test_the_job_name_is_not_double_prefixed() -> None:
    """run_name already carries the task7- prefix; a second one would be wrong."""
    for body in emj.wave_bodies(wave=0, **_kwargs()):
        assert body["name"].count("task7-") == 1


def test_every_arm_flag_survives_into_the_command() -> None:
    # Given: the arms' declared flags.
    bodies = {b["name"]: b for b in emj.wave_bodies(wave=0, **_kwargs())}
    # When/Then: each command carries exactly its own arm's argv tail.
    for run in am.build_matrix(ngpus=emj.GPUS_PER_JOB):
        if run.seed != am.TRAINING_SEEDS[0]:
            continue
        assert f'EXTRA_ARGS="{run.argv_tail}"' in str(bodies[run.run_name]["command"])


def test_the_budget_is_the_frozen_one() -> None:
    # Given: the frozen 2B budget, and a 256-rows/step schedule at 8 GPUs.
    body = emj.wave_bodies(wave=0, **_kwargs())[0]
    per_step = am.tokens_per_step(ngpus=emj.GPUS_PER_JOB)
    steps = am.steps_for_budget(am.FROZEN_TOKEN_BUDGET, ngpus=emj.GPUS_PER_JOB)
    # When/Then: the job trains exactly that many steps.
    assert f"STEPS={steps} " in str(body["command"])
    assert per_step * steps >= am.FROZEN_TOKEN_BUDGET


def test_each_wave_is_one_seed() -> None:
    """The wave boundary is the seed, so a per-seed interaction cannot masquerade
    as a per-arm one."""
    for wave in range(3):
        bodies = emj.wave_bodies(wave=wave, **_kwargs())
        seeds = {am.TRAINING_SEEDS[wave]}
        for body in bodies:
            assert any(f"-s{s}" in body["name"] for s in seeds), body["name"]


def test_an_out_of_range_wave_is_refused() -> None:
    for wave in (-1, 3):
        with pytest.raises(am.MatrixRefusal, match="outside"):
            emj.wave_bodies(wave=wave, **_kwargs())


# --------------------------------------------------------------------------
# the emitter writes what the submitter reads
# --------------------------------------------------------------------------


def test_main_writes_one_parseable_body_per_run(tmp_path: Path) -> None:
    code = emj.main(["--wave", "0", "--out-dir", str(tmp_path / "b"),
                     "--staged-scale-dir", "/S", "--launcher", "/L.sh",
                     "--token-dir", "/T", "--save-root", "/R", "--tb-path", "/TB"])
    assert code == 0
    written = sorted((tmp_path / "b").glob("*.json"))
    assert len(written) == 4
    for path in written:
        body = json.loads(path.read_text())
        # every field the create API accepts, and nothing it rejects
        assert set(body["framework_config"][0]) == {
            "image", "image_type", "instance_count", "shm_gi", "spec_id"}
        assert body["command"].startswith("bash -lc '")
        assert body["description"]


def test_a_run_name_with_a_separator_is_refused() -> None:
    """A run name becomes a directory; a separator would write outside save_root."""
    run = am.ArmRun(arm_id="A1", seed=17, run_name="task7/a1-s17", steps=1, ngpus=8)
    with pytest.raises(am.MatrixRefusal, match="not usable as a directory"):
        emj.run_body(run, **_kwargs())
