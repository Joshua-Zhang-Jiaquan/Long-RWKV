"""Emit the arm matrix as TWELVE independent 1-node jobs, not one 4-node job.

Why this replaced the single-job design
---------------------------------------

The single 4-node job failed three times, and the second failure localised the
cause: every run reached the launcher's `PREFLIGHT OK`, the data-position fix
fired, torchrun's rendezvous succeeded once given an explicit endpoint --- and
then NCCL died in `checkTimeout`/`ncclCommWatchdog` with `DistBackendError`.

A 1-node smoke with the SAME arm, the SAME launcher and the SAME explicit
endpoint completed 20 steps in 11 minutes.  So the surviving hypothesis is that
the failure is a property of running four 8-GPU process groups inside one job,
not of any arm.  Rather than keep buying 32-GPU evidence to test that, this
emits the shape that is KNOWN to work: one job per run, one node each, with the
platform doing the placement.

What that removes, all of it machinery this program had to build only because it
was multiplexing four runs into one job:

* the node-claim markers (and the stale-claim bug that shifted a launch's index),
* the wave barriers,
* the interaction between platform auto-restart and the claim namespace,
* and the coupling where one arm's failure took three healthy arms with it.

Twelve independent jobs fail independently, resume independently, and each is the
exact shape the smoke validated.  The cost is bookkeeping, not GPU time: the
total is the same 12 runs x 8 GPUs x ~1908 steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from scale.experiments.nonlatent_iclr.task7 import arm_matrix as am
from scale.experiments.nonlatent_iclr.task7.emit_job import REFERENCE_INFRA

SCHEMA: Final = "nonlatent_task7_matrix_jobs_v1"

#: One node per job: the shape the smoke proved, and the shape that leaves
#: placement to the scheduler instead of to a driver this program maintains.
NODES_PER_JOB: Final = 1
GPUS_PER_JOB: Final = 8

#: Distinct from the multiplexed design's 29511 so a stray process group from an
#: earlier attempt cannot be mistaken for this one's.
BASE_MASTER_PORT: Final = 29600


def run_body(run: am.ArmRun, *, staged_scale_dir: str, launcher: str,
             model_dir: str, token_dir: str, save_root: str, tb_path: str,
             steps: int | None = None) -> dict[str, object]:
    """One job body for one arm x seed."""
    steps = run.steps if steps is None else steps
    per_step = am.tokens_per_step(ngpus=GPUS_PER_JOB)
    if not run.run_name or "/" in run.run_name:
        raise am.MatrixRefusal(f"run name {run.run_name!r} is not usable as a directory")
    log_dir = f"{save_root}/logs"
    command = (
        "bash -lc 'set -e; "
        f"export DAN_SCALE_DIR={staged_scale_dir}; "
        f"export MODEL_DIR={model_dir}; "
        f"export TOKEN_DIR={token_dir}; "
        f"export SAVE_ROOT={save_root}; "
        f"export LOGDIR={log_dir}; "
        # The endpoint is explicit rather than --standalone: the launcher uses
        # the explicit form whenever MASTER_PORT is set, and that is the
        # difference between the smoke's 20 completed steps and the multiplexed
        # job's RendezvousTimeoutError.
        f"export MASTER_PORT={BASE_MASTER_PORT}; "
        f"MODE=train RUN_NAME={run.run_name} NNODES={NODES_PER_JOB} "
        f"NGPUS={GPUS_PER_JOB} MICROBATCH={am.DEFAULT_MICROBATCH} "
        f"GRAD_ACCUM={am.DEFAULT_GRAD_ACCUM} STEPS={steps} "
        f"EXTRA_ARGS=\"{run.argv_tail}\" "
        f"bash {launcher}'"
    )
    return {
        "name": run.run_name,
        "description": (
            f"Task-7 controlled arm matrix, arm {run.arm_id} seed {run.seed}: one 1-node "
            f"8-GPU job, {steps} steps at {per_step:,} tokens/step "
            f"({per_step * steps:,} tokens). Emitted this way because the "
            f"multiplexed 4-node form died in NCCL checkTimeout after a clean preflight "
            f"and a successful rendezvous, while a 1-node smoke of the same launcher "
            f"completed. Flags: {run.argv_tail}."),
        "command": command,
        "framework": REFERENCE_INFRA["framework"],
        "framework_config": [{
            "image": REFERENCE_INFRA["image"],
            "image_type": REFERENCE_INFRA["image_type"],
            "instance_count": NODES_PER_JOB,
            "shm_gi": REFERENCE_INFRA["shm_gi"],
            "spec_id": REFERENCE_INFRA["spec_id"],
        }],
        "logic_compute_group_id": REFERENCE_INFRA["logic_compute_group_id"],
        "project_id": REFERENCE_INFRA["project_id"],
        "workspace_id": REFERENCE_INFRA["workspace_id"],
        "max_running_time_ms": REFERENCE_INFRA["max_running_time_ms"],
        "task_priority": 4,
        # Off for the same reason as the multiplexed design, and more so here: an
        # independent job does not need the platform to restart it, because a
        # restart re-runs from the latest checkpoint anyway and the supervisor
        # below decides continuation from the checkpoints on disk.
        "auto_fault_tolerance": False,
        "fault_tolerance_max_retry": 0,
        "fault_tolerance_retry_interval_sec": 300,
        "tb_summary_path": tb_path,
        "enable_notification": False,
        "is_publicpath_readonly": False,
        "envs": [],
    }


def wave_bodies(*, wave: int, **kwargs) -> list[dict[str, object]]:
    """Every run in one wave, each as its own job."""
    waves = am.wave_plan(ngpus_per_run=GPUS_PER_JOB, job_gpus=32)
    if not 0 <= wave < len(waves):
        raise am.MatrixRefusal(f"wave {wave} is outside 0..{len(waves) - 1}")
    return [run_body(run, **kwargs) for run in waves[wave].runs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--staged-scale-dir", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--model-dir", default=am.SMALL_MODEL_DIR)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--tb-path", required=True)
    args = parser.parse_args(argv)
    bodies = wave_bodies(
        wave=args.wave, staged_scale_dir=args.staged_scale_dir, launcher=args.launcher,
        model_dir=args.model_dir, token_dir=args.token_dir, save_root=args.save_root,
        tb_path=args.tb_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for body in bodies:
        path = args.out_dir / f"{body['name']}.json"
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        written.append({"name": body["name"], "path": str(path)})
    print(json.dumps({"schema": SCHEMA, "wave": args.wave, "jobs": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
