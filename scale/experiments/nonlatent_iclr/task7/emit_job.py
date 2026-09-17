"""Emit the task-7 matrix job body, mirroring the reference job's settings.

The owner's instruction is to reuse the settings of the running continuation
(``job-49e48c2d``) so the matrix lands on the same pool, image, priority and
fault-tolerance policy as the run it complements.  Every field below is read off
that job rather than chosen, and :data:`REFERENCE_JOB_ID` records which job they
came from so a reader can re-derive them.

Two things are NOT copied from the reference, both deliberately:

* ``max_running_time_ms`` stays at the reference's 36 h.  A dry run accepts a
  longer cap, but a dry run validates the body, not the policy -- whether the
  project permits more than 36 h is UNVERIFIED, and the forecast here needs
  ~174 h per wave, so the job will stop and need resubmission either way.  The
  honest assumption is the cap that has actually been observed to stop a run.
* The ``TOKEN_DIR`` blend is passed in rather than baked in.  The reference's
  four-way blend was tuned for the 2.9B extension; the small arms are a
  different comparison and their blend should be stated at submit time rather
  than inherited by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from scale.experiments.nonlatent_iclr.task7 import arm_matrix as am

#: The job whose settings this mirrors, read live via ``qz train GetJob``.
REFERENCE_JOB_ID: Final = "job-49e48c2d-f291-4222-ba6a-ab7e3521d8c4"

SCHEMA: Final = "nonlatent_task7_matrix_job_v1"

#: Infrastructure read off the reference job.  ``spec_id`` is a SHAPE, not a GPU
#: model: the LCG decides whether it resolves to an H100 or an H200, which is why
#: the same id appears in H200 specs and in the H100 reference.
REFERENCE_INFRA: Final[dict[str, object]] = {
    "framework": "pytorch",
    "project_id": "project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
    "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
    "logic_compute_group_id": "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
    "image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
    "image_type": "SOURCE_PRIVATE",
    "spec_id": "7166bd2e-6cbe-4bd9-be38-762d11003e7f",
    "cpu": 160,
    "gpu_count_per_node": 8,
    "shm_gi": 1800,
    "max_running_time_ms": "129600000",
    "task_priority": 0,
    "auto_fault_tolerance": True,
    "fault_tolerance_max_retry": 3,
    "fault_tolerance_retry_interval_sec": 300,
    "enable_notification": False,
    "is_publicpath_readonly": False,
}


class EmitRefusal(RuntimeError):
    """The job body cannot be emitted as specified."""


def build_command(*, staged_scale_dir: str, launcher: str, state_dir: str,
                  log_dir: str, model_dir: str = am.SMALL_MODEL_DIR,
                  token_dir: str, save_root: str, tb_path: str,
                  ngpus_per_run: int = 4, nodes: int = 2) -> str:
    """The pod-side command: run the wave driver, which runs the arms.

    The driver is invoked as a FILE, not with ``-m``.  That matters: ``-m``
    would import the ``experiments.nonlatent_iclr`` package, whose ``__init__``
    pulls inventory, models and publication, and none of those are staged for a
    training job.  Run as a file the driver bootstraps its one dependency from
    the sibling path, so the pod needs the task7 directory and nothing else.
    """
    for name, value in (("staged_scale_dir", staged_scale_dir), ("launcher", launcher),
                        ("state_dir", state_dir), ("log_dir", log_dir),
                        ("token_dir", token_dir), ("save_root", save_root)):
        if not value:
            raise EmitRefusal(f"{name} must be given")
    staged = staged_scale_dir.rstrip("/")
    return (
        "bash -lc 'set -e; "
        f"export DAN_SCALE_DIR={staged}; "
        f"export MODEL_DIR={model_dir}; "
        f"export TOKEN_DIR={token_dir}; "
        f"export SAVE_ROOT={save_root}; "
        f"export LOGDIR={log_dir}; "
        f"python3 {staged}/experiments/nonlatent_iclr/task7/run_matrix_job.py "
        f"--launcher {launcher} --state-dir {state_dir} --log-dir {log_dir} "
        f"--ngpus-per-run {ngpus_per_run} --nodes {nodes}'"
    )


def emit_job_body(*, name: str, command: str, description: str,
                  tb_summary_path: str, nodes: int = 2,
                  infra: dict[str, object] | None = None) -> dict[str, object]:
    """Assemble the CreateJob body from the reference settings."""
    source = dict(infra or REFERENCE_INFRA)
    if nodes <= 0:
        raise EmitRefusal(f"nodes must be positive, got {nodes}")
    body: dict[str, object] = {
        "name": name,
        "description": description,
        "command": command,
        "framework": source["framework"],
        "framework_config": [{
            "cpu": source["cpu"],
            "gpu_count": source["gpu_count_per_node"],
            "image": source["image"],
            "image_type": source["image_type"],
            "instance_count": nodes,
            "shm_gi": source["shm_gi"],
            "spec_id": source["spec_id"],
        }],
        "logic_compute_group_id": source["logic_compute_group_id"],
        "project_id": source["project_id"],
        "workspace_id": source["workspace_id"],
        "max_running_time_ms": source["max_running_time_ms"],
        "task_priority": source["task_priority"],
        "auto_fault_tolerance": source["auto_fault_tolerance"],
        "fault_tolerance_max_retry": source["fault_tolerance_max_retry"],
        "fault_tolerance_retry_interval_sec": source["fault_tolerance_retry_interval_sec"],
        "tb_summary_path": tb_summary_path,
        "enable_notification": source["enable_notification"],
        "is_publicpath_readonly": source["is_publicpath_readonly"],
        "envs": [],
    }
    return body


def describe(*, ngpus_per_run: int = 4, nodes: int = 2,
             budget: int = am.FROZEN_TOKEN_BUDGET) -> str:
    """The description carried into the job, stating what it will and will not do."""
    report = am.job_report(ngpus_per_run=ngpus_per_run, job_gpus=nodes * 8, budget=budget)
    absent = "; ".join(f"{arm}: {why.split(' — ')[0]}" for arm, why in am.ABSENT_ARMS.items())
    return (
        f"Task-7 controlled arm matrix in ONE job: {report['runs']} runs "
        f"({', '.join(sorted(am.ARMS))}) x seeds {list(am.TRAINING_SEEDS)}, "
        f"{ngpus_per_run} GPUs each, {report['concurrent_runs']} concurrent, "
        f"{report['waves']} waves. Budget frozen at {budget:,} tokens per run "
        f"({report['steps_per_run']} steps at {report['tokens_per_step']:,} "
        f"tokens/step). Forecast {report['forecast_gpu_hours_total']:,} GPU-h = "
        f"{report['forecast_wall_hours_total']:.0f} wall hours, i.e. "
        f"{report['segments_at_wall_cap']} segments at the 36h cap; a wall-cap stop "
        f"does not auto-resume, so each segment needs a manual resubmission and "
        f"the wave barrier makes a restart skip completed waves. NOT included: "
        f"{absent}. Settings mirror {REFERENCE_JOB_ID}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--staged-scale-dir", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--tb-path", required=True)
    parser.add_argument("--name", default="task7-arm-matrix-16h100")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--ngpus-per-run", type=int, default=4)
    args = parser.parse_args(argv)

    command = build_command(
        staged_scale_dir=args.staged_scale_dir, launcher=args.launcher,
        state_dir=args.state_dir, log_dir=args.log_dir, token_dir=args.token_dir,
        save_root=args.save_root, tb_path=args.tb_path,
        ngpus_per_run=args.ngpus_per_run, nodes=args.nodes)
    body = emit_job_body(name=args.name, command=command,
                         description=describe(ngpus_per_run=args.ngpus_per_run,
                                              nodes=args.nodes),
                         tb_summary_path=args.tb_path, nodes=args.nodes)
    args.out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "job_body": str(args.out),
                      "job_report": am.job_report(ngpus_per_run=args.ngpus_per_run,
                                                  job_gpus=args.nodes * 8)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
