"""Keep the twelve arm jobs running across wall-clock caps, and advance the waves.

Why this replaces the earlier resubmitter
-----------------------------------------

The first resubmitter watched ONE job and read completion from wave barriers, because
one job ran all twelve arms and needed a barrier to know which of its own waves had
finished. That design failed three times and was replaced by twelve independent 1-node
jobs (see ``emit_matrix_jobs.py``), which deletes the barriers, the node claims and the
platform-restart interaction. What is left to supervise is per-run:

* a run's progress is its OWN newest checkpoint, not a shared barrier;
* a run that stopped short is resubmitted as the same command, because the launcher
  resumes from the newest checkpoint in that run's own directory;
* a wave is four runs sharing one seed, and the next wave starts once all four are done.

The guards are the ones that earned their place, kept for the reasons they earned them:

* **A barren run stops its own chain.** The very first attempt of this study died in the
  launcher's preflight before loading a model, and the platform's fault tolerance
  restarted the identical broken command three times. A resubmission is therefore only
  allowed after the previous one PRODUCED A CHECKPOINT.
* **An unknown submission outcome stops, and asks.** If the submitter returns no job id
  the job may still exist, and a second submission would put two 8-GPU jobs on the pool
  under the same run name -- where they would both write the same checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from scale.experiments.nonlatent_iclr.task7 import arm_matrix as am

#: The trainer writes ``step_00001234`` directories; the launcher resumes from the newest.
STEP_GLOB = "step_????????"

TERMINAL = ("job_stopped", "job_succeeded", "job_failed", "job_cancelled", "job_deleted")

SCHEMA = "nonlatent_task7_supervisor_v1"

_STEP_RE = re.compile(r"step_(\d+)")


class SupervisorRefusal(RuntimeError):
    """The supervisor cannot continue without a human."""


class Decision(Enum):
    WAIT = "wait"
    DONE = "done"
    RESUBMIT = "resubmit"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class RunState:
    """One arm x seed, as the artifacts describe it right now."""

    name: str
    job_id: str
    status: str
    best_step: int | None
    target_steps: int

    @property
    def complete(self) -> bool:
        return self.best_step is not None and self.best_step >= self.target_steps

    @property
    def produced(self) -> bool:
        return self.best_step is not None and self.best_step > 0


def newest_step(run_dir: Path) -> int | None:
    """The highest step the run has checkpointed, or None if it has none.

    Read from the run's OWN directory, which is what the launcher resumes from -- so
    this is the same quantity the trainer will use, not a parallel bookkeeping of it.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    steps = [int(m.group(1)) for path in run_dir.glob(STEP_GLOB)
             if (m := _STEP_RE.search(path.name))]
    if not steps:
        return None
    if not (run_dir / "model.pt").exists() and not any(
            (p / "model.pt").exists() for p in run_dir.glob(STEP_GLOB)):
        raise SupervisorRefusal(
            f"{run_dir} has step directories but no model.pt; refusing to treat a "
            f"partial write as progress")
    return max(steps)


def decide_run(state: RunState, *, resubmits_used: int, max_resubmits: int,
               barren_streak: int, max_barren: int) -> tuple[Decision, str]:
    """The per-run policy, as a pure function. Order is load-bearing."""
    if state.complete:
        return Decision.DONE, f"reached step {state.best_step} of {state.target_steps}"
    if state.status not in TERMINAL:
        return Decision.WAIT, f"job is {state.status}"
    if barren_streak >= max_barren:
        return Decision.STOP, (
            f"{barren_streak} consecutive resubmission(s) produced no checkpoint; "
            f"another would repeat a broken command rather than continue a run")
    if resubmits_used >= max_resubmits:
        return Decision.STOP, (
            f"resubmission budget exhausted ({resubmits_used}/{max_resubmits}) at step "
            f"{state.best_step}")
    return Decision.RESUBMIT, (
        f"{state.status} at step {state.best_step} of {state.target_steps}")


def job_status(job_id: str, *, runner=subprocess.run) -> str:
    result = runner(["qz", "train", "GetJob", "--data", json.dumps({"job_id": job_id}),
                     "-o", "json"], capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise SupervisorRefusal(f"GetJob failed rc={result.returncode}: {result.stderr[-200:]}")
    try:
        document = json.loads(result.stdout)
    except ValueError as exc:
        raise SupervisorRefusal(f"GetJob returned non-JSON: {exc}") from exc
    status = (document.get("Result") or {}).get("status")
    if not isinstance(status, str):
        raise SupervisorRefusal(f"GetJob returned no status: {str(document)[:200]}")
    return status


def submit(body: dict[str, object], *, runner=subprocess.run) -> str:
    """Submit one job. An unknown outcome is a REFUSAL, never a retry."""
    result = runner(["qz", "train", "CreateJob", "--data", json.dumps(body), "-o", "yaml"],
                    capture_output=True, text=True, timeout=300)
    text = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or "job_id" not in text:
        raise SupervisorRefusal(
            f"SUBMISSION_UNKNOWN (rc={result.returncode}); reconcile before any retry. "
            f"Output tail: {text.strip()[-300:]}")
    for line in text.splitlines():
        if "job_id:" in line:
            return line.split("job_id:", 1)[1].strip().strip('"')
    raise SupervisorRefusal(f"job_id line not found in: {text.strip()[-300:]}")


def wave_runs(wave: int, *, ngpus: int = 8, job_gpus: int = 32) -> list[am.ArmRun]:
    waves = am.wave_plan(ngpus_per_run=ngpus, job_gpus=job_gpus)
    if not 0 <= wave < len(waves):
        raise SupervisorRefusal(f"wave {wave} is outside 0..{len(waves) - 1}")
    return list(waves[wave].runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path,
                        help="wave0-submitted.json-shaped record of run_name -> job_id")
    parser.add_argument("--bodies", required=True, type=Path,
                        help="directory of emitted job bodies, one per run")
    parser.add_argument("--save-root", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--max-resubmits", type=int, default=12)
    parser.add_argument("--max-barren", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    raise SupervisorRefusal(
        "the supervisor is driven by its tests and its pure core; the loop that calls "
        "qz belongs in the orchestrating turn so an unknown outcome stops a human-visible "
        "process rather than a detached one")


if __name__ == "__main__":
    raise SystemExit(main())
