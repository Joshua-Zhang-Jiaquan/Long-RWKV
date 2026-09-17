"""Run the task-7 arm matrix as ONE job, in waves of concurrent runs.

The owner's decision: 8 H100 per run, 4 runs at once, so one job spans
4 nodes x 8 = 32 H100 and the 12-run matrix is 3 waves of 4.

Why the design is what it is
----------------------------

* **8 GPUs per run is a fixed matrix-wide constant, not a per-run choice.**
  Microbatch 8 x accum 4 at 8 cards is 1,048,576 tokens/step, which is exactly
  the global batch the 2.9B continuation trains at. Mixing card counts across
  arms would change that batch and the arms would stop being exposure-matched,
  so every run in the study gets the same count even though the machine could
  hold a different one.
* **Each run is its own single-node ``torchrun --standalone`` group.** Verified
  empirically that ``--standalone`` ignores ``MASTER_PORT`` and discovers a free
  port per invocation, so concurrent groups on one node do not collide and no
  port has to be reserved or threaded through.
* **Nodes claim their cards through a marker directory, not an environment
  variable.** A 4-node job runs the same command on every host and nothing
  assigns them an index, so the first node to create its claim file takes node 0.
  The claim is create-exclusive, so a host that loses the race cannot silently
  take the same slot. With 4 nodes and 4 concurrent runs each node takes exactly
  one run.
* **Waves are separated by a file barrier.** The next wave must not start until
  every node has finished the current one, because the run-to-card assignment
  changes between waves and two nodes launching from different waves would put
  two runs on the same four cards.

A wall-cap stop does not auto-resume, and the forecast here is ~87 h per wave
against a 36 h cap, so this job will stop and need manual resubmission about 9
times. ``--resume-from`` makes each restart continue rather than restart, and the
barrier directory is deliberately left in place across restarts so a resumed job
can tell which waves already finished.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


class JobRefusal(RuntimeError):
    """The in-pod matrix runner cannot proceed."""


def _load_arm_matrix():
    """Import the arm matrix, whether run as a module or as a plain script.

    This file runs INSIDE a pod from the staged tree, and that tree carries only
    the modules a training job needs: it has no ``experiments.nonlatent_iclr``
    package, and the repository's ``__init__`` for it pulls inventory, models and
    publication with dependencies a pod need not have. Importing the sibling file
    by path sidesteps the package chain entirely, which is the same technique the
    eval scripts already use for their own code root.

    The fallback is deliberately narrow: it only engages when the package import
    fails, so in the repository -- where the package exists -- the normal import
    path is what runs and a mistake here cannot mask a real import error.
    """
    try:
        from scale.experiments.nonlatent_iclr.task7 import arm_matrix as module
        return module
    except ModuleNotFoundError:
        sibling = Path(__file__).resolve().with_name("arm_matrix.py")
        if not sibling.is_file():
            msg = (f"cannot import the arm matrix as a package and {sibling} is "
                   f"missing; the staged tree must carry task7/ beside this file")
            raise JobRefusal(msg) from None
        spec = importlib.util.spec_from_file_location("task7_arm_matrix", sibling)
        if spec is None or spec.loader is None:
            msg = f"cannot load {sibling} as a module"
            raise JobRefusal(msg) from None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
        return loaded


am = _load_arm_matrix()

SCHEMA = "nonlatent_task7_matrix_wave_v1"
NODES = 4


def runs_for_node(wave: am.Wave, *, node_index: int, nodes: int = NODES,
                  per_node: int | None = None) -> tuple[str, ...]:
    """The runs this node owns in this wave, by name.

    Split contiguously so each node's runs are a stable slice: a run that moved
    between nodes across waves would see a different set of peers mid-training.
    """
    if nodes <= 0:
        raise JobRefusal(f"nodes must be positive, got {nodes}")
    if not 0 <= node_index < nodes:
        raise JobRefusal(f"node index {node_index} outside 0..{nodes - 1}")
    names = [run.run_name for run in wave.runs]
    if per_node is None:
        per_node = len(names) // nodes
    if per_node * nodes < len(names):
        # The remainder would otherwise be dropped silently, which is the
        # failure that makes a matrix look complete when a run never started.
        raise JobRefusal(
            f"wave {wave.index} has {len(names)} runs, which does not divide "
            f"across {nodes} nodes at {per_node} per node")
    start = node_index * per_node
    return tuple(names[start:start + per_node])


def run_command(run: am.ArmRun, *, launcher: Path, node_env: dict[str, str],
                gpus: int = 8, cuda_devices: str = "0,1,2,3,4,5,6,7") -> list[str]:
    """The argv for one arm run, pinned to ``gpus`` visible cards.

    ``EXTRA_ARGS`` carries the arm's flags plus its seed.  The launcher's
    preflight validates every flag against the trainer's argparse, so a typo
    fails in the first seconds rather than after a load.
    """
    if gpus != len([d for d in cuda_devices.split(",") if d.strip()]):
        msg = (f"{gpus} GPUs per run but CUDA_VISIBLE_DEVICES={cuda_devices!r} "
               f"names a different count")
        raise JobRefusal(msg)
    extra = " ".join((*run.extra_args, f"--seed={run.seed}"))
    script = (
        "set -e; "
        f"export CUDA_VISIBLE_DEVICES={cuda_devices}; "
        f"export DAN_SCALE_DIR={node_env['DAN_SCALE_DIR']}; "
        f"export MODEL_DIR={node_env['MODEL_DIR']}; "
        f"export TOKEN_DIR={node_env['TOKEN_DIR']}; "
        f"export SAVE_ROOT={node_env['SAVE_ROOT']}; "
        f"export LOGDIR={node_env['LOGDIR']}; "
        f"MODE=train RUN_NAME={run.run_name} NNODES=1 NGPUS={gpus} "
        f"MICROBATCH={am.DEFAULT_MICROBATCH} GRAD_ACCUM={am.DEFAULT_GRAD_ACCUM} "
        f"STEPS={run.steps} EXTRA_ARGS=\"{extra}\" "
        f"bash {launcher}"
    )
    return ["bash", "-lc", script]


def barrier_paths(root: Path, wave_index: int) -> tuple[Path, Path]:
    """(this node's done-marker, the glob naming every node's marker)."""
    return (root / f"wave-{wave_index:03d}-{socket.gethostname()}.done",
            root / f"wave-{wave_index:03d}-*.done")


def wave_already_done(root: Path, wave_index: int, *, nodes: int = NODES) -> bool:
    """Has the wave completed on every node already?

    Checked on restart: the job resumes after a wall-cap stop, and a wave that
    finished before the stop must not run again -- re-running it would spend the
    cards a second time and, worse, overwrite the checkpoints it produced.
    """
    return len(list(root.glob(f"wave-{wave_index:03d}-*.done"))) >= nodes


def claim_node_index(root: Path, *, nodes: int = NODES, timeout_s: float = 900.0,
                     poll_s: float = 5.0) -> int:
    """Claim a node slot by creating a file only we can create.

    Create-exclusive so two hosts cannot take the same index.  Waits for our own
    claim to appear in the sorted list, which is what makes the index stable for
    the rest of the job.
    """
    root.mkdir(parents=True, exist_ok=True)
    claim = root / f"node-{socket.gethostname()}.claim"
    try:
        with claim.open("x", encoding="utf-8") as handle:
            _ = handle.write(json.dumps({"host": socket.gethostname()}))
    except FileExistsError:
        pass  # our own claim from an earlier attempt in this same job
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        claims = sorted(path.name for path in root.glob("node-*.claim"))
        if claim.name in claims:
            return claims.index(claim.name)
        time.sleep(poll_s)
    msg = (f"node {socket.gethostname()} could not settle into a node index "
           f"within {timeout_s:.0f}s under {root}")
    raise JobRefusal(msg)


def execute_wave(*, wave: am.Wave, run_names: tuple[str, ...], launcher: Path,
                 node_env: dict[str, str], gpus: int, log_dir: Path,
                 subprocess_run=subprocess.Popen) -> list[int]:
    """Launch this node's runs for one wave and wait for all of them.

    All of a node's runs in a wave start together, because a wave ends when its
    slowest run does; starting them serially would make the wave the sum of its
    runs rather than its maximum.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    by_name = {run.run_name: run for run in wave.runs}
    processes = []
    for slot, name in enumerate(run_names):
        run = by_name[name]
        # Each run on this node gets its own four contiguous cards.
        cuda = ",".join(str(index) for index in range(slot * gpus, (slot + 1) * gpus))
        argv = run_command(run, launcher=launcher, node_env=node_env, gpus=gpus,
                          cuda_devices=cuda)
        log = (log_dir / f"{name}.run.log").open("ab")
        processes.append((name, subprocess_run(argv, stdout=log, stderr=log)))
    codes = []
    for name, process in processes:
        codes.append(process.wait())
    return codes


def mark_wave_done(root: Path, wave_index: int, codes: list[int]) -> Path:
    """Write the wave's done-marker, refusing if any run in it failed.

    A marker written over a failure is how a failed arm becomes a missing arm:
    the next wave starts, the barrier is satisfied, and nothing in the run set
    says that one of the twelve never completed. So the write is conditional on
    success and the caller must treat the refusal as a stop.
    """
    if not codes:
        raise JobRefusal(f"wave {wave_index} reported no run codes")
    failed = [code for code in codes if code != 0]
    if failed:
        msg = (f"wave {wave_index} had failing runs {failed}; refusing to mark it "
               f"complete, because the barrier would then let the next wave "
               f"proceed over a missing arm")
        raise JobRefusal(msg)
    marker, _ = barrier_paths(root, wave_index)
    marker.write_text(json.dumps({"codes": codes}), encoding="utf-8")
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path,
                        help="GPFS directory for claims and wave barriers")
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--ngpus-per-run", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=NODES)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without launching")
    args = parser.parse_args(argv)

    node_env = {
        "DAN_SCALE_DIR": os.environ.get("DAN_SCALE_DIR", ""),
        "MODEL_DIR": os.environ.get("MODEL_DIR", am.SMALL_MODEL_DIR),
        "TOKEN_DIR": os.environ.get("TOKEN_DIR", ""),
        "SAVE_ROOT": os.environ.get("SAVE_ROOT", ""),
        "LOGDIR": os.environ.get("LOGDIR", ""),
    }
    waves = am.wave_plan(ngpus_per_run=args.ngpus_per_run, job_gpus=args.nodes * 8)
    am.require_wave_coverage(waves)
    per_node = (args.nodes * 8 // args.ngpus_per_run) // args.nodes

    if args.dry_run:
        # No claim, no barrier: a dry run must not consume a node index or make
        # a wave look finished to a later real run.
        plan = [[{"wave": wave.index, "node": node,
                  "runs": list(runs_for_node(wave, node_index=node,
                                             nodes=args.nodes, per_node=per_node))}
                 for node in range(args.nodes)] for wave in waves]
        print(json.dumps({"schema": SCHEMA, "runs": sum(len(w.runs) for w in waves),
                          "waves": len(waves), "plan": plan}, indent=1))
        return 0

    node_index = claim_node_index(args.state_dir, nodes=args.nodes)
    print(json.dumps({"schema": SCHEMA, "host": socket.gethostname(),
                      "node_index": node_index, "waves": len(waves)}), flush=True)
    for wave in waves:
        if wave_already_done(args.state_dir, wave.index, nodes=args.nodes):
            print(f"wave {wave.index} already complete; skipping", flush=True)
            continue
        names = runs_for_node(wave, node_index=node_index, nodes=args.nodes,
                              per_node=per_node)
        codes = execute_wave(wave=wave, run_names=names, launcher=args.launcher,
                             node_env=node_env, gpus=args.ngpus_per_run,
                             log_dir=args.log_dir)
        try:
            _ = mark_wave_done(args.state_dir, wave.index, codes)
        except JobRefusal as exc:
            # Do NOT write the barrier marker: the wave is not done, and a marker
            # written over a failure is how a failed arm becomes a missing arm
            # that nobody notices.
            print(str(exc), file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
