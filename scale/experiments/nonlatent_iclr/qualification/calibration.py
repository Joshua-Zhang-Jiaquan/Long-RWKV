"""GPU-job entrypoint for the bounded throughput calibration.

Reuses the proven job contract: the same ``parse_probe_config`` CLI/environment surface as
``probe.py``, the same rank binding, and the same write-once terminal publication. Only the work
differs: instead of running the qualification checks it measures one arm's canvas ladder and
publishes a calibration sidecar.

Each rank measures exactly one arm, named by ``calibration_arms.RANK_ARM_TABLE``. The checkpoint
step is **declared by the launcher** through ``QUALIFICATION_CHECKPOINT_STEP`` rather than guessed
from the payload, and the parameter count is measured from the loaded model. A job that cannot
state which checkpoint it measured fails closed.

The model is built with gradient checkpointing on, mirroring the training configuration this
forecast exists to price, and the first gradient step is an untimed warm-up so one-time kernel or
lazy-binding cost is never billed to a measured canvas.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections.abc import Sequence

from .calibration_arms import (
    LARGE_CHECKPOINT_MODEL_SHA256,
    RANK_COUNT,
    ArmSpec,
    arm_for_rank,
    measure_geometry,
    measure_parameters,
    verify_arm_weights,
)
from .calibration_backward import (
    ALLOC_CONF_NOTE,
    CONTENDED_NODE_NOTE,
    OPTIMIZER_MEASUREMENT_NOTE,
    OPTIMIZER_NOT_MEASURED_SCHEDULED_OFF,
    OPTIMIZER_NOT_MEASURED_SHARDED,
    WARMUP_NOTE,
    build_adamw,
    build_gradient_step,
)
from .calibration_contracts import CalibrationSidecar
from .calibration_runtime import CANVAS_LADDER, run_ladder
from .contracts import QualificationInputError
from .controls import ProbeConfig, parse_probe_config

CHECKPOINT_STEP_ENV = "QUALIFICATION_CHECKPOINT_STEP"
#: Leaves room inside the 1200 s runtime stage for the 2.9B checkpoint load, one overshooting
#: canvas, and aggregation. Canvases that would start past it are recorded as budget-exhausted.
LADDER_BUDGET_SECONDS = 600.0


def checkpoint_step(environment: dict[str, str] | None = None) -> int:
    """Read the declared checkpoint step, refusing an absent or non-positive value."""
    source = os.environ if environment is None else environment
    raw = source.get(CHECKPOINT_STEP_ENV, "")
    try:
        step = int(raw)
    except ValueError as error:
        raise QualificationInputError("checkpoint_step_not_an_integer") from error
    if step <= 0:
        raise QualificationInputError("checkpoint_step_not_positive")
    return step


def rank_from_environment(environment: dict[str, str] | None = None) -> int:
    """Bind this process to one rank, rejecting anything outside the declared world size."""
    source = os.environ if environment is None else environment
    raw = source.get("RANK", source.get("LOCAL_RANK", ""))
    try:
        rank = int(raw)
    except ValueError as error:
        raise QualificationInputError("rank_environment_not_integer") from error
    if rank not in range(RANK_COUNT):
        raise QualificationInputError("rank_environment_out_of_range")
    return rank


def _call_kwargs(arm: ArmSpec) -> dict[str, object]:
    """The trainer's call shape: full canvas, no cache, no z-slots, arm direction, arm depth.

    The recycled-depth override belongs here, not only on the backward path. An arm that is built
    at a depth it must not execute (``A2_large_noloop`` is built at ``loop_reps=1`` so the step-4750
    checkpoint loads strictly, then measured at zero) would otherwise time a loop-ON forward while
    its record declares loop-OFF, and its forward would not be comparable to its own backward.
    """
    kwargs: dict[str, object] = {
        "force_forward": arm.force_forward,
        "z_slots": None,
        "state_cache": None,
        "use_cache": False,
    }
    if arm.measured_reps() != arm.loop_reps:
        kwargs["loop_reps_override"] = arm.measured_reps()
    return kwargs


def measure(config: ProbeConfig, rank: int) -> CalibrationSidecar:
    """Load this rank's arm and measure its canvas ladder."""
    import torch  # imported here so the module stays importable without a GPU

    from ..trainer_semantics import load_functions
    from .model_checks import construct_model, load_checkpoint

    arm = arm_for_rank(rank)
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    weights = verify_arm_weights(arm)
    model = construct_model(
        config,
        torch,
        device,
        model_root=arm.model_root,
        loop_range=arm.loop_range,
        loop_reps=arm.loop_reps,
        gradient_checkpointing=True,
    )
    if arm.checkpoint_dir is None:
        declared_step: int | None = None
        checkpoint_sha: str | None = None
    else:
        tensors = load_checkpoint(model, config, torch, checkpoint_dir=arm.checkpoint_dir)
        if tensors <= 0:
            raise QualificationInputError("checkpoint_declared_no_tensors")
        declared_step = checkpoint_step()
        if declared_step != arm.checkpoint_step:
            raise QualificationInputError("declared_checkpoint_step_disagrees_with_arm")
        checkpoint_sha = LARGE_CHECKPOINT_MODEL_SHA256
    geometry = measure_geometry(model)
    parameters = measure_parameters(model)

    def forward_once(canvas_tokens: int) -> None:
        """One full-canvas masked denoise call in the trainer's call shape."""
        ids = torch.randint(1, 65_535, (1, canvas_tokens), device=device)
        span = max(1, canvas_tokens // 8)
        ids[:, 8 : 8 + span] = 65_535
        _ = model(ids, **_call_kwargs(arm))

    optimizer = None
    optimizer_reason: str | None
    optimizer_notes: list[str] = []
    if arm.optimizer_step_measured:
        optimizer = build_adamw(model, torch)
        optimizer_reason = None
        optimizer_notes.append(OPTIMIZER_MEASUREMENT_NOTE)
    elif arm.checkpoint_dir is not None:
        optimizer_reason = OPTIMIZER_NOT_MEASURED_SHARDED
    else:
        optimizer_reason = OPTIMIZER_NOT_MEASURED_SCHEDULED_OFF

    observed_selection = {"tokens": 0}

    def record_selection(selected: int) -> None:
        observed_selection["tokens"] = selected

    backwards, optimizer_steps, reason = build_gradient_step(
        model,
        load_functions(),
        torch,
        device,
        loop_reps_override=arm.measured_reps() if arm.measured_reps() != arm.loop_reps else None,
        force_forward=arm.force_forward,
        optimizer=optimizer,
        optimizer_reason=optimizer_reason or None,
        synchronize=_device_synchronize(torch, device),
        observer=record_selection,
    )
    # Untimed warm-up of the whole gradient path at the smallest rung, so the backward's
    # cold-start cost never lands inside a timed value.
    _ = backwards(min(CANVAS_LADDER))
    deadline = time.perf_counter() + LADDER_BUDGET_SECONDS
    measurements, warnings, unavailable = run_ladder(
        forward_once,
        torch.cuda,
        device=device,
        backward_step=backwards,
        optimizer_step=optimizer_steps,
        optimizer_not_measured_reason=reason,
        selected_tokens=lambda: observed_selection["tokens"],
        deadline=deadline,
        warmup=True,
    )
    if not measurements:
        # Name why, so a failed rank is diagnosable from its sidecar alone. The deployment's
        # GetJobLog returns InternalError, so the sidecar is the only channel a reviewer has.
        raise QualificationInputError(f"no_canvas_could_be_measured:{'; '.join(warnings) or 'no warning recorded'}")
    return CalibrationSidecar(
        rank=rank,
        nonce=config.nonce,
        status="PASSED",
        detail=f"{arm.arm_id}: measured {len(measurements)} of {len(measurements) + len(unavailable)} canvases",
        arm_id=arm.arm_id,
        arm_digest=arm.digest(),
        nominal_label=arm.nominal_label,
        measured=arm.measured,
        derived_from=arm.derived_from,
        loop_reps_configured=arm.loop_reps,
        loop_reps_measured=arm.measured_reps(),
        force_forward=arm.force_forward,
        gradient_checkpointing=True,
        model_geometry=geometry,
        parameters=parameters,
        checkpoint_step=declared_step,
        checkpoint_sha256=checkpoint_sha,
        weights_sha256=weights,
        measurements=measurements,
        warnings=tuple(warnings) + tuple(optimizer_notes) + (WARMUP_NOTE, ALLOC_CONF_NOTE, CONTENDED_NODE_NOTE),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one rank and publish exactly one write-once calibration record."""
    effective = sys.argv[1:] if argv is None else argv
    if "--help" in effective:
        usage = (
            "usage: calibration.py --output-dir PATH --nonce TOKEN --run-id ID "
            "--controller-receipt PATH --manifest PATH --manifest-sha256 DIGEST"
        )
        print(f"{usage} (also requires RANK in the environment; {CHECKPOINT_STEP_ENV} for checkpoint arms)")
        return 0
    parsed = parse_probe_config(effective)
    if parsed.status != "READY" or parsed.config is None:
        print(json.dumps({"detail": parsed.detail, "status": parsed.status}, sort_keys=True))
        return 2
    config = parsed.config
    rank: int | None = None
    try:
        rank = rank_from_environment()
        arm = arm_for_rank(rank)
        sidecar = measure(config, rank)
    except Exception as error:  # noqa: BROAD_EXCEPT_OK - a rank must publish a terminal record
        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr, end="")
        _capture_failure(config.output_dir, rank, traceback_text)
        if rank is None:
            return 1
        arm = arm_for_rank(rank)
        try:
            step = checkpoint_step()
        except QualificationInputError:
            step = None
        sidecar = CalibrationSidecar(
            rank=rank,
            nonce=config.nonce,
            status="FAILED",
            detail=f"{type(error).__name__}: {error}"[:500],
            arm_id=arm.arm_id,
            arm_digest=arm.digest(),
            nominal_label=arm.nominal_label,
            measured=arm.measured,
            derived_from=arm.derived_from,
            loop_reps_configured=arm.loop_reps,
            loop_reps_measured=arm.measured_reps(),
            force_forward=arm.force_forward,
            gradient_checkpointing=True,
            checkpoint_step=step if arm.checkpoint_dir is not None else None,
            checkpoint_sha256=LARGE_CHECKPOINT_MODEL_SHA256 if arm.checkpoint_dir is not None else None,
        )
    try:
        _ = sidecar.write_once(config.output_dir)
    except Exception as error:  # noqa: BROAD_EXCEPT_OK - publication failure is job-level
        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr, end="")
        _capture_failure(config.output_dir, rank, traceback_text)
        return 1
    print(sidecar.model_dump_json())
    # One arm failing to measure is a recorded outcome, not a job-level failure. Exiting non-zero
    # here would make torchrun tear the whole group down (``--max-restarts=0``) and kill the ranks
    # that were still loading, which is what happened on the first run of this payload.
    return 0


def _device_synchronize(torch_module, device):
    """Bind the device into the synchronize callable.

    ``torch.cuda.synchronize()`` with no argument drains the process's *current* device, so relying
    on that would make every timed backward depend on something this code never sets. Binding it
    removes the dependency rather than assuming a launcher sets it.
    """
    from functools import partial

    return partial(torch_module.cuda.synchronize, device)


def _capture_failure(output_dir, rank: int | None, traceback_text: str) -> None:
    """Write the rank's traceback next to its records.

    ``GetJobLog`` returns business ``InternalError`` on this deployment, so stdout/stderr never
    reach a reviewer. A failed rank is otherwise only a status with no cause.
    """
    suffix = "unknown" if rank is None else str(rank)
    target = output_dir / f"calibration-rank-{suffix}.stderr.txt"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        _ = target.write_text(traceback_text, encoding="utf-8")
    except OSError:
        print(f"could not record the rank traceback at {target}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
