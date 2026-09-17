"""Task 7: the controlled small-model arm matrix as an emitted job set.

The plan registers six arms (A0-A5) at three training seeds.  Two of those six
cannot be trained, and the owner has ruled on both:

* **A0 (causal RWKV, autoregressive)** is replaced by the on-disk
  ``RWKV7-Goose-2.9B`` AR checkpoint, labelled external-validity.  The plan's own
  wording already draws that line -- "published checkpoint comparisons ... are
  external-validity results, not causal same-data comparisons" -- and it removes
  the need to edit ``train_birwkv_diffusion.py``, whose sha256 is pinned in
  ``architecture_contract.json``.
* **A4 (untied extra blocks)** has no implementation anywhere in the code, so it
  cannot be emitted.  The ledger says so itself: its parameter count "does not
  exist in the code".  The capacity control is therefore MISSING from this
  matrix, and :func:`matrix_report` states that rather than letting a reader
  assume four arms cover six.

What remains is A1, A2, A3, A5 x seeds 17/29/43 = 12 runs.  Every one is a single
invocation of the same trainer with different flags, which is why no new training
code is needed -- only a driver that enumerates the runs, freezes the seed and
the budget, and can say in advance whether a run fits its wall-clock cap.

The budget is FROZEN at 2B tokens by the owner's decision, before any comparative
result was observed, as the plan requires.  It is a module constant so that a
caller cannot quietly choose a different one, and :func:`require_matrix` refuses
a plan that disagrees with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

#: The common token budget, frozen before any comparative result was observed.
FROZEN_TOKEN_BUDGET: Final = 2_000_000_000

#: Training seeds, from the plan's controlled-arms table.
TRAINING_SEEDS: Final = (17, 29, 43)

#: The packaged row length.  The launcher defaults to it and no arm overrides it.
PACK_LEN: Final = 4096

#: Microbatch and accumulation chosen so the global batch is 1,048,576 tokens/step
#: at `GPUS_PER_RUN` cards: 8 x 4 x 4096 x 8 = 1,048,576.  That is EXACTLY the
#: batch the 2.9B continuation trains at (4 x 4 x 16 x 4096), so the small arms
#: and the large run share an optimization shape rather than merely a token
#: budget.  It also makes steps-per-budget a clean function of the card count.
DEFAULT_MICROBATCH: Final = 8
DEFAULT_GRAD_ACCUM: Final = 4

#: The small model the arms warm-start from (the ledger's 591,054,849-parameter
#: wrapped geometry; the raw HF file holds 450,767,872).
SMALL_MODEL_DIR: Final = "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B"

#: Layers the small arms recycle, from ``calibration_arms.SMALL_LOOP_RANGE``.
SMALL_LOOP_RANGE: Final = (12, 24)

#: Per-GPU training throughput at the 4096 canvas, from the task-6 ledger.  Used
#: ONLY to forecast wall clock and refuse a plan that cannot fit its cap; it is a
#: measured number about a different (591M) geometry's step time, so every figure
#: derived from it is labelled a forecast.
LEDGER_GPU_HOURS_AT_4B: Final = {
    "A1": 1390.40,
    "A2": 1253.14,
    "A3": 1159.32,
    "A5": 1357.95,
}

#: Wall-clock ceiling observed in practice: the 2.9B continuation was stopped at
#: exactly 36 h by its own ``max_running_time_ms``.  A wall-cap stop does NOT
#: auto-resume -- it reads as a normal completion -- so a run that cannot finish
#: inside one segment costs a manual resubmission cycle per segment.
WALL_CAP_HOURS: Final = 36.0


class MatrixRefusal(ValueError):
    """The matrix plan does not describe a runnable, registered study."""


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One controlled arm, as the flags that select it."""

    arm_id: str
    #: Shell TOKENS, not ``--flag=value`` strings.  The launcher's preflight
    #: AST-parses the trainer's argparse and rejects any token that is not
    #: EXACTLY a defined flag name, so ``--seed=17`` fails while ``--seed 17``
    #: passes.  Values are separate tokens for that reason.
    extra_args: tuple[str, ...]
    force_forward: bool
    loop_reps: int
    rationale: str

    @property
    def flag_string(self) -> str:
        return " ".join(self.extra_args)


#: The four trainable arms.  A0 and A4 are absent with their reasons recorded in
#: the module docstring and in :data:`ABSENT_ARMS`.
ARMS: Final[dict[str, ArmSpec]] = {
    "A1": ArmSpec(
        arm_id="A1",
        extra_args=("--force-forward", "--gradient-checkpointing"),
        force_forward=True, loop_reps=0,
        rationale="objective control: forward-only recurrent masked denoiser, no loop",
    ),
    "A2": ArmSpec(
        arm_id="A2",
        extra_args=("--gradient-checkpointing",),
        force_forward=False, loop_reps=0,
        rationale="directionality control: bidirectional masked denoiser, no loop",
    ),
    "A3": ArmSpec(
        arm_id="A3",
        extra_args=("--gradient-checkpointing", "--loop-range",
                    f"{SMALL_LOOP_RANGE[0]}:{SMALL_LOOP_RANGE[1]}", "--loop-reps", "1"),
        force_forward=False, loop_reps=1,
        rationale="the candidate: bidirectional masked denoiser with the tied loop",
    ),
    "A5": ArmSpec(
        arm_id="A5",
        extra_args=("--force-forward", "--gradient-checkpointing", "--loop-range",
                    f"{SMALL_LOOP_RANGE[0]}:{SMALL_LOOP_RANGE[1]}", "--loop-reps", "1"),
        force_forward=True, loop_reps=1,
        rationale="loop x directionality interaction: forward-only tied loop",
    ),
}

#: Registered arms this matrix does NOT contain, and why.  Kept beside the arms
#: so a reader sees the gap in the same place they see what is planned.
ABSENT_ARMS: Final[dict[str, str]] = {
    "A0": ("causal RWKV autoregressive — no causal-objective switch exists in the "
           "trainer, so it cannot be trained; the owner ruled it replaced by the "
           "on-disk RWKV7-Goose-2.9B AR checkpoint, labelled external-validity"),
    "A4": ("untied extra blocks at equal executed depth — the capacity control is "
           "not implemented anywhere in the code, so no A4 checkpoint can exist; "
           "the ledger prices it as a declared derivation for the same reason"),
}


def tokens_per_step(*, microbatch: int = DEFAULT_MICROBATCH,
                    grad_accum: int = DEFAULT_GRAD_ACCUM,
                    ngpus: int, max_length: int = PACK_LEN) -> int:
    """Clean tokens consumed by one optimizer step across all ranks."""
    for name, value in (("microbatch", microbatch), ("grad_accum", grad_accum),
                        ("ngpus", ngpus), ("max_length", max_length)):
        if value <= 0:
            raise MatrixRefusal(f"{name} must be positive, got {value}")
    return microbatch * grad_accum * ngpus * max_length


def steps_for_budget(budget: int, *, ngpus: int,
                     microbatch: int = DEFAULT_MICROBATCH,
                     grad_accum: int = DEFAULT_GRAD_ACCUM,
                     max_length: int = PACK_LEN) -> int:
    """Steps to reach AT LEAST ``budget`` tokens.  Ceil, so no arm is short.

    Rounding down would leave every run slightly under the frozen budget, which
    is the one thing the exposure-matched comparison cannot absorb: the arms
    would differ from the budget and from each other by a step's worth of tokens
    chosen by arithmetic rather than by design.
    """
    if budget <= 0:
        raise MatrixRefusal(f"budget must be positive, got {budget}")
    per_step = tokens_per_step(microbatch=microbatch, grad_accum=grad_accum,
                               ngpus=ngpus, max_length=max_length)
    return math.ceil(budget / per_step)


def forecast_gpu_hours(arm_id: str, *, budget: int = FROZEN_TOKEN_BUDGET) -> float:
    """Forecast GPU-hours for one arm-seed at ``budget`` tokens.

    Scales the ledger's measured 4B figure linearly.  It is a FORECAST: the
    ledger's number is a measurement of a different geometry's step time, and the
    ledger itself applies no multi-GPU scaling factor.
    """
    if arm_id not in LEDGER_GPU_HOURS_AT_4B:
        msg = (f"no ledger throughput for arm {arm_id!r}; known trainable arms: "
               f"{sorted(LEDGER_GPU_HOURS_AT_4B)}")
        raise MatrixRefusal(msg)
    if budget <= 0:
        raise MatrixRefusal(f"budget must be positive, got {budget}")
    return LEDGER_GPU_HOURS_AT_4B[arm_id] * (budget / 4_000_000_000)


def forecast_wall_hours(arm_id: str, *, ngpus: int,
                        budget: int = FROZEN_TOKEN_BUDGET) -> float:
    """Forecast wall-clock hours for one run on ``ngpus`` cards."""
    gpu_hours = forecast_gpu_hours(arm_id, budget=budget)
    if ngpus <= 0:
        raise MatrixRefusal(f"ngpus must be positive, got {ngpus}")
    return gpu_hours / ngpus


def gpus_for_one_segment(arm_id: str, *, budget: int = FROZEN_TOKEN_BUDGET,
                         cap_hours: float = WALL_CAP_HOURS,
                         candidates: tuple[int, ...] = (8, 16, 24, 32)) -> int:
    """The smallest card count whose forecast fits one wall-cap segment.

    Exists because a wall-cap stop does not auto-resume: a run that overruns
    needs a human to notice and resubmit, so the number of segments is an
    operational cost, not just a scheduling detail.  Returns 0 when no candidate
    count fits, which the caller must treat as "this run will need segments".
    """
    for ngpus in sorted(candidates):
        if forecast_wall_hours(arm_id, ngpus=ngpus, budget=budget) <= cap_hours:
            return ngpus
    return 0


@dataclass(frozen=True, slots=True)
class ArmRun:
    """One arm-seed run, fully specified."""

    arm_id: str
    seed: int
    steps: int
    ngpus: int
    run_name: str
    extra_args: tuple[str, ...] = field(default=())

    @property
    def tokens(self) -> int:
        return self.steps * tokens_per_step(ngpus=self.ngpus)

    @property
    def argv_tail(self) -> str:
        """The arm's flags plus its seed, as space-separated shell tokens."""
        return " ".join((*self.extra_args, "--seed", str(self.seed)))


def run_name_for(arm_id: str, seed: int) -> str:
    """A run name that names the arm and the seed.

    Both, because 12 runs land in one save root and a name that omitted the seed
    would make two runs overwrite each other -- the failure mode the plan's
    "preserve failed attempts, never overwrite" rule exists to prevent.
    """
    return f"task7-{arm_id.lower()}-s{seed}"


# A property of the seed that is worth stating because it is easy to get wrong
# and invisible when it is:
#
# ``train_birwkv_diffusion.py:1070`` builds the data sampler with
# ``seed=args.seed``, so the trainer's ``--seed`` is BOTH the initialisation seed
# and the sampler seed.  The permutation a run reads is a function of that seed
# and the epoch only -- it does not depend on the arm.
#
# So this matrix's packing gives it two useful properties at once, and both are
# deliberate:
#
# * WITHIN a wave every arm shares one seed, so all four arms read the SAME rows.
#   The architecture is the only thing that differs, which is what a controlled
#   comparison requires.
# * ACROSS waves the seed changes (17 -> 29 -> 43), so the data differs too and
#   the three seeds are genuine data-level replication rather than three initials
#   on one fixed slice.
#
# The wave packing is therefore not an arbitrary grouping: it is one seed per
# wave precisely so those two properties hold.  If a future edit interleaved
# seeds within a wave, the arms would stop being compared on identical data.


def build_matrix(*, ngpus: int, budget: int = FROZEN_TOKEN_BUDGET,
                 seeds: tuple[int, ...] = TRAINING_SEEDS,
                 arms: tuple[str, ...] = tuple(sorted(ARMS))) -> list[ArmRun]:
    """The arm-seed runs, with the seed passed as a real trainer flag."""
    if budget != FROZEN_TOKEN_BUDGET:
        msg = (f"the common budget is frozen at {FROZEN_TOKEN_BUDGET:,} tokens; "
               f"asked for {budget:,}. Changing it after results are observed is "
               f"the thing the freeze exists to prevent")
        raise MatrixRefusal(msg)
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        msg = (f"arm(s) {unknown} are not trainable; this matrix emits "
               f"{sorted(ARMS)}. A0 and A4 are absent because no implementation "
               f"exists -- see ABSENT_ARMS, and do not add them by relabelling "
               f"another arm")
        raise MatrixRefusal(msg)
    steps = steps_for_budget(budget, ngpus=ngpus)
    runs: list[ArmRun] = []
    for arm_id in arms:
        spec = ARMS[arm_id]
        for seed in seeds:
            runs.append(ArmRun(
                arm_id=arm_id, seed=seed, steps=steps, ngpus=ngpus,
                run_name=run_name_for(arm_id, seed), extra_args=spec.extra_args))
    return runs


def require_matrix(runs: list[ArmRun], *, seeds: tuple[int, ...] = TRAINING_SEEDS,
                   arms: tuple[str, ...] = tuple(sorted(ARMS))) -> None:
    """Refuse a run set that is not the registered study.

    The plan's failure QA is explicit that removing a seed must fail the
    matched-study gate rather than quietly shrinking the study, so this checks
    coverage rather than trusting the caller.
    """
    expected = {(arm, seed) for arm in arms for seed in seeds}
    seen = {(run.arm_id, run.seed) for run in runs}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        msg = (f"arm-seed coverage is wrong: missing {missing}, unexpected "
               f"{extra}. The matched-study gate refuses an incomplete matrix "
               f"rather than reporting a smaller study as complete")
        raise MatrixRefusal(msg)
    for run in runs:
        if run.tokens < FROZEN_TOKEN_BUDGET:
            msg = (f"run {run.run_name} consumes {run.tokens:,} tokens, under the "
                   f"frozen budget {FROZEN_TOKEN_BUDGET:,}")
            raise MatrixRefusal(msg)
        tokens = run.argv_tail.split()
        if ["--seed", str(run.seed)] != tokens[-2:]:
            msg = (f"run {run.run_name} does not pass its seed to the trainer as "
                   f"the trailing '--seed <n>' tokens; got {tokens[-2:]}")
            raise MatrixRefusal(msg)


def matrix_report(*, ngpus: int, budget: int = FROZEN_TOKEN_BUDGET) -> dict[str, object]:
    """The honest summary: what runs, what it costs, what is missing."""
    runs = build_matrix(ngpus=ngpus, budget=budget)
    require_matrix(runs)
    per_arm = {arm: round(forecast_gpu_hours(arm, budget=budget), 1) for arm in sorted(ARMS)}
    total = sum(per_arm.values()) * len(TRAINING_SEEDS)
    return {
        "arms_included": sorted(ARMS),
        "arms_absent": dict(ABSENT_ARMS),
        "seeds": list(TRAINING_SEEDS),
        "runs": len(runs),
        "budget_tokens_per_run": budget,
        "steps_per_run": runs[0].steps,
        "tokens_per_step": tokens_per_step(ngpus=ngpus),
        "ngpus_per_run": ngpus,
        "forecast_gpu_hours_per_arm_seed": per_arm,
        "forecast_gpu_hours_total": round(total, 1),
        "forecast_wall_hours_per_run": {
            arm: round(forecast_wall_hours(arm, ngpus=ngpus, budget=budget), 1)
            for arm in sorted(ARMS)
        },
        "segments_per_run_at_wall_cap": {
            arm: math.ceil(forecast_wall_hours(arm, ngpus=ngpus, budget=budget)
                           / WALL_CAP_HOURS)
            for arm in sorted(ARMS)
        },
        "note": ("gpu-hour figures are FORECASTS scaled from the task-6 ledger's "
                 "measured per-arm throughput at 4096 tokens; the ledger applies "
                 "no multi-GPU scaling factor and measures a different geometry's "
                 "step time, so treat them as sizing, not as a measurement"),
    }


# --------------------------------------------------------------------------
# packing the matrix into ONE job
# --------------------------------------------------------------------------
#
# The owner's decision is to run the whole matrix inside a single job on the
# reference job's shape (2 nodes x 8 = 16 H100) with 4 GPUs per run, which packs
# as 4 concurrent runs and therefore 3 waves of 4.  Two consequences are worth
# stating plainly because the split does not change them:
#
# * Total GPU-hours are fixed by the budget, not by the packing.  7,741 GPU-h on
#   16 cards is ~484 wall hours ~= 20 days however the runs are divided; a
#   narrower split makes each run slower, not the study cheaper.
# * A 4-GPU run is a DIFFERENT global batch from an 8-GPU run at the same
#   microbatch and accumulation (524,288 vs 1,048,576 tokens/step), so the arms
#   must all use the same ``ngpus_per_run`` or they stop being exposure-matched.
#   :func:`wave_plan` keeps one value for the whole matrix for that reason.

#: GPUs inside the job.  The reference job ran 2 nodes x 8 = 16; this matrix
#: takes 4 nodes x 8 = 32 so that 4 runs of 8 GPUs can go at once.  Four nodes
#: rather than two is a deliberate departure and it buys two things:
#:
#: * 8 GPUs per run with microbatch 8 and accum 4 gives 1,048,576 tokens/step --
#:   EXACTLY the global batch the 2.9B continuation trains at, so the small arms
#:   and the large run share an optimization shape rather than merely a budget;
#: * 4 concurrent runs still cover all four arms of a wave at once, so a wave is
#:   one seed across every arm and the controlled comparison is unchanged.
JOB_GPUS: Final = 32

#: Cards per run.  Fixed for the whole matrix: mixing this across arms would
#: change the global batch and the arms would stop being exposure-matched.
GPUS_PER_RUN: Final = 8


@dataclass(frozen=True, slots=True)
class Wave:
    """One round of concurrent runs inside the job."""

    index: int
    runs: tuple[ArmRun, ...]

    @property
    def gpus(self) -> int:
        return sum(run.ngpus for run in self.runs)

    @property
    def forecast_hours(self) -> float:
        """Wall hours for this wave: the SLOWEST run in it, not the sum.

        The runs are concurrent, so the wave ends when its slowest member does.
        Summing would overstate the wall clock by the number of concurrent runs
        and understate nothing, which is the direction that gets a job killed at
        its cap.
        """
        return max(forecast_wall_hours(run.arm_id, ngpus=run.ngpus)
                   for run in self.runs)


def wave_plan(*, ngpus_per_run: int = GPUS_PER_RUN, job_gpus: int = JOB_GPUS,
              budget: int = FROZEN_TOKEN_BUDGET,
              seeds: tuple[int, ...] = TRAINING_SEEDS,
              arms: tuple[str, ...] = tuple(sorted(ARMS))) -> list[Wave]:
    """Pack the matrix into waves of concurrent runs.

    Runs are interleaved across arms rather than grouped by arm, so every wave
    contains a spread of throughputs and the wave boundary is not systematically
    early for the slow arms.  Grouping would put A1 and A2 in one wave and A3 and
    A5 in another, and the second wave's wall clock would then be set by
    whichever arm happened to land there.
    """
    if ngpus_per_run <= 0:
        msg = f"ngpus_per_run must be positive, got {ngpus_per_run}"
        raise MatrixRefusal(msg)
    concurrent = job_gpus // ngpus_per_run
    if concurrent < 1:
        msg = (f"{ngpus_per_run} GPUs per run does not fit in a {job_gpus}-GPU "
               f"job")
        raise MatrixRefusal(msg)
    runs = build_matrix(ngpus=ngpus_per_run, budget=budget, seeds=seeds, arms=arms)
    require_matrix(runs, seeds=seeds, arms=arms)
    # Round-robin over arms so seeds of the same arm are adjacent, which keeps
    # the interleaving deterministic and easy to check.
    grouped: list[ArmRun] = []
    per_arm = {arm: [r for r in runs if r.arm_id == arm] for arm in arms}
    for index in range(len(seeds)):
        for arm in arms:
            grouped.append(per_arm[arm][index])
    waves = [Wave(index=i // concurrent,
                  runs=tuple(grouped[i:i + concurrent]))
             for i in range(0, len(grouped), concurrent)]
    for wave in waves:
        if wave.gpus > job_gpus:
            msg = (f"wave {wave.index} claims {wave.gpus} GPUs of {job_gpus}")
            raise MatrixRefusal(msg)
    return waves


def require_wave_coverage(waves: list[Wave], *,
                          seeds: tuple[int, ...] = TRAINING_SEEDS,
                          arms: tuple[str, ...] = tuple(sorted(ARMS))) -> None:
    """Every arm-seed pair appears in exactly one wave, exactly once.

    The packing is the easiest place to lose a run: an off-by-one in a slice
    drops one silently and the study looks complete.
    """
    seen: list[tuple[str, int]] = []
    for wave in waves:
        seen.extend((run.arm_id, run.seed) for run in wave.runs)
    expected = {(arm, seed) for arm in arms for seed in seeds}
    if len(seen) != len(expected) or set(seen) != expected:
        msg = (f"wave packing covers {len(seen)} runs ({len(set(seen))} distinct) "
               f"but the study is {len(expected)} arm-seed pairs")
        raise MatrixRefusal(msg)


def job_report(*, ngpus_per_run: int = GPUS_PER_RUN, job_gpus: int = JOB_GPUS,
               budget: int = FROZEN_TOKEN_BUDGET) -> dict[str, object]:
    """What one job would contain, and how many segments it needs."""
    waves = wave_plan(ngpus_per_run=ngpus_per_run, job_gpus=job_gpus, budget=budget)
    require_wave_coverage(waves)
    total_gpu_hours = sum(forecast_gpu_hours(arm, budget=budget) for arm in ARMS) * len(TRAINING_SEEDS)
    wall = sum(wave.forecast_hours for wave in waves)
    return {
        "job_gpus": job_gpus,
        "ngpus_per_run": ngpus_per_run,
        "concurrent_runs": job_gpus // ngpus_per_run,
        "waves": len(waves),
        "runs": sum(len(wave.runs) for wave in waves),
        "steps_per_run": waves[0].runs[0].steps,
        "tokens_per_step": tokens_per_step(ngpus=ngpus_per_run),
        "forecast_gpu_hours_total": round(total_gpu_hours, 1),
        "forecast_wall_hours_total": round(wall, 1),
        "forecast_wall_hours_by_wave": [round(w.forecast_hours, 1) for w in waves],
        "segments_at_wall_cap": math.ceil(wall / WALL_CAP_HOURS),
        "per_wave_arms": [[f"{r.arm_id}/s{r.seed}" for r in w.runs] for w in waves],
        "note": ("one job, run sequentially in waves; a wall-cap stop does not "
                 "auto-resume, so the segment count is a manual-resubmission "
                 "count. Total GPU-hours are set by the budget and do not change "
                 "with the split; a narrower split only makes each run slower."),
    }
