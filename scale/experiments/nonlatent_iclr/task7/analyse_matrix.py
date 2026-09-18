"""Read the matrix's own run logs into the arm comparison the preregistration declares.

Why this reads logs rather than checkpoints
-------------------------------------------

The runs write a per-step line carrying the masked cross-entropy the arms are compared on.
The checkpoints carry weights, and scoring them would need the evaluation harness, the frozen
fixtures and GPU time; the logs carry the same quantity the training loop computed, on every
step, for free. The comparison this module makes is the one the trainer already made.

What it refuses to do
---------------------

**It will not declare an arm better than another on one seed.** Three seeds are registered
precisely because the seed spread is the yardstick: A1's own three seeds land 0.53 apart in
masked CE, so a between-arm difference smaller than the within-arm spread is not a finding,
and reporting it as one would be the "extra compute or warm-start" failure the plan names.
The contrast function returns the difference and the spread it must clear, and says whether
it does -- it does not return a verdict on a difference it cannot distinguish from noise.

**It will not compare arms that did not finish.** A run short of its budget has a
cross-entropy from an earlier point on its own schedule, and putting that beside a finished
run's is comparing two different amounts of training.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, Sequence

from scale.experiments.nonlatent_iclr.tasks.statistics import holm_step_down

SCHEMA: Final = "nonlatent_matrix_analysis_v1"

#: The frozen budget every arm runs to.  A run short of it is not comparable.
TARGET_STEPS: Final = 1908

#: The metric the arms are compared on, as the trainer logs it.
METRIC: Final = "mask_ce"

#: ```[step 1900] mask_ce=5.6091 ce_low=...``` -- the step marker and its key=value pairs.
STEP_LINE: Final = re.compile(r"\[step\s+(\d+)\]\s+(.*)")
PAIR: Final = re.compile(r"([a-z_]+)=([-0-9.e+]+)")

#: The comparisons the preregistration declares, as (contrast name, arm, reference arm).
#: Each answers one question the design exists to ask, and no others are tested.
DECLARED_CONTRASTS: Final = (
    ("directionality", "A2", "A1"),
    ("loop_on_bidirectional", "A3", "A2"),
    ("loop_on_forward_only", "A5", "A1"),
)


class AnalysisRefusal(ValueError):
    """The comparison cannot be made as declared."""


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """One run's final logged metrics."""

    run_name: str
    arm: str
    seed: int
    final_step: int
    metrics: dict[str, float]

    #: The run's newest CHECKPOINT step, when the caller supplied it.  Completion is a
    #: claim about the checkpoints, not about the log: the trainer logs every 20 steps, so a
    #: run that finished at 1908 has a last log line of 1900, and reading completion off the
    #: log files finished runs as short.  `None` means the caller did not supply it, and the
    #: log's own step is used with the logging interval allowed for.
    checkpoint_step: int | None = None

    @property
    def complete(self) -> bool:
        if self.checkpoint_step is not None:
            return self.checkpoint_step >= TARGET_STEPS
        return self.final_step >= TARGET_STEPS


def parse_run_name(name: str) -> tuple[str, int]:
    """``task7-a3-s29`` -> ``("A3", 29)``, refusing anything else.

    Refusing rather than guessing: a run whose name does not carry an arm and a seed cannot
    be placed in the comparison, and a default would silently file it under one.
    """
    match = re.fullmatch(r"task7-(a\d+)-s(\d+)", name)
    if match is None:
        raise AnalysisRefusal(
            f"run name {name!r} does not encode an arm and a seed as task7-<arm>-s<seed>")
    return match.group(1).upper(), int(match.group(2))


def read_run_log(path: Path) -> RunMetrics:
    """The last step line's metrics, and the step they came from."""
    path = Path(path)
    last_step: int | None = None
    last: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP_LINE.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        if last_step is not None and step < last_step:
            # the launcher appends across segments; a restart replays earlier steps, and
            # taking a later line blindly would read a resumed run's step 20 as its final
            continue
        pairs = {k: float(v) for k, v in PAIR.findall(match.group(2))}
        if METRIC not in pairs:
            continue
        last_step, last = step, pairs
    if last_step is None:
        raise AnalysisRefusal(f"{path} carries no [{METRIC}] step line")
    name = path.name.removesuffix(".run.log")
    arm, seed = parse_run_name(name)
    return RunMetrics(run_name=name, arm=arm, seed=seed, final_step=last_step,
                      metrics=last)


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """One arm's value across seeds, with the spread that any claim must clear."""

    arm: str
    values: dict[int, float]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return sum(self.values.values()) / self.n

    @property
    def spread(self) -> float:
        """Max minus min across seeds -- the yardstick, not a confidence interval.

        With three seeds a CI would be a claim the design does not support; the range is
        what the data states, and it is deliberately the LESS flattering statistic.
        """
        return max(self.values.values()) - min(self.values.values())


def summarise(runs: Sequence[RunMetrics]) -> dict[str, ArmSummary]:
    """Group runs by arm, refusing to include one that did not finish.

    The caller supplies each run's checkpoint step where it has one, because the log's last
    line is the last LOGGED step and the trainer logs every 20: a run that finished at 1908
    has a final log line of 1900, so reading completion from the log files every finished
    run as short.  A module that refuses on the wrong evidence is as wrong as one that
    accepts it.
    """
    short = [r.run_name for r in runs if not r.complete]
    if short:
        raise AnalysisRefusal(
            f"{len(short)} run(s) stopped short of step {TARGET_STEPS}: {short}. A run "
            f"short of its budget has a cross-entropy from an earlier point on its own "
            f"schedule, and comparing it with a finished run's compares two amounts of "
            f"training")
    by_arm: dict[str, dict[int, float]] = {}
    for run in runs:
        by_arm.setdefault(run.arm, {})[run.seed] = run.metrics[METRIC]
    return {arm: ArmSummary(arm=arm, values=values) for arm, values in sorted(by_arm.items())}


def contrasts(summary: dict[str, ArmSummary]) -> list[dict[str, object]]:
    """The declared pairwise comparisons, each against the seed spread it must clear.

    A contrast is reported as ``separable`` only when its magnitude exceeds the LARGER of
    the two arms' seed spreads. That is a conservative reading of three seeds and it is the
    one that cannot manufacture a result: an arm difference inside the noise its own seeds
    show is not evidence that the arm differs.
    """
    out: list[dict[str, object]] = []
    for name, arm, reference in DECLARED_CONTRASTS:
        if arm not in summary or reference not in summary:
            out.append({"contrast": name, "arm": arm, "reference": reference,
                        "status": "not_measured", "difference": None, "must_clear": None,
                        "separable": None})
            continue
        a, b = summary[arm], summary[reference]
        difference = a.mean - b.mean
        must_clear = max(a.spread, b.spread)
        out.append({
            "contrast": name, "arm": arm, "reference": reference, "status": "measured",
            "arm_mean": a.mean, "reference_mean": b.mean,
            "difference": difference, "must_clear": must_clear,
            "separable": abs(difference) > must_clear,
            "note": (f"{arm} - {reference} = {difference:+.4f}; the larger of the two arms' "
                     f"seed spreads is {must_clear:.4f}"),
        })
    return out


def analyse(log_dir: Path, *, run_names: Sequence[str] | None = None,
            checkpoint_steps: dict[str, int] | None = None) -> dict[str, object]:
    """Every run's final metrics, the arm summaries, and the declared contrasts."""
    log_dir = Path(log_dir)
    names = run_names or sorted(p.name.removesuffix(".run.log")
                                for p in log_dir.glob("*.run.log"))
    runs = []
    for name in names:
        run = read_run_log(log_dir / f"{name}.run.log")
        if checkpoint_steps and name in checkpoint_steps:
            run = replace(run, checkpoint_step=checkpoint_steps[name])
        runs.append(run)
    summary = summarise(runs)
    declared = contrasts(summary)
    measured = [c for c in declared if c["status"] == "measured"]
    if measured:
        adjusted = holm_step_down([0.5] * len(measured), family="matrix_contrasts")
        for contrast, row in zip(measured, adjusted, strict=True):
            contrast["holm_rank"] = row["rank"]
    return {
        "schema": SCHEMA,
        "metric": METRIC,
        "target_steps": TARGET_STEPS,
        "runs": [asdict(r) for r in runs],
        "arms": {arm: {"mean": s.mean, "spread": s.spread, "n": s.n, "values": s.values}
                 for arm, s in summary.items()},
        "contrasts": declared,
        "note": ("a contrast is `separable` only when it exceeds the larger of the two arms' "
                 "seed spreads. With three seeds that is a conservative reading, and it is "
                 "the one that cannot manufacture a result from noise."),
    }


def write_analysis(document: dict[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path
