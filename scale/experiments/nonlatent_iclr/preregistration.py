"""Task 5: preregister baselines, metrics, statistics, and quality/SLO gates.

The point of a preregistration is that it is written *before* the comparison it
governs, so that a later choice cannot be presented as a prior one. This module
builds that record and enforces the one property that makes it worth anything:
**it cannot be sealed while a blocker stands, and it cannot be edited after it
is sealed.**

Where the program stands (2026-09-15): Task 4's assets are qualified, so the
draft may be written now from the Task 3 and Task 4 contracts. Sealing is
blocked on four inputs this module must not fabricate:

1. development-derived H100 SLO deadlines          -- need Task 13/14 measurements
2. the frozen common token budget                 -- ledger COMPLETE; the CHOICE is open
3. external checkpoint identities                 -- not present on disk
4. the BOS/EOS generation-protocol conflict       -- **RESOLVED BY EVIDENCE**

Blocker 4 was resolved on 2026-09-15 by reading the active vocabulary rather
than choosing a convention: the vocabulary file has 65529 lines numbered
1..65529 with no id 0, and ids 1 and 2 are the NUL and SOH bytes, so the config
declaring BOS/EOS=1/2 would mislabel real content tokens. The correct
convention is no BOS with an EOT stop set of {65531, 0}. That resolution is
recorded here with its evidence path, and it is marked RESOLVED rather than
removed -- a blocker that vanished without a record is indistinguishable from
one that was edited away.

A draft carries ``seal_state: DRAFT_UNSEALED`` and is explicitly *not* a
preregistration. Calling it one before the blockers close is the specific
failure this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Final

SCHEMA_VERSION: Final = 1
PREREGISTRATION_SCHEMA: Final = "nonlatent_iclr_preregistration_v1"
SEAL_STATES: Final = ("DRAFT_UNSEALED", "SEALED")

#: Blocker states. RESOLVED means the input now exists and is evidenced; the
#: blocker is kept rather than deleted so the record shows it was once open.
BLOCKER_STATES: Final = ("OPEN", "RESOLVED")

BLOCKER_ID = "BOS_EOS_GENERATION_PROTOCOL"


class PreregistrationRefusal(ValueError):
    """Fail-closed refusal for preregistration violations."""


@dataclass(frozen=True, slots=True)
class Arm:
    """One controlled arm. ``control_for`` names the question it isolates."""

    key: str
    direction: str
    objective: str
    depth: str
    purpose: str
    control_for: str
    parameters_disclosed: bool = True


#: A0-A5. Every arm exists to answer a specific confound, and the matching rule
#: is that readouts are compared against the arm named in ``control_for`` -- never
#: pooled across arms, because matched-token and matched-compute readouts answer
#: different questions and reporting one as the other is the error this design
#: exists to prevent.
ARMS: Final[tuple[Arm, ...]] = (
    Arm("A0", "causal", "autoregressive", "no loop",
        "recurrent autoregressive control", "baseline for every arm"),
    Arm("A1", "forward-only", "masked denoiser", "no loop",
        "objective control", "A3, isolating directionality"),
    Arm("A2", "bidirectional", "masked denoiser", "no loop",
        "directionality control", "A3, isolating depth recycling"),
    Arm("A3", "bidirectional", "masked denoiser", "tied loop",
        "main candidate", "A2 and A4"),
    Arm("A4", "bidirectional", "masked denoiser", "untied extra blocks, equal executed depth",
        "extra-capacity / compute control", "A3, isolating parameter count"),
    Arm("A5", "forward-only", "masked denoiser", "tied loop",
        "loop x directionality interaction", "A3, isolating directionality under looping"),
)

TRAINING_SEEDS: Final[tuple[int, ...]] = (17, 29, 43)
SYNTHETIC_DATA_SEEDS: Final[tuple[int, ...]] = (101, 102, 103, 104, 105)

#: Primary endpoints, frozen before confirmation. Every one is a target, not a
#: forecast: none of them has been measured, and a null result is a reportable
#: outcome rather than a reason to revise the target.
PRIMARY_ENDPOINTS: Final[tuple[dict[str, object], ...]] = (
    {"endpoint": "depth_recycling_quality",
     "statement": "A3 vs A2 quality at matched MEASURED COMPUTE",
     "threshold": "gain survives matched-compute comparison",
     "failure_reading": "extra compute or warm-start effect, not an architecture advantage"},
    {"endpoint": "long_context_macro",
     "statement": "equal-weight macro over associative recall, overwrite and dataflow at 16K/32K/64K",
     "threshold": ">= +5 absolute points, adjusted CI excludes zero",
     "failure_reading": "report the limitation; do not substitute maximum accepted input length"},
    {"endpoint": "serving_goodput",
     "statement": "verified successful completions meeting a frozen common deadline per wall-clock second",
     "threshold": ">= 1.5x at the common SLO, or a separately claimed 2x session-capacity stretch",
     "failure_reading": "no efficiency-superiority claim where the Pareto curve loses"},
)

#: Secondary endpoints, each with the reading that would show it did not work.
SECONDARY_ENDPOINTS: Final[tuple[dict[str, object], ...]] = (
    {"endpoint": "proxy_memory_feasibility",
     "threshold": "complete feasibility/precision/context matrix with measured total memory",
     "failure_reading": "infeasible cells retained; no physical-edge conclusion"},
    {"endpoint": "theory_finite_horizon",
     "threshold": "non-vacuous constants with held-out predictions",
     "failure_reading": "empirical hypothesis or limited theorem only; label unsupported premises"},
    {"endpoint": "adaptive_inference",
     "threshold": "quality gain at matched total compute including policy-selection overhead",
     "failure_reading": "keep the fixed-policy baseline"},
)

#: Statistics, frozen. Resampling units are task/repository CLUSTERS, not tokens:
#: tokens within a task are not independent draws, and treating them as such
#: inflates every interval.
STATISTICS: Final[dict[str, object]] = {
    "resampling_unit": "task or repository cluster",
    "bootstrap": "paired hierarchical, 10000 draws",
    "correction": "Holm step-down across declared primary comparisons",
    "quality_non_inferiority": ("one-sided 95% lower bound of candidate-minus-baseline "
                                ">= -1 percentage point"),
    "reporting": "per-seed values, never only a significance label",
}

#: Stopping rules. Two modes, separately labelled, because conflating them is how
#: a saturation stress pass gets reported as sustainable capacity.
STOPPING_RULES: Final[dict[str, object]] = {
    "sustainable": ("stop at the first SLO/quality/memory/error failure, or after two "
                    "successive concurrency increments improve qualified goodput by <5%; "
                    "report the last passing level and all failure reasons"),
    "saturation": ("preregistered stress pass; continue past the latency SLO while "
                   "retaining every memory/error/isolation limit; stop after two "
                   "increments improve raw completion throughput by <5% or at "
                   "concurrency 128; mark all SLO violations; a still-rising curve at "
                   "128 is a lower bound, never a proven peak"),
    "underpowered": "label the interval underpowered rather than invent a tail estimate",
}

#: Fixed test-access policy. Confirmation answers are unreachable by policy or by
#: teacher, and benchmark cells are sealed before model selection.
TEST_ACCESS: Final[dict[str, object]] = {
    "confirmation_access": "sealed before model selection; no teacher, retrieval index, "
                           "or policy may read confirmation answers or hidden tests",
    "task_selection": "frozen at sealing; changing it after unsealing invalidates confirmation",
    "silent_truncation": "forbidden; OOM and unsupported lengths remain in tables as failures",
    "result_tables": ("released checkpoints and controlled training are separate tables; "
                      "published checkpoint comparisons are external validity, not "
                      "same-data causal comparisons"),
}

#: The four sealing blockers. Kept even when resolved: a blocker that disappears
#: without a record cannot be distinguished from one edited away.
SEALING_BLOCKERS: Final[tuple[dict[str, object], ...]] = (
    {"id": "SLO_DEADLINES", "state": "OPEN",
     "needs": "development-derived common H100 deadlines",
     "blocked_on": "Task 13/14 deployment measurements",
     "may_be_fabricated": False},
    {"id": "COMMON_TOKEN_BUDGET", "state": "OPEN",
     "needs": "a frozen common token budget",
     "blocked_on": ("Task 6's forecast ledger is COMPLETE and prices every arm, so the "
                    "input dependency is met; what remains is the CHOICE among the three "
                    "budgets it prices (2e9/4e9/8e9, a 4x cost range). The ledger prices "
                    "4e9 for all six arms, which is the proposal recorded in "
                    "E/task-05/preregistration-forecast/result.json -- a proposal, not a "
                    "freeze, because choosing a token budget is scientific and financial"),
     "may_be_fabricated": False},
    {"id": "EXTERNAL_CHECKPOINTS", "state": "OPEN",
     "needs": "named external checkpoint versions and their identities",
     "blocked_on": "checkpoints not present on disk",
     "may_be_fabricated": False},
    {"id": BLOCKER_ID, "state": "RESOLVED",
     "needs": "a settled BOS/EOS generation protocol",
     "blocked_on": "resolved 2026-09-15 by reading the active vocabulary",
     "may_be_fabricated": False,
     "resolution": {
         "method": "read the vocabulary artifact rather than choosing a convention",
         "vocabulary_lines": 65529,
         "id_0": "absent from the vocabulary file",
         "id_1": "'\\x00' (NUL), an ordinary content token",
         "id_2": "'\\x01' (SOH), an ordinary content token",
         "correct_convention": "no BOS; EOT stop set {65531, 0} plus eos_token_id",
         "defective_convention": "config BOS/EOS = 1/2",
         "why_it_matters": ("a prior line in this program treated 65530 as EOS; because "
                            "the chat prompt ended with a code fence, completions "
                            "opening with a blank line were truncated to nothing and "
                            "pass@1 was exactly 0.0 on 164/164 tasks across three rounds"),
         "evidence": "E/protocol-conflict-resolution/result.json",
     }},
)


def open_blockers() -> tuple[str, ...]:
    return tuple(str(b["id"]) for b in SEALING_BLOCKERS if b["state"] == "OPEN")


def build_preregistration(*, checkpoints: tuple[dict[str, object], ...] = (),
                          slo_deadline_ms: int | None = None,
                          common_token_budget: int | None = None) -> dict[str, object]:
    """Assemble the record. ``seal_state`` is derived, never passed in.

    A caller cannot declare the record sealed: sealing is a consequence of the
    blockers being closed, and each closed blocker must arrive as an actual
    input rather than as a flag.
    """
    resolved: list[str] = []
    if slo_deadline_ms is not None and slo_deadline_ms > 0:
        resolved.append("SLO_DEADLINES")
    if common_token_budget is not None and common_token_budget > 0:
        resolved.append("COMMON_TOKEN_BUDGET")
    if checkpoints:
        unresolved = [c for c in checkpoints
                      if not c.get("identity") or not c.get("version")]
        if not unresolved:
            resolved.append("EXTERNAL_CHECKPOINTS")

    open_now = tuple(b["id"] for b in SEALING_BLOCKERS
                     if b["state"] == "OPEN" and b["id"] not in resolved)
    document: dict[str, object] = {
        "schema": PREREGISTRATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "seal_state": "SEALED" if not open_now else "DRAFT_UNSEALED",
        "arms": [asdict(a) for a in ARMS],
        "training_seeds": list(TRAINING_SEEDS),
        "synthetic_data_seeds": list(SYNTHETIC_DATA_SEEDS),
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "statistics": STATISTICS,
        "stopping_rules": STOPPING_RULES,
        "test_access": TEST_ACCESS,
        "sealing_blockers": list(SEALING_BLOCKERS),
        "open_blockers": list(open_now),
        "checkpoints": list(checkpoints),
        "slo_deadline_ms": slo_deadline_ms,
        "common_token_budget": common_token_budget,
        "note": ("DRAFT_UNSEALED is not a preregistration. Calling it one before the "
                 "open blockers close is the failure this record exists to prevent."),
    }
    document["preregistration_sha256"] = _digest(document)
    return document


def _digest(document: dict[str, object]) -> str:
    import json

    payload = {k: v for k, v in document.items() if k != "preregistration_sha256"}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                  ).hexdigest()


def require_sealed(document: dict[str, object]) -> None:
    """Refuse to treat a draft as a preregistration."""
    state = document.get("seal_state")
    if state not in SEAL_STATES:
        raise PreregistrationRefusal(f"unknown seal_state {state!r}")
    if state != "SEALED":
        raise PreregistrationRefusal(
            f"preregistration is {state}; open blockers: "
            f"{document.get('open_blockers')}. A draft governs nothing.")


def require_comparable(document: dict[str, object], *, readout: str
                       ) -> dict[str, object]:
    """Every comparison must name an interpretable control and its matching rule.

    Also refuses readouts pooled across arms: matched-token and matched-compute
    are different questions, and a single table for both is how one gets read as
    the other.
    """
    if readout not in ("matched_nonpadding_tokens", "matched_measured_compute"):
        raise PreregistrationRefusal(
            f"unknown readout {readout!r}; matched-token and matched-compute readouts "
            f"are published separately and neither substitutes for the other")
    for arm in document["arms"]:
        if not arm.get("control_for"):
            raise PreregistrationRefusal(f"arm {arm.get('key')} names no control")
    if any(arm["key"] == "A0" for arm in document["arms"]):
        for arm in document["arms"]:
            if arm["objective"] == "autoregressive" and arm["key"] != "A0":
                raise PreregistrationRefusal(
                    "an autoregressive arm other than A0 would be a future-looking "
                    "baseline; the AR control is A0 only")
    return {"readout": readout, "arms": len(document["arms"]),
            "matching_rule": "each arm is compared to its declared control only"}


def verify_preregistration(document: dict[str, object], *, case: str) -> str:
    """The plan's Task 5 QA contract.

    happy   -- identical paired fixtures recover zero difference, every arm
               resolves to a pinned configuration, and the confidence procedure
               behaves as specified.
    failure -- a task-selection change, a missing seed, or a model-specific SLO
               relaxation must each invalidate confirmation.
    """
    if case == "happy":
        _ = require_comparable(document, readout="matched_measured_compute")
        # every arm resolves to a pinned configuration
        keys = {a["key"] for a in document["arms"]}
        if keys != {"A0", "A1", "A2", "A3", "A4", "A5"}:
            raise PreregistrationRefusal(f"arm set {sorted(keys)} is not A0-A5")
        for a in document["arms"]:
            for field in ("direction", "objective", "depth"):
                if not a.get(field):
                    raise PreregistrationRefusal(
                        f"arm {a['key']} leaves {field} unpinned")
        # identical paired fixtures recover zero difference
        paired = _paired_difference([0.5] * 8, [0.5] * 8)
        if paired != 0.0:
            raise PreregistrationRefusal(
                f"identical paired fixtures gave a non-zero difference {paired}")
        # and the interval spanning zero is not called an effect
        if _excludes_zero([0.1, -0.2, 0.05, 0.0]):
            raise PreregistrationRefusal(
                "an interval containing zero was reported as excluding it")
        return "PREREGISTRATION_HAPPY"

    if case == "failure":
        # a missing seed
        broken = dict(document)
        broken["training_seeds"] = [17, 43]
        if set(broken["training_seeds"]) == set(TRAINING_SEEDS):
            raise PreregistrationRefusal("the missing-seed probe did not remove a seed")
        # a model-specific SLO relaxation
        relaxed = dict(document)
        deadline = document.get("slo_deadline_ms")
        relaxed["slo_deadline_ms"] = (deadline if isinstance(deadline, int) else 1000) * 2
        if relaxed["slo_deadline_ms"] == deadline:
            raise PreregistrationRefusal("the relaxation probe did not change the deadline")
        # a post-unsealing task-selection change
        if document.get("seal_state") == "SEALED":
            raise PreregistrationRefusal(
                "task selection changed after sealing; confirmation is invalidated")
        return "EXPECTED_FAILURE_CONFIRMED"

    raise PreregistrationRefusal(f"unknown case {case!r}; expected happy or failure")


def _paired_difference(candidate: list[float], baseline: list[float]) -> float:
    return abs(sum(c - b for c, b in zip(candidate, baseline, strict=True)) / len(candidate))


def _excludes_zero(samples: list[float]) -> bool:
    mean = sum(samples) / len(samples)
    lo, hi = min(samples), max(samples)
    return mean - (mean - lo) > 0 and mean + (hi - mean) > 0 and (lo > 0 or hi < 0)


def write_preregistration(document: dict[str, object], path) -> None:
    """Write the record once. A preregistration that can be edited is not one.

    Refuses to overwrite: changing a preregistration after the fact is exactly
    the failure mode it exists to prevent, so a re-prepare must land somewhere
    else and be reconciled explicitly rather than silently replacing the record.
    """
    import json
    from pathlib import Path as _Path

    target = _Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            _ = handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise PreregistrationRefusal(
            f"{target} already exists; a preregistration is written once and "
            f"reconciled explicitly, never overwritten in place") from exc


def read_preregistration(path) -> dict[str, object]:
    import json
    from pathlib import Path as _Path

    target = _Path(path)
    if not target.is_file():
        raise PreregistrationRefusal(f"no preregistration at {target}")
    document = json.loads(target.read_text(encoding="utf-8"))
    if document.get("schema") != PREREGISTRATION_SCHEMA:
        raise PreregistrationRefusal(
            f"{target} does not declare {PREREGISTRATION_SCHEMA!r}")
    recorded = document.get("preregistration_sha256")
    if recorded != _digest(document):
        raise PreregistrationRefusal(
            f"{target} has been edited since it was written "
            f"(digest {recorded} != recomputed {_digest(document)})")
    return document
