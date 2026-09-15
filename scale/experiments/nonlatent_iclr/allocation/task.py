"""Task-6 prepare/analyze/verify: the forecast ledger and the allocation gate it feeds.

The gate's job is to refuse work that the evidence does not fund. It admits a payload only when an
approval record exists, the pool identity matches, the payload's priced GPU-hours fit what is left
after work already consumed, and the payload does not relabel a derived arm as a measured one.
Each refusal names which condition failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from ..qualification.calibration_contracts import CalibrationAggregate
from .ledger import (
    CANDIDATE_TOKEN_BUDGETS,
    CONTROLLED_ARMS,
    DERIVATIONS,
    TRAINING_SEEDS,
    LedgerDocument,
    build_ledger,
)

DEFAULT_BUDGET: Final = 4_000_000_000
LEDGER_NAME: Final = "allocation_ledger.json"
LEDGER_MARKDOWN_NAME: Final = "allocation_ledger.md"


class Pool(BaseModel):
    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True, strict=True)

    project_id: str = Field(min_length=1)
    logic_compute_group_id: str = Field(min_length=1)
    gpu_type: str = Field(min_length=1)


class AllocationRecord(BaseModel):
    """An approval to spend, including what has already been spent under it."""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int = 1
    authorization_id: str = Field(min_length=1)
    authorized_gpu_hours: float | None = None
    consumed_gpu_hours: float = Field(ge=0)
    pool: Pool
    receipt_reference: str = Field(min_length=1)
    supersedes: tuple[str, ...] = ()

    def remaining_gpu_hours(self) -> float | None:
        """None means no user-imposed ceiling; spending is then gated by receipts and quota."""
        if self.authorized_gpu_hours is None:
            return None
        return max(0.0, self.authorized_gpu_hours - self.consumed_gpu_hours)


class Payload(BaseModel):
    """One planned run, priced from a ledger row."""

    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    arm: str = Field(min_length=1)
    seeds: int = Field(gt=0)
    claimed_source_kind: str = Field(pattern="^(measured|derived)$")
    gpu_hours: float = Field(ge=0)


class Decision(BaseModel):
    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    admitted: bool
    reason: str


def registered_payloads(ledger: LedgerDocument, *, budget: int) -> tuple[Payload, ...]:
    """The plan's minimum matrix, priced per arm at one token budget."""
    rows = {row.arm: row for row in ledger.arms}
    payloads: list[Payload] = []
    for arm in CONTROLLED_ARMS:
        row = rows[arm]
        hours = (row.gpu_hours_per_seed or 0.0) * len(TRAINING_SEEDS)
        payloads.append(Payload(
            name=f"{arm}_x{len(TRAINING_SEEDS)}seeds_{budget}",
            arm=arm,
            seeds=len(TRAINING_SEEDS),
            claimed_source_kind="derived" if arm in DERIVATIONS else "measured",
            gpu_hours=hours,
        ))
    return tuple(payloads)


def admit(
    payload: Payload,
    *,
    allocation: AllocationRecord | None,
    pool: Pool,
    ledger: LedgerDocument,
) -> Decision:
    """Decide one payload, naming the first condition that fails."""
    if allocation is None:
        return Decision(name=payload.name, admitted=False, reason="no approval record exists")
    if allocation.pool != pool:
        return Decision(name=payload.name, admitted=False, reason="allocation names a different pool")
    row = next((candidate for candidate in ledger.arms if candidate.arm == payload.arm), None)
    if row is None:
        return Decision(name=payload.name, admitted=False, reason=f"arm {payload.arm} is not in the ledger")
    if not row.forecastable:
        return Decision(
            name=payload.name, admitted=False,
            reason=f"arm {payload.arm} is unforecastable: {row.reason_unforecastable}",
        )
    expected_kind = "derived" if payload.arm in DERIVATIONS else "measured"
    if payload.claimed_source_kind != expected_kind:
        return Decision(
            name=payload.name, admitted=False,
            reason=(
                f"arm {payload.arm} is {expected_kind} in the ledger but the payload claims "
                f"{payload.claimed_source_kind}"
            ),
        )
    is_derived = payload.arm in DERIVATIONS
    if allocation.authorized_gpu_hours is None and "no_user_imposed_ceiling" not in allocation.receipt_reference:
        return Decision(
            name=payload.name, admitted=False,
            reason=(
                "an allocation with no declared ceiling must name that policy in its receipt; this one "
                "presents quota already consumed as if it were new funding"
            ),
        )
    priced = (row.gpu_hours_per_seed or 0.0) * payload.seeds
    if priced > 0 and abs(payload.gpu_hours - priced) > priced * 0.01:
        return Decision(
            name=payload.name, admitted=False,
            reason=(
                f"declared {payload.gpu_hours:.1f} GPU-hours for {payload.seeds} seed(s) of arm "
                f"{payload.arm}, but the bound ledger prices that at {priced:.1f}; a payload does not "
                "set its own price"
            ),
        )
    remaining = allocation.remaining_gpu_hours()
    if remaining is not None and payload.gpu_hours > remaining:
        return Decision(
            name=payload.name, admitted=False,
            reason=f"priced {payload.gpu_hours:.1f} GPU-hours exceeds the {remaining:.1f} remaining",
        )
    detail = f"priced {payload.gpu_hours:.1f} GPU-hours"
    if is_derived:
        detail += f"; derived from {row.source_arm}, not measured"
    return Decision(name=payload.name, admitted=True, reason=detail)


def load_aggregate(path: Path) -> CalibrationAggregate:
    return CalibrationAggregate.model_validate_json(path.read_text(encoding="utf-8"))


def _archive_then_write(path: Path, payload: str) -> None:
    """Publish the ledger, keeping any previous one instead of overwriting it.

    A regenerated ledger can legitimately differ (a later calibration covers more arms), so the
    previous bytes are archived under a content-addressed name rather than replaced. An identical
    rewrite is a no-op. This mirrors the registry's archive-then-write convention.
    """
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing == payload:
            return
        suffix = hashlib.sha256(existing.encode("utf-8")).hexdigest()[:16]
        archived = path.with_name(f"{path.stem}.archive-{suffix}{path.suffix}")
        if not archived.is_file():
            _ = archived.write_text(existing, encoding="utf-8")
    _ = path.write_text(payload, encoding="utf-8")


def render_markdown(ledger: LedgerDocument) -> str:
    lines = [
        "# Task-6 allocation ledger",
        "",
        f"Calibration manifest: `{ledger.calibration_manifest_sha256}` (status {ledger.calibration_status}).",
        "",
        "| arm | source | basis | step s | tokens/s | GPU-h per seed @4B | GPU-h for 3 seeds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ledger.arms:
        if row.forecastable:
            lines.append(
                f"| {row.arm} | {row.source_kind} ({row.source_arm}) | canvas {row.measured_canvas} | "
                f"{row.step_seconds:.3f} | {row.tokens_per_second:.1f} | {row.gpu_hours_per_seed:.2f} | "
                f"{row.gpu_hours_all_seeds:.2f} |"
            )
        else:
            lines.append(f"| {row.arm} | {row.source_kind} ({row.source_arm}) | unforecastable | - | - | - | - |")
    # Each row's own caveats belong beside its number. Carrying them only in the JSON left the
    # human-readable table asserting more than the priced rows do -- A4 showed A3's cost with no
    # warning that its untied blocks make that an understatement.
    lines += ["", "## Per-row caveats", ""]
    for row in ledger.arms:
        if not row.forecastable:
            continue
        lines.append(f"### {row.arm}")
        lines.append("")
        for note in row.assumptions:
            lines.append(f"- {note}")
        lines.append("")
    lines += ["", "## Unforecastable", ""]
    for arm in ledger.unforecastable:
        lines.append(f"- **{arm}**: {ledger.unforecast_reasons.get(arm, '')}")
    lines += ["", "## Storage", "", ledger.storage.note, ""]
    lines += ["", "## 2.9B confirmation", "", ledger.confirmation_note, ""]
    lines += ["", "## Assumptions", ""]
    lines += [f"- {item}" for item in ledger.assumptions]
    lines += ["", "## Not claimed", ""]
    lines += [f"- {item}" for item in ledger.claims_not_made]
    return "\n".join(lines) + "\n"


def prepare(aggregate_path: Path, output_root: Path, *, manifest_sha: str, budget: int) -> tuple[LedgerDocument, int]:
    aggregate = load_aggregate(aggregate_path)
    ledger = build_ledger(aggregate, calibration_manifest_sha256=manifest_sha, token_budget=budget)
    output_root.mkdir(parents=True, exist_ok=True)
    _archive_then_write(output_root / LEDGER_NAME, json.dumps(ledger.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    _archive_then_write(output_root / LEDGER_MARKDOWN_NAME, render_markdown(ledger))
    return ledger, 0 if any(row.forecastable for row in ledger.arms) else 2


def matrix_gpu_hours(ledger: LedgerDocument) -> float:
    """What the whole controlled matrix costs at this ledger's prices."""
    return sum(row.gpu_hours_all_seeds or 0.0 for row in ledger.arms)


def roomy_ceiling(ledger: LedgerDocument) -> float:
    """A ceiling that covers the priced matrix, so a probe can isolate a rule other than budget.

    Every planted-failure probe except the budget one needs a ceiling large enough that no other
    rule can fire first. A probe that is refused for the wrong reason verifies nothing.
    """
    return max(matrix_gpu_hours(ledger) * 1.1, 1.0)


def _synthetic_allocation(ledger: LedgerDocument, *, authorized: float | None, consumed: float, pool: Pool,
                          receipt: str = "no_user_imposed_ceiling") -> AllocationRecord:
    return AllocationRecord(
        authorization_id="nonlatent-gpu-campaign-20260912",
        authorized_gpu_hours=authorized,
        consumed_gpu_hours=consumed,
        pool=pool,
        receipt_reference=receipt,
    )


def happy_probe(ledger: LedgerDocument, *, budget: int, pool: Pool) -> tuple[bool, dict[str, str]]:
    """Show the gate admitting what it can price and refusing what it cannot, each with a reason."""
    ceiling = roomy_ceiling(ledger)
    allocation = _synthetic_allocation(ledger, authorized=ceiling, consumed=0.0, pool=pool)
    admitted: dict[str, str] = {}
    refused: dict[str, str] = {}
    for payload in registered_payloads(ledger, budget=budget):
        decision = admit(payload, allocation=allocation, pool=pool, ledger=ledger)
        (admitted if decision.admitted else refused)[payload.name] = decision.reason
    if not admitted:
        return False, {"detail": "no arm of the matrix could be priced from this calibration"}
    unexplained = {
        name: reason for name, reason in refused.items()
        if "unforecastable" not in reason
    }
    return not unexplained, {"admitted": json.dumps(admitted), "refused": json.dumps(refused)}



def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task-6 forecast ledger and allocation gate.")
    parser.add_argument("command", choices=("prepare", "analyze", "verify"))
    parser.add_argument("--case", choices=("happy", "failure"), default="happy")
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("DAN/nonlatent_iclr"))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args(argv)
    pool = Pool(
        project_id="project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
        logic_compute_group_id="lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
        gpu_type="NVIDIA_H100_SXM_80G",
    )
    if args.command == "verify" and args.case == "failure":
        return _failure_probe(args.aggregate, manifest_sha=args.manifest_sha256, budget=args.budget, pool=pool)
    ledger, code = prepare(args.aggregate, args.output_root, manifest_sha=args.manifest_sha256, budget=args.budget)
    payloads = registered_payloads(ledger, budget=args.budget)
    if args.command == "prepare":
        print(json.dumps({
            "status": "LEDGER_READY" if code == 0 else "BLOCKED_EXTERNAL",
            "forecastable": [row.arm for row in ledger.arms if row.forecastable],
            "unforecastable": list(ledger.unforecastable),
            "candidate_budgets": list(CANDIDATE_TOKEN_BUDGETS),
        }, sort_keys=True))
        return code
    if args.command == "analyze":
        print(json.dumps({
            "status": "LEDGER_REPRODUCED",
            "priced": {row.arm: round(row.gpu_hours_all_seeds or 0.0, 2) for row in ledger.arms if row.forecastable},
            "unforecastable": list(ledger.unforecastable),
        }, sort_keys=True))
        return code
    del payloads
    ok, detail = happy_probe(ledger, budget=args.budget, pool=pool)
    if not ok:
        print(json.dumps({"status": "ALLOCATION_REFUSED", **detail}, sort_keys=True))
        return 2
    print(json.dumps({"status": "ALLOCATION_ADMITTED", **detail}, sort_keys=True))
    return 0


def _failure_probe(aggregate_path: Path, *, manifest_sha: str, budget: int, pool: Pool) -> int:
    """Each planted violation must be refused, and for the stated reason."""
    aggregate = load_aggregate(aggregate_path)
    ledger = build_ledger(aggregate, calibration_manifest_sha256=manifest_sha, token_budget=budget)
    rows = {row.arm: row for row in ledger.arms}
    priced_arm = next((arm for arm in CONTROLLED_ARMS if rows[arm].forecastable), None)
    if priced_arm is None:
        print(json.dumps({"status": "FAILURE_PROBE_MISSED", "detail": "no forecastable arm to plant against"}, sort_keys=True))
        return 2
    hours = rows[priced_arm].gpu_hours_all_seeds or 1.0
    payload = Payload(
        name="probe", arm=priced_arm, seeds=len(TRAINING_SEEDS),
        claimed_source_kind="derived" if priced_arm in DERIVATIONS else "measured", gpu_hours=hours,
    )
    roomy = roomy_ceiling(ledger)
    derived_arm = next((arm for arm in CONTROLLED_ARMS if arm in DERIVATIONS), "A0")
    probes = {
        # Only the absence of an approval can refuse this one.
        "absent_approval_record": admit(payload, allocation=None, pool=pool, ledger=ledger),
        # Only the remaining-budget rule can refuse this one: nothing else about it is wrong.
        "exhausted_budget": admit(
            payload, pool=pool, ledger=ledger,
            allocation=_synthetic_allocation(ledger, authorized=1.0, consumed=0.5, pool=pool),
        ),
        # A ceiling that covers the matrix, so only the pool mismatch remains.
        "invalid_pool_identity": admit(
            payload, pool=pool, ledger=ledger,
            allocation=_synthetic_allocation(
                ledger, authorized=roomy, consumed=0.0,
                pool=Pool(project_id="project-other", logic_compute_group_id="lcg-other", gpu_type="NVIDIA_H100_SXM_80G"),
            ),
        ),
        # The receipt rule is checked before the budget rule, so a roomy ceiling keeps it isolated.
        "consumed_quota_as_new_funding": admit(
            payload, pool=pool, ledger=ledger,
            allocation=_synthetic_allocation(ledger, authorized=None, consumed=0.0, pool=pool, receipt="carried-over-quota"),
        ),
        # A payload that prices itself at nothing, which the gate must price from the ledger instead.
        "underpriced_payload": admit(
            payload.model_copy(update={"gpu_hours": 0.0}),
            pool=pool, ledger=ledger,
            allocation=_synthetic_allocation(ledger, authorized=roomy, consumed=0.0, pool=pool),
        ),
        # A derived arm carrying a measured sibling's price rather than its own.
        "derived_arm_borrowed_price": admit(
            payload.model_copy(update={
                "arm": derived_arm, "claimed_source_kind": "derived",
                "gpu_hours": (rows[derived_arm].gpu_hours_all_seeds or 1.0) * 0.9,
            }),
            pool=pool, ledger=ledger,
            allocation=_synthetic_allocation(ledger, authorized=roomy, consumed=0.0, pool=pool),
        ),
        # Only the label mismatch can refuse this one.
        "derived_arm_labeled_measured": admit(
            payload.model_copy(update={"arm": derived_arm, "claimed_source_kind": "measured",
                                       "gpu_hours": (rows[derived_arm].gpu_hours_all_seeds or 1.0)}),
            pool=pool, ledger=ledger,
            allocation=_synthetic_allocation(ledger, authorized=roomy, consumed=0.0, pool=pool),
        ),
    }
    missed = {name: decision.reason for name, decision in probes.items() if decision.admitted}
    # A probe that is refused for the wrong reason verifies nothing, so require the stated rule.
    expected = {
        "absent_approval_record": "no approval record",
        "exhausted_budget": "remaining",
        "invalid_pool_identity": "different pool",
        "consumed_quota_as_new_funding": "no declared ceiling",
        "derived_arm_labeled_measured": "derived",
        "underpriced_payload": "does not",
        "derived_arm_borrowed_price": "prices that at",
    }
    if missed:
        print(json.dumps({"status": "FAILURE_PROBE_MISSED", "admitted_but_should_not_be": missed}, sort_keys=True))
        return 2
    wrong_reason = {
        name: decision.reason for name, decision in probes.items()
        if expected[name] not in decision.reason
    }
    if wrong_reason:
        print(json.dumps({"status": "FAILURE_PROBE_MISSED", "refused_for_the_wrong_reason": wrong_reason}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "EXPECTED_FAILURE_CONFIRMED",
        "rejections": {name: decision.reason for name, decision in probes.items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
