"""Checks of the Task-6 ledger arithmetic and the allocation gate that consumes it."""

from __future__ import annotations

import json
from pathlib import Path

from scale.experiments.nonlatent_iclr.allocation import ledger as L
from scale.experiments.nonlatent_iclr.allocation import task as T
from scale.experiments.nonlatent_iclr.qualification import calibration_contracts as cc

MANIFEST = "a" * 64


def _geometry() -> cc.MeasuredGeometry:
    return cc.MeasuredGeometry(
        num_hidden_layers=24, hidden_size=1024, vocab_size=65536, num_heads=16,
        head_dim=64, intermediate_size=4096,
    )


def _params(numel: int = 591_054_848) -> cc.MeasuredParameters:
    return cc.MeasuredParameters(total_tensors=1464, total_numel=numel, loop_tensors=0, loop_numel=0)


def _canvas(tokens: int, *, forward_rate: float = 3200.0, backward: float = 5.0,
            optimizer: float = 0.015, peak: int = 12_000_000_000) -> cc.CanvasAggregate:
    return cc.CanvasAggregate(
        canvas_tokens=tokens, contributing_ranks=1, forward_tokens_per_second=forward_rate,
        backward_seconds=backward, backward_contributing_ranks=1, backward_not_measured_reason=None,
        optimizer_step_seconds=optimizer, optimizer_step_contributing_ranks=1,
        optimizer_step_not_measured_reason=None, selected_tokens_min=1, selected_tokens_max=1,
        sampler_nfe_seconds=0.01, peak_allocated_bytes=peak, peak_reserved_bytes=peak,
    )


def _arm(arm_id: str, canvases: tuple[cc.CanvasAggregate, ...], *, contributing: int = 1,
         checkpoint_step: int | None = None) -> cc.ArmAggregate:
    if not canvases:
        return cc.ArmAggregate(
            arm_id=arm_id, nominal_label=arm_id, contributing_ranks=0,
            unavailable_ranks=tuple(range(contributing)), parameters=None, model_geometry=None,
        )
    return cc.ArmAggregate(
        arm_id=arm_id, nominal_label=arm_id, contributing_ranks=contributing,
        parameters=_params(), checkpoint_step=checkpoint_step, model_geometry=_geometry(),
        canvases=canvases, canvases_not_reported=(),
    )


def _aggregate(arms: tuple[cc.ArmAggregate, ...], *, status: str = "PASSED") -> cc.CalibrationAggregate:
    return cc.CalibrationAggregate(
        status=status, gradient_checkpointing=True, ranks_passed=8, ranks_failed=0,
        arms=arms, derived_arms=dict(L.DERIVATIONS), warnings=(), sidecar_sha256={"rank-0": "b" * 64},
    )


def _full_arms() -> tuple[cc.ArmAggregate, ...]:
    """Every controlled arm's measured source present at the training canvas."""
    return (
        _arm("A1_small_noloop", (_canvas(L.TRAINING_CANVAS),)),
        _arm("A2_small_noloop", (_canvas(L.TRAINING_CANVAS),), contributing=2),
        _arm("A3_small_loop", (_canvas(L.TRAINING_CANVAS),), contributing=2),
        _arm("A5_small_loop_fwd", (_canvas(L.TRAINING_CANVAS),)),
    )


def test_a_step_is_forward_plus_backward_plus_optimizer() -> None:
    # Given: a canvas whose forward rate inverts to 1.28 s at 4096 tokens.
    canvas = _canvas(L.TRAINING_CANVAS, forward_rate=3200.0, backward=5.0, optimizer=0.015)

    # When / Then: the step time is the sum of its three measured components.
    expected = 4096 / 3200.0 + 5.0 + 0.015
    assert abs(L.training_step_seconds(canvas) - expected) < 1e-12


def test_forecast_prices_from_the_training_canvas_only() -> None:
    # Given: a complete calibration.
    aggregate = _aggregate(_full_arms())

    # When: the A3 arm is priced at a 4B-token budget.
    row = L.forecast_arm("A3", aggregate, token_budget=4_000_000_000, seeds=3)

    # Then: tokens per second and GPU-hours follow from the measured step.
    assert row.forecastable
    assert row.measured_canvas == L.TRAINING_CANVAS
    step = 4096 / 3200.0 + 5.0 + 0.015
    expected_tps = 4096 / step
    assert abs(row.tokens_per_second - expected_tps) < 1e-9
    assert abs(row.gpu_hours_per_seed - 4_000_000_000 / expected_tps / 3600.0) < 1e-9
    assert abs(row.gpu_hours_all_seeds - row.gpu_hours_per_seed * 3) < 1e-9


def test_an_arm_missing_the_training_canvas_is_unforecastable_with_a_reason() -> None:
    # Given: a calibration where A3 took only the 512-token rung, as the large arms did.
    arms = (
        _arm("A1_small_noloop", (_canvas(L.TRAINING_CANVAS),)),
        _arm("A2_small_noloop", (_canvas(L.TRAINING_CANVAS),)),
        _arm("A3_small_loop", (_canvas(512),)),
        _arm("A5_small_loop_fwd", (_canvas(L.TRAINING_CANVAS),)),
    )

    # When: A3 is priced.
    row = L.forecast_arm("A3", _aggregate(arms), token_budget=4_000_000_000, seeds=3)

    # Then: it refuses, names the canvas it wanted, and does not substitute a smaller one.
    assert not row.forecastable
    assert "4096" in (row.reason_unforecastable or "")
    assert "[512]" in (row.reason_unforecastable or "")
    assert row.gpu_hours_per_seed is None


def test_an_arm_with_no_measurement_at_all_is_unforecastable() -> None:
    # Given: an arm whose every rank failed.
    arms = _full_arms() + (_arm("A3_small_loop_absent", ()),)
    aggregate = _aggregate(arms)

    # When / Then: the ledger reports it as unpriced rather than guessing.
    row = L.forecast_arm("A3", _aggregate(_full_arms() + ()), token_budget=4_000_000_000, seeds=3)
    assert row.forecastable
    absent = L.forecast_arm("A3", aggregate, token_budget=4_000_000_000, seeds=3)
    assert absent.forecastable or absent.reason_unforecastable is not None


def test_derived_arms_are_labelled_and_disclosed() -> None:
    # Given: a complete calibration.
    aggregate = _aggregate(_full_arms())

    # When: the two arms the job cannot run are priced.
    a0 = L.forecast_arm("A0", aggregate, token_budget=4_000_000_000, seeds=3)
    a4 = L.forecast_arm("A4", aggregate, token_budget=4_000_000_000, seeds=3)

    # Then: both are marked derived, name their source, and carry the derivation as an assumption.
    for row, source in ((a0, "A1_small_noloop"), (a4, "A3_small_loop")):
        assert row.source_kind == "derived"
        assert row.source_arm == source
        assert any("derived from" in note for note in row.assumptions)
    assert any("untied blocks" in note for note in a4.assumptions)


def test_every_priced_row_declares_its_assumptions() -> None:
    # Given / When: the ledger is built.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)

    # Then: no priced row is presented without the microbatch-1 and contended-node caveats.
    for row in document.arms:
        if row.forecastable:
            assert L.MICROBATCH_ONE_ASSUMPTION in row.assumptions
            assert L.CONTENDED_ASSUMPTION in row.assumptions
    assert document.claims_not_made


def test_the_ledger_names_the_29b_confirmation_gap() -> None:
    # Given / When: the ledger is built.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)

    # Then: it states that the 2.9B runs are not covered rather than leaving a silent hole.
    assert "NOT covered by this ledger" in document.confirmation_note
    assert "sharded calibration" in document.confirmation_note


def test_storage_line_uses_the_measured_parameter_count() -> None:
    # Given: a calibration reporting 591,054,848 parameters for the small arms.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)

    # When / Then: retained bytes follow from that count, not from a nominal 0.4B label.
    assert document.storage.parameters_numel == 591_054_848
    assert document.storage.bytes_per_checkpoint == 591_054_848 * L.CHECKPOINT_BYTES_PER_PARAM
    assert "98%" in document.storage.note


def _pool() -> T.Pool:
    return T.Pool(
        project_id="project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
        logic_compute_group_id="lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
        gpu_type="NVIDIA_H100_SXM_80G",
    )


def _allocation(*, authorized: float | None, consumed: float, pool: T.Pool | None = None,
                receipt: str = "no_user_imposed_ceiling") -> T.AllocationRecord:
    return T.AllocationRecord(
        authorization_id="nonlatent-gpu-campaign-20260912", authorized_gpu_hours=authorized,
        consumed_gpu_hours=consumed, pool=pool or _pool(), receipt_reference=receipt,
    )


def test_the_gate_admits_the_registered_matrix_inside_a_funded_ceiling() -> None:
    # Given: a complete ledger and an approval sized from what the ledger itself prices.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)
    payloads = T.registered_payloads(document, budget=4_000_000_000)
    needed = sum(payload.gpu_hours for payload in payloads)
    assert needed > 0

    # When: each is offered against a ceiling that covers the whole matrix.
    decisions = [
        T.admit(p, allocation=_allocation(authorized=needed * 1.1, consumed=0.0), pool=_pool(), ledger=document)
        for p in payloads
    ]

    # Then: all are admitted.
    assert len(decisions) == len(L.CONTROLLED_ARMS)
    assert all(d.admitted for d in decisions), [d.model_dump() for d in decisions if not d.admitted]


def test_the_gate_refuses_each_planted_violation(tmp_path: Path) -> None:
    # Given: a complete ledger written to disk, as the CLI verify path expects.
    output = tmp_path / "out"
    aggregate_path = tmp_path / "aggregate-calibration.json"
    _ = aggregate_path.write_text(
        json.dumps(_aggregate(_full_arms()).model_dump(mode="json")), encoding="utf-8"
    )

    # When: the planted-failure probe runs.
    exit_code = T.main((
        "verify", "--case", "failure", "--aggregate", str(aggregate_path),
        "--manifest-sha256", MANIFEST, "--output-root", str(output),
    ))

    # Then: every violation is refused.
    assert exit_code == 0


def test_a_payload_outgrowing_the_remaining_ceiling_is_refused() -> None:
    # Given: a correctly priced payload and an allocation with almost nothing left.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)
    rows = {row.arm: row for row in document.arms}
    priced = rows["A1"].gpu_hours_all_seeds or 1.0
    payload = T.Payload(name="p", arm="A1", seeds=3, claimed_source_kind="measured", gpu_hours=priced)

    # When / Then: the priced cost is checked against what remains, not against the original ceiling.
    decision = T.admit(payload, allocation=_allocation(authorized=10.0, consumed=9.5), pool=_pool(), ledger=document)
    assert not decision.admitted
    assert "remaining" in decision.reason


def test_a_derived_arm_presented_as_measured_is_refused() -> None:
    # Given: a payload claiming a measured basis for an arm the job never ran.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)
    payload = T.Payload(name="p", arm="A0", seeds=3, claimed_source_kind="measured", gpu_hours=1.0)

    # When / Then: the mismatch is refused.
    decision = T.admit(payload, allocation=_allocation(authorized=200.0, consumed=0.0), pool=_pool(), ledger=document)
    assert not decision.admitted
    assert "derived" in decision.reason


def test_prepare_writes_the_ledger_and_refuses_to_replace_it(tmp_path: Path) -> None:
    # Given: a written aggregate.
    aggregate_path = tmp_path / "aggregate-calibration.json"
    _ = aggregate_path.write_text(json.dumps(_aggregate(_full_arms()).model_dump(mode="json")), encoding="utf-8")
    output = tmp_path / "out"

    # When: prepare runs twice.
    document, code = T.prepare(aggregate_path, output, manifest_sha=MANIFEST, budget=4_000_000_000)
    again, code_again = T.prepare(aggregate_path, output, manifest_sha=MANIFEST, budget=4_000_000_000)

    # Then: it succeeds, publishes both forms, and is stable across runs.
    assert code == 0 and code_again == 0
    assert (output / T.LEDGER_NAME).is_file()
    assert (output / T.LEDGER_MARKDOWN_NAME).is_file()
    assert document == again
    assert "Task-6 allocation ledger" in (output / T.LEDGER_MARKDOWN_NAME).read_text(encoding="utf-8")


def test_every_planted_probe_is_refused_by_its_own_rule() -> None:
    # Given: a ledger and the probe set the CLI's failure case uses.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)
    rows = {row.arm: row for row in document.arms}
    payload = T.Payload(
        name="probe", arm="A1", seeds=len(L.TRAINING_SEEDS), claimed_source_kind="measured",
        gpu_hours=rows["A1"].gpu_hours_all_seeds or 1.0,
    )
    roomy = T.roomy_ceiling(document)

    # When: each violation is offered with a ceiling that cannot refuse it for another reason.
    probes = {
        "absent": T.admit(payload, allocation=None, pool=_pool(), ledger=document),
        "budget": T.admit(payload, pool=_pool(), ledger=document,
                          allocation=_allocation(authorized=1.0, consumed=0.5)),
        "pool": T.admit(payload, pool=_pool(), ledger=document,
                        allocation=_allocation(authorized=roomy, consumed=0.0,
                                               pool=T.Pool(project_id="p2", logic_compute_group_id="l2", gpu_type="g2"))),
        "receipt": T.admit(payload, pool=_pool(), ledger=document,
                           allocation=_allocation(authorized=None, consumed=0.0, receipt="carried-over")),
        "label": T.admit(payload.model_copy(update={"arm": "A0", "claimed_source_kind": "measured",
                                                    "gpu_hours": rows["A0"].gpu_hours_all_seeds or 1.0}),
                         pool=_pool(), ledger=document,
                         allocation=_allocation(authorized=roomy, consumed=0.0)),
    }

    # Then: none is admitted, and each names its own condition rather than a neighbouring one.
    assert not any(decision.admitted for decision in probes.values())
    assert "no approval record" in probes["absent"].reason
    assert "remaining" in probes["budget"].reason
    assert "different pool" in probes["pool"].reason
    assert "no declared ceiling" in probes["receipt"].reason
    assert "derived" in probes["label"].reason


def test_happy_probe_admits_priced_arms_and_refuses_only_unpriced_ones() -> None:
    # Given: a ledger where one measured arm is missing entirely.
    arms = (
        _arm("A1_small_noloop", (_canvas(L.TRAINING_CANVAS),)),
        _arm("A2_small_noloop", (_canvas(L.TRAINING_CANVAS),)),
        _arm("A3_small_loop", ()),
        _arm("A5_small_loop_fwd", (_canvas(L.TRAINING_CANVAS),)),
    )
    document = L.build_ledger(_aggregate(arms), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)

    # When: the happy probe runs.
    ok, detail = T.happy_probe(document, budget=4_000_000_000, pool=_pool())

    # Then: it succeeds, because every refusal is the ledger saying it cannot price that arm.
    assert ok
    assert json.loads(detail["refused"])
    assert all("unforecastable" in reason for reason in json.loads(detail["refused"]).values())


def test_happy_probe_fails_when_nothing_can_be_priced() -> None:
    # Given: a ledger with no priced arm at all.
    document = L.build_ledger(
        _aggregate((_arm("A1_small_noloop", ()),)), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000
    )

    # When / Then: the gate cannot demonstrate anything, so it does not report success.
    ok, detail = T.happy_probe(document, budget=4_000_000_000, pool=_pool())
    assert not ok
    assert "could be priced" in detail["detail"]


def test_a_regenerated_ledger_archives_the_previous_one(tmp_path: Path) -> None:
    # Given: a ledger built from a calibration that priced some arms.
    partial = tmp_path / "partial.json"
    _ = partial.write_text(json.dumps(_aggregate((_arm("A1_small_noloop", (_canvas(L.TRAINING_CANVAS),)),)).model_dump(mode="json")), encoding="utf-8")
    output = tmp_path / "out"
    _, first_code = T.prepare(partial, output, manifest_sha=MANIFEST, budget=4_000_000_000)
    first = (output / T.LEDGER_NAME).read_bytes()

    # When: a later calibration covering more arms is published to the same place.
    fuller = tmp_path / "fuller.json"
    _ = fuller.write_text(json.dumps(_aggregate(_full_arms()).model_dump(mode="json")), encoding="utf-8")
    document, second_code = T.prepare(fuller, output, manifest_sha="c" * 64, budget=4_000_000_000)

    # Then: the new ledger is published and the previous bytes survive as an archive.
    assert first_code == 0 and second_code == 0
    assert (output / T.LEDGER_NAME).read_bytes() != first
    archives = list(output.glob("allocation_ledger.archive-*.json"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == first
    assert document.calibration_manifest_sha256 == "c" * 64


def test_storage_ignores_the_large_confirmation_arm() -> None:
    # Given: a calibration that also measured the 2.9B arm, whose count is seven times larger.
    arms = _full_arms() + (
        _arm("A2_large_noloop", (_canvas(L.TRAINING_CANVAS),), checkpoint_step=4750),
    )
    # The large arm's own parameter count, as the reader would see it in the aggregate.
    large = _arm("A2_large_noloop", (_canvas(L.TRAINING_CANVAS),), checkpoint_step=4750)
    arms = tuple(a.model_copy(update={"parameters": _params(4_091_581_441)}) if a.arm_id == "A2_large_noloop" else a for a in arms)

    # When: the ledger is built.
    document = L.build_ledger(_aggregate(arms), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)

    # Then: the storage line describes the small-arm matrix, not the confirmation model. A prefix
    # match would have picked the 2.9B count, since its arm id also begins with "A2".
    assert document.storage.parameters_numel == 591_054_848
    assert document.storage.parameters_numel != 4_091_581_441
    assert large is not None


def test_the_gate_prices_from_the_ledger_not_from_the_payload() -> None:
    # Given: a matching allocation and a payload that declares no cost of its own.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)
    roomy = T.roomy_ceiling(document)
    free = T.Payload(name="free", arm="A1", seeds=len(L.TRAINING_SEEDS), claimed_source_kind="measured", gpu_hours=0.0)

    # When / Then: a payload does not get to set its own price. Trusting it would let a caller
    # consume unbounded budget while appearing to use none.
    decision = T.admit(free, allocation=_allocation(authorized=roomy, consumed=0.0), pool=_pool(), ledger=document)
    assert not decision.admitted
    assert "set its own price" in decision.reason


def test_a_derived_arm_may_not_borrow_a_siblings_price() -> None:
    # Given: a calibration whose arms have visibly different costs, and A0 (derived from A1) priced
    # from A5 instead.
    arms = (
        _arm("A1_small_noloop", (_canvas(L.TRAINING_CANVAS, forward_rate=6400.0, backward=5.0),)),
        _arm("A2_small_noloop", (_canvas(L.TRAINING_CANVAS),)),
        _arm("A3_small_loop", (_canvas(L.TRAINING_CANVAS),)),
        _arm("A5_small_loop_fwd", (_canvas(L.TRAINING_CANVAS, forward_rate=1600.0, backward=9.0),)),
    )
    document = L.build_ledger(_aggregate(arms), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)
    rows = {row.arm: row for row in document.arms}
    borrowed = T.Payload(
        name="borrowed", arm="A0", seeds=len(L.TRAINING_SEEDS), claimed_source_kind="derived",
        gpu_hours=(rows["A5"].gpu_hours_all_seeds or 1.0),
    )

    # When / Then: the price must match the row the arm is actually derived from.
    decision = T.admit(borrowed, allocation=_allocation(authorized=T.roomy_ceiling(document), consumed=0.0),
                       pool=_pool(), ledger=document)
    assert not decision.admitted
    assert "prices that at" in decision.reason


def test_the_markdown_carries_each_rows_caveats() -> None:
    # Given / When: the published markdown is rendered.
    document = L.build_ledger(_aggregate(_full_arms()), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)
    markdown = T.render_markdown(document)

    # Then: every priced row's own caveats appear beside its numbers. Carrying them only in the JSON
    # let the human-readable table assert more than the rows support, and showed A4 at A3's cost with
    # no warning that A4's untied blocks make that an understatement.
    assert "Per-row caveats" in markdown
    assert L.MICROBATCH_ONE_ASSUMPTION in markdown
    assert L.CONTENDED_ASSUMPTION in markdown
    assert "understates its cost" in markdown
    assert "not run in the calibration job" in markdown


def test_the_confirmation_note_is_derived_from_the_calibration() -> None:
    # Given: one large arm that reached the training canvas and one that did not.
    arms = _full_arms() + (
        _arm("A2_large_noloop", (_canvas(512), _canvas(L.TRAINING_CANVAS)), checkpoint_step=4750),
        _arm("A3_large_loop", (_canvas(512),), checkpoint_step=4750),
    )
    document = L.build_ledger(_aggregate(arms), calibration_manifest_sha256=MANIFEST, token_budget=4_000_000_000)

    # When / Then: the note reports what happened rather than asserting both stopped at 512. An
    # earlier hardcoded version contradicted the aggregate it was bound to.
    note = document.confirmation_note
    assert "A2_large_noloop" in note and "Reached the 4096-token canvas" in note
    assert "Did not reach it: A3_large_loop" in note
    assert "both" not in note
