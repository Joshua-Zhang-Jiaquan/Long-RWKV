"""Calibration-contract tests: per-arm grouping, partial jobs, mismatched ranks, promotion guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import pytest

from scale.experiments.nonlatent_iclr.qualification import calibration_contracts as cc


def _geometry(layers: int = 32) -> cc.MeasuredGeometry:
    return cc.MeasuredGeometry(
        num_hidden_layers=layers, hidden_size=64, vocab_size=128, num_heads=None,
        head_dim=8, intermediate_size=128,
    )


def _parameters(numel: int = 4_091_581_441) -> cc.MeasuredParameters:
    return cc.MeasuredParameters(total_tensors=10, total_numel=numel, loop_tensors=2, loop_numel=8)


def _measurement(canvas: int, rank: int) -> cc.CalibrationMeasurement:
    return cc.CalibrationMeasurement(
        canvas_tokens=canvas,
        forward_tokens_per_second=1000.0 + rank,
        backward_seconds=1.0 + rank / 100,
        optimizer_step_seconds=0.25,
        selected_tokens=canvas // 3,
        sampler_nfe_seconds=0.5,
        peak_allocated_bytes=8_000_000_000,
        peak_reserved_bytes=9_000_000_000,
    )


def _sidecar(
    rank: int,
    *,
    status: Literal["PASSED", "FAILED"] = "PASSED",
    step: int | None = 4750,
    numel: int = 4_091_581_441,
    arm_id: str = "A3_large_loop",
    layers: int = 32,
) -> cc.CalibrationSidecar:
    measurements = tuple(_measurement(canvas, rank) for canvas in cc.CANVAS_TOKENS) if status == "PASSED" else ()
    return cc.CalibrationSidecar(
        rank=rank, nonce="0" * 32, status=status,
        detail="measured" if status == "PASSED" else "rank failed",
        arm_id=arm_id, arm_digest="a" * 64, nominal_label=f"{arm_id} label",
        measured=True, loop_reps_configured=1, loop_reps_measured=step is not None and 1 or 0,
        force_forward=False, gradient_checkpointing=True,
        model_geometry=_geometry(layers) if status == "PASSED" else None,
        parameters=_parameters(numel) if status == "PASSED" else None,
        checkpoint_step=step, checkpoint_sha256="b" * 64 if step is not None else None,
        measurements=measurements,
    )


def test_sidecar_rejects_a_passing_record_without_measurements() -> None:
    # Given: a rank claiming success but reporting nothing.
    with pytest.raises(ValueError, match="without_measurements"):
        _ = cc.CalibrationSidecar(
            rank=0, nonce="0" * 32, status="PASSED", detail="ok", arm_id="A1", arm_digest="a" * 64,
            nominal_label="x", measured=True, loop_reps_configured=0, loop_reps_measured=0,
            force_forward=False, gradient_checkpointing=True,
            model_geometry=_geometry(), parameters=_parameters(), checkpoint_step=1,
            checkpoint_sha256="b" * 64,
        )


def test_sidecar_rejects_a_failed_record_carrying_measurements() -> None:
    # Given: a rank claiming failure while reporting measurements.
    with pytest.raises(ValueError, match="carries_measurements"):
        _ = cc.CalibrationSidecar(
            rank=0, nonce="0" * 32, status="FAILED", detail="failed", arm_id="A1", arm_digest="a" * 64,
            nominal_label="x", measured=True, loop_reps_configured=0, loop_reps_measured=0,
            force_forward=False, gradient_checkpointing=True, measurements=(_measurement(4096, 0),),
        )


def test_sidecar_rejects_a_passing_record_without_a_measured_identity() -> None:
    # Given: a rank that measured a canvas but never recorded what it measured.
    with pytest.raises(ValueError, match="without_a_measured_identity"):
        _ = cc.CalibrationSidecar(
            rank=0, nonce="0" * 32, status="PASSED", detail="ok", arm_id="A1", arm_digest="a" * 64,
            nominal_label="x", measured=True, loop_reps_configured=0, loop_reps_measured=0,
            force_forward=False, gradient_checkpointing=True, measurements=(_measurement(4096, 0),),
        )


def test_sidecar_rejects_a_derived_arm_publishing_measurements() -> None:
    # Given: an arm the job cannot run, claiming a measurement.
    with pytest.raises(ValueError, match="derived_arm_cannot_publish"):
        _ = cc.CalibrationSidecar(
            rank=0, nonce="0" * 32, status="PASSED", detail="ok", arm_id="A0", arm_digest="a" * 64,
            nominal_label="x", measured=False, derived_from="A1_small_noloop",
            loop_reps_configured=0, loop_reps_measured=0, force_forward=True,
            gradient_checkpointing=True, model_geometry=_geometry(), parameters=_parameters(),
            measurements=(_measurement(4096, 0),),
        )


def test_sidecar_rejects_a_checkpoint_step_without_its_digest() -> None:
    # Given: a checkpoint identity half-declared.
    with pytest.raises(ValueError, match="declared_together"):
        _ = cc.CalibrationSidecar(
            rank=0, nonce="0" * 32, status="FAILED", detail="failed", arm_id="A1", arm_digest="a" * 64,
            nominal_label="x", measured=True, loop_reps_configured=0, loop_reps_measured=0,
            force_forward=False, gradient_checkpointing=True, checkpoint_step=4750,
        )


def test_sidecar_accepts_an_hf_only_arm_without_a_checkpoint() -> None:
    # Given: a small arm that deliberately loads no checkpoint.
    sidecar = _sidecar(0, step=None, arm_id="A2_small_noloop")
    # Then: it publishes with an explicitly absent checkpoint rather than a borrowed one.
    assert sidecar.checkpoint_step is None
    assert sidecar.checkpoint_sha256 is None


def test_measurement_rejects_reserved_below_allocated() -> None:
    # Given: an impossible memory reading.
    with pytest.raises(ValueError, match="reserved_bytes_below"):
        _ = cc.CalibrationMeasurement(
            canvas_tokens=4096, forward_tokens_per_second=1.0, backward_seconds=1.0,
            optimizer_step_not_measured_reason="scheduled off", selected_tokens=1,
            sampler_nfe_seconds=1.0, peak_allocated_bytes=10, peak_reserved_bytes=9,
        )


def test_sidecar_write_once_refuses_to_replace(tmp_path: Path) -> None:
    # Given: a published rank record.
    sidecar = _sidecar(0)
    path = sidecar.write_once(tmp_path)
    # When: it is rewritten unchanged, then with different content.
    assert sidecar.write_once(tmp_path) == path
    # Then: a conflicting rewrite is refused.
    with pytest.raises(ValueError, match="already exists"):
        _ = _sidecar(0, numel=7).write_once(tmp_path)


def test_aggregate_requires_every_rank() -> None:
    # Given: fewer than eight rank records.
    with pytest.raises(ValueError, match="expected 8"):
        _ = cc.aggregate(tuple(_sidecar(rank) for rank in range(7)))


def test_aggregate_marks_a_failed_rank_partial_and_names_it() -> None:
    # Given: eight records where one rank failed.
    sidecars = tuple(_sidecar(rank, status="FAILED" if rank == 3 else "PASSED") for rank in range(8))
    # When: the job is summarised.
    result = cc.aggregate(sidecars)
    # Then: the job is partial, the failure is named, and it is not silently dropped.
    assert result.status == "PARTIAL"
    assert result.ranks_failed == 1
    assert result.ranks_passed == 7
    assert result.arms[0].unavailable_ranks == (3,)
    assert result.arms[0].contributing_ranks == 7


def test_aggregate_rejects_disagreeing_checkpoint_identity_within_an_arm() -> None:
    # Given: ranks of one arm reporting different checkpoints.
    sidecars = tuple(_sidecar(rank, step=4750 if rank else 4000) for rank in range(8))
    # When / Then: the aggregate refuses rather than averaging across identities.
    with pytest.raises(ValueError, match="model identity"):
        _ = cc.aggregate(sidecars)


def test_aggregate_keeps_arms_apart() -> None:
    # Given: two arms measured on different ranks, one of them twice.
    sidecars = tuple(
        _sidecar(rank, arm_id="A2_small_noloop", numel=900, step=None, layers=24)
        if rank in (1, 4)
        else _sidecar(rank)
        for rank in range(8)
    )
    # When: the job is summarised.
    result = cc.aggregate(sidecars)
    # Then: each arm keeps its own identity and rank set, and no timing is mixed across arms.
    by_id = {arm.arm_id: arm for arm in result.arms}
    assert set(by_id) == {"A2_small_noloop", "A3_large_loop"}
    assert by_id["A2_small_noloop"].contributing_ranks == 2
    assert by_id["A2_small_noloop"].parameters is not None
    assert by_id["A2_small_noloop"].parameters.total_numel == 900
    assert by_id["A2_small_noloop"].checkpoint_step is None
    assert by_id["A3_large_loop"].contributing_ranks == 6
    assert by_id["A3_large_loop"].checkpoint_step == 4750


def test_aggregate_reports_per_canvas_medians_within_an_arm() -> None:
    # Given: eight passing ranks of one arm with per-rank variation.
    result = cc.aggregate(tuple(_sidecar(rank) for rank in range(8)))
    # Then: each canvas carries a median over all eight ranks, plus the identity and bindings.
    arm = result.arms[0]
    assert result.ranks_passed == 8
    assert [canvas.canvas_tokens for canvas in arm.canvases] == list(cc.CANVAS_TOKENS)
    for canvas in arm.canvases:
        assert canvas.contributing_ranks == 8
        assert abs(canvas.forward_tokens_per_second - 1003.5) < 1e-9
        assert canvas.backward_seconds is not None
        assert abs(canvas.backward_seconds - 1.035) < 1e-9
        assert canvas.backward_contributing_ranks == 8
        assert canvas.backward_not_measured_reason is None
        assert abs(canvas.optimizer_step_seconds - 0.25) < 1e-9
        assert canvas.selected_tokens_min == canvas.canvas_tokens // 3
        assert canvas.selected_tokens_max == canvas.canvas_tokens // 3
    assert arm.parameters is not None
    assert arm.parameters.total_numel == 4_091_581_441
    assert set(result.sidecar_sha256) == {f"rank-{rank}" for rank in range(8)}
    assert all(len(digest) == 64 for digest in result.sidecar_sha256.values())


def test_aggregate_carries_the_scope_and_refuses_downstream_claims() -> None:
    # Given: a complete job.
    result = cc.aggregate(tuple(_sidecar(rank) for rank in range(8)))
    # Then: the record declares itself as forecast input only, and names what it is not.
    assert result.scope == "task6_forecast_input_only"
    assert "long-context capability" in result.claims_not_made
    assert "sustainable goodput" in result.claims_not_made
    assert "training throughput at production sharding" in result.claims_not_made


def test_aggregate_carries_derived_arms_without_measuring_them() -> None:
    # Given: arms the job cannot run.
    result = cc.aggregate(
        tuple(_sidecar(rank) for rank in range(8)), derived_arms={"A0": "A1_small_noloop"}
    )
    # Then: they are declared as derived and contribute no measurement.
    assert result.derived_arms == {"A0": "A1_small_noloop"}
    assert "A0" not in {arm.arm_id for arm in result.arms}


def test_aggregate_collects_warnings_from_every_rank() -> None:
    # Given: ranks reporting precision or support caveats.
    sidecars = tuple(
        _sidecar(rank).model_copy(update={"warnings": ("bf16_state_precision",) if rank % 2 else ("no_cache_qualified",)})
        for rank in range(8)
    )
    # When: the job is summarised.
    result = cc.aggregate(sidecars)
    # Then: no caveat is dropped.
    assert set(result.warnings) == {"bf16_state_precision", "no_cache_qualified"}


def test_aggregate_rejects_disagreeing_gradient_checkpointing() -> None:
    # Given: ranks configured differently, which would make the timings incomparable.
    sidecars = tuple(
        _sidecar(rank).model_copy(update={"gradient_checkpointing": rank % 2 == 0}) for rank in range(8)
    )
    # When / Then: the aggregate refuses.
    with pytest.raises(ValueError, match="gradient-checkpointing"):
        _ = cc.aggregate(sidecars)


def test_load_sidecar_round_trips_a_published_record(tmp_path: Path) -> None:
    # Given: a record published by a rank.
    sidecar = _sidecar(2)
    path = sidecar.write_once(tmp_path)
    # When: it is read back.
    loaded = cc.load_sidecar(path)
    # Then: it round-trips exactly.
    assert loaded == sidecar


def test_load_sidecar_rejects_an_unknown_field(tmp_path: Path) -> None:
    # Given: a record carrying a field the contract does not define.
    payload = cast("dict[str, object]", json.loads(_sidecar(1).model_dump_json()))
    payload["invented_metric"] = 1
    path = tmp_path / "calibration-rank-1.json"
    _ = path.write_text(json.dumps(payload), encoding="utf-8")
    # When / Then: extra fields are forbidden, so a stray claim cannot slip in.
    with pytest.raises(Exception, match="invented_metric"):
        _ = cc.load_sidecar(path)


def test_measurement_rejects_a_backward_step_that_is_neither_measured_nor_explained() -> None:
    # Given: a row that silently omits the backward step.
    with pytest.raises(ValueError, match="either_measured_or_explained"):
        _ = cc.CalibrationMeasurement(
            canvas_tokens=4096, forward_tokens_per_second=1.0, selected_tokens=1,
            optimizer_step_not_measured_reason="scheduled off", sampler_nfe_seconds=1.0,
            peak_allocated_bytes=10, peak_reserved_bytes=10,
        )


def test_measurement_rejects_both_a_measurement_and_an_excuse() -> None:
    # Given: a row claiming both a duration and an excuse.
    with pytest.raises(ValueError, match="either_measured_or_explained"):
        _ = cc.CalibrationMeasurement(
            canvas_tokens=4096, forward_tokens_per_second=1.0, backward_seconds=1.0,
            backward_not_measured_reason="contradiction", selected_tokens=1,
            optimizer_step_not_measured_reason="scheduled off", sampler_nfe_seconds=1.0,
            peak_allocated_bytes=10, peak_reserved_bytes=10,
        )


def test_measurement_rejects_an_optimizer_step_that_is_neither_measured_nor_explained() -> None:
    # Given: a row that silently omits the optimizer step.
    with pytest.raises(ValueError, match="optimizer_measurement_must_be_either"):
        _ = cc.CalibrationMeasurement(
            canvas_tokens=4096, forward_tokens_per_second=1.0, backward_seconds=1.0,
            selected_tokens=1, sampler_nfe_seconds=1.0,
            peak_allocated_bytes=10, peak_reserved_bytes=10,
        )


def test_aggregate_reports_an_unmeasured_backward_step_with_its_reason() -> None:
    # Given: eight ranks that measured forward but not the backward step.
    unmeasured = cc.CalibrationMeasurement(
        canvas_tokens=4096, forward_tokens_per_second=10.0, sampler_nfe_seconds=0.1,
        backward_not_measured_reason="no runner", selected_tokens=0,
        optimizer_step_not_measured_reason="scheduled off",
        peak_allocated_bytes=10, peak_reserved_bytes=20,
    )
    sidecars = tuple(
        _sidecar(rank).model_copy(update={"measurements": (unmeasured,), "detail": "forward only"})
        for rank in range(8)
    )
    # When: the job is summarised.
    result = cc.aggregate(sidecars)
    # Then: the gap is reported, not averaged away into a number.
    canvas = result.arms[0].canvases[0]
    assert canvas.backward_seconds is None
    assert canvas.backward_contributing_ranks == 0
    assert canvas.backward_not_measured_reason == "no runner"
