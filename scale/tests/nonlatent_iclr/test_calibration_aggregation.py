"""Checks of the calibration aggregation command over published rank records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from scale.experiments.nonlatent_iclr.qualification import calibration_contracts as cc
from scale.experiments.nonlatent_iclr.qualification import cli as qualification_cli

NONCE = "0" * 32
AGGREGATE = qualification_cli.CALIBRATION_AGGREGATE_NAME


def _geometry(layers: int) -> cc.MeasuredGeometry:
    return cc.MeasuredGeometry(
        num_hidden_layers=layers, hidden_size=64, vocab_size=128, num_heads=None,
        head_dim=8, intermediate_size=128,
    )


def _sidecar(rank: int, *, status: Literal["PASSED", "FAILED"] = "PASSED", arm_id: str = "A3_large_loop",
             step: int | None = 4750, layers: int = 32, nonce: str = NONCE) -> cc.CalibrationSidecar:
    measurements = (
        cc.CalibrationMeasurement(
            canvas_tokens=4096, forward_tokens_per_second=1000.0, backward_seconds=1.0,
            optimizer_step_not_measured_reason="sharded training is not a single-GPU step",
            selected_tokens=100, sampler_nfe_seconds=0.5,
            peak_allocated_bytes=10, peak_reserved_bytes=20,
        ),
    ) if status == "PASSED" else ()
    return cc.CalibrationSidecar(
        rank=rank, nonce=nonce, status=status,
        detail="measured" if status == "PASSED" else "rank failed",
        arm_id=arm_id, arm_digest="a" * 64, nominal_label=arm_id,
        measured=True, loop_reps_configured=1, loop_reps_measured=0,
        force_forward=False, gradient_checkpointing=True,
        model_geometry=_geometry(layers) if status == "PASSED" else None,
        parameters=cc.MeasuredParameters(total_tensors=4, total_numel=100, loop_tensors=1, loop_numel=8)
        if status == "PASSED" else None,
        checkpoint_step=step, checkpoint_sha256="b" * 64 if step is not None else None,
        measurements=measurements,
    )


def _publish(output_dir: Path, sidecars: tuple[cc.CalibrationSidecar, ...]) -> None:
    for sidecar in sidecars:
        _ = sidecar.write_once(output_dir)


def _aggregate_command(output_dir: Path) -> tuple[str, ...]:
    """The launcher's aggregation invocation: the shared probe surface plus the run identity."""
    return (
        "aggregate-calibration",
        "--output-dir", str(output_dir),
        "--nonce", NONCE,
        "--manifest-sha256", "c" * 64,
        "--controller-receipt", str(output_dir / "controller-receipt.json"),
        "--run-id", "qualification-20260912-01-abcdefgh",
    )


def _arm_table_sidecars(**overrides) -> tuple[cc.CalibrationSidecar, ...]:
    """Eight records matching the declared rank table, so grouping is exercised for real."""
    from scale.experiments.nonlatent_iclr.qualification import calibration_arms as arms

    out = []
    for rank in range(arms.RANK_COUNT):
        spec = arms.arm_for_rank(rank)
        out.append(_sidecar(
            rank,
            arm_id=spec.arm_id,
            step=spec.checkpoint_step,
            layers=32 if spec.checkpoint_dir is not None else 24,
            **overrides,
        ))
    return tuple(out)


def test_aggregate_calibration_publishes_per_arm_results(tmp_path: Path) -> None:
    # Given: a complete, passing job whose ranks follow the declared arm table.
    _publish(tmp_path, _arm_table_sidecars())

    # When: the aggregation command runs.
    exit_code = qualification_cli.main(_aggregate_command(tmp_path))

    # Then: it publishes an aggregate grouped by arm, with the derived arms declared.
    assert exit_code == 0
    document = json.loads((tmp_path / AGGREGATE).read_text(encoding="utf-8"))
    assert document["status"] == "PASSED"
    assert document["ranks_passed"] == 8
    assert {"A0", "A4"} == set(document["derived_arms"])
    arm_ids = {arm["arm_id"] for arm in document["arms"]}
    assert "A0" not in arm_ids
    assert "A3_large_loop" in arm_ids
    by_id = {arm["arm_id"]: arm for arm in document["arms"]}
    assert by_id["A2_small_noloop"]["contributing_ranks"] == 2
    assert by_id["A2_small_noloop"]["checkpoint_step"] is None
    assert by_id["A3_large_loop"]["checkpoint_step"] == 4750


def test_aggregate_calibration_publishes_a_partial_job(tmp_path: Path) -> None:
    # Given: one rank that failed to measure anything.
    sidecars = list(_arm_table_sidecars())
    sidecars[3] = _sidecar(3, status="FAILED", arm_id="A5_small_loop_fwd")
    _publish(tmp_path, tuple(sidecars))

    # When: the aggregation command runs.
    exit_code = qualification_cli.main(_aggregate_command(tmp_path))

    # Then: the partial job is published, and the failure is named rather than dropped.
    assert exit_code == 0
    document = json.loads((tmp_path / AGGREGATE).read_text(encoding="utf-8"))
    assert document["status"] == "PARTIAL"
    assert document["ranks_failed"] == 1
    by_id = {arm["arm_id"]: arm for arm in document["arms"]}
    assert by_id["A5_small_loop_fwd"]["unavailable_ranks"] == [3]
    assert by_id["A5_small_loop_fwd"]["contributing_ranks"] == 0


def test_aggregate_calibration_refuses_a_missing_rank(tmp_path: Path) -> None:
    # Given: a job that never published rank 5.
    sidecars = tuple(sidecar for sidecar in _arm_table_sidecars() if sidecar.rank != 5)
    _publish(tmp_path, sidecars)

    # When: the aggregation command runs.
    exit_code = qualification_cli.main(_aggregate_command(tmp_path))

    # Then: nothing is published from an incomplete job.
    assert exit_code == 1
    assert not (tmp_path / AGGREGATE).exists()


def test_aggregate_calibration_refuses_a_foreign_nonce(tmp_path: Path) -> None:
    # Given: a record bound to a different run.
    _publish(tmp_path, _arm_table_sidecars(nonce="1" * 32))

    # When / Then: the nonce binding rejects it.
    assert qualification_cli.main(_aggregate_command(tmp_path)) == 1
    assert not (tmp_path / AGGREGATE).exists()


def test_aggregate_calibration_refuses_an_unexpected_record(tmp_path: Path) -> None:
    # Given: a complete job plus a stray record outside the declared rank range.
    _publish(tmp_path, _arm_table_sidecars())
    (tmp_path / "calibration-rank-99.json").write_text(_sidecar(7).model_dump_json(), encoding="utf-8")

    # When / Then: the stray record is named and nothing is published.
    assert qualification_cli.main(_aggregate_command(tmp_path)) == 1
    assert not (tmp_path / AGGREGATE).exists()


def test_aggregate_calibration_refuses_a_job_with_nothing_measured(tmp_path: Path) -> None:
    # Given: every rank failed.
    sidecars = tuple(
        _sidecar(rank, status="FAILED", arm_id="A3_large_loop") for rank in range(8)
    )
    _publish(tmp_path, sidecars)

    # When / Then: a job with no usable measurement is a failure, not a partial success.
    assert qualification_cli.main(_aggregate_command(tmp_path)) == 1


def test_aggregate_calibration_is_write_once(tmp_path: Path) -> None:
    # Given: an aggregate already published by an earlier identical invocation.
    _publish(tmp_path, _arm_table_sidecars())
    assert qualification_cli.main(_aggregate_command(tmp_path)) == 0
    first = (tmp_path / AGGREGATE).read_bytes()

    # When: the same command runs again.
    assert qualification_cli.main(_aggregate_command(tmp_path)) == 0

    # Then: the published bytes are unchanged rather than rewritten.
    assert (tmp_path / AGGREGATE).read_bytes() == first


def test_aggregate_calibration_reports_a_reason_instead_of_raising(tmp_path: Path) -> None:
    # Given: a truncated record, as a killed rank would leave behind.
    _publish(tmp_path, _arm_table_sidecars())
    (tmp_path / "calibration-rank-6.json").write_text('{"rank": 6, "arm_id"', encoding="utf-8")

    # When: the aggregation command runs.
    exit_code = qualification_cli.main(_aggregate_command(tmp_path))

    # Then: it refuses with a printed reason. Raising here would leave stdout empty, and the
    # launcher publishes stdout verbatim, so the reviewer would get an empty aggregate.json.
    assert exit_code == 1
    assert not (tmp_path / AGGREGATE).exists()


def test_aggregate_calibration_reports_disagreeing_identity_instead_of_raising(tmp_path: Path) -> None:
    # Given: two ranks of the same arm that disagree about the model they measured.
    sidecars = list(_arm_table_sidecars())
    sidecars[4] = _sidecar(4, arm_id="A2_small_noloop", step=None, layers=48)
    sidecars[1] = _sidecar(1, arm_id="A2_small_noloop", step=None, layers=24)
    _publish(tmp_path, tuple(sidecars))

    # When / Then: the disagreement is a reported refusal, not a traceback.
    assert qualification_cli.main(_aggregate_command(tmp_path)) == 1
    assert not (tmp_path / AGGREGATE).exists()
