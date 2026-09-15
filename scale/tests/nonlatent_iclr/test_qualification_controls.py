from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from scale.experiments.nonlatent_iclr.qualification.controller_receipt import (
    ControllerReceipt,
)
from scale.experiments.nonlatent_iclr.qualification.controls import (
    REQUIRED_CHECKS,
    QualificationLimits,
    RankResult,
    aggregate_results,
    parse_probe_config,
)


def _controller_binding(
    nonce: str,
    override: tuple[str, JsonValue] | None = None,
) -> JsonValue:
    payload: dict[str, JsonValue] = {
        "schema_version": 1,
        "job_id": "job-12345678-1234-4abc-8def-1234567890ab",
        "job_identity_origin": "controller_qz_createjob_receipt",
        "submitted_request_sha256": "8" * 64,
        "authorization_id": "nonlatent-h100-qualification-20260912-01",
        "project_id": "project-160ccb20-98ab-4538-a847-01d1f83d5b0f",
        "workspace_id": "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
        "logic_compute_group_id": "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e",
        "spec_id": "7166bd2e-6cbe-4bd9-be38-762d11003e7f",
        "run_id": "qualification-20260912-01-deadbeef",
        "nonce": nonce,
        "source_manifest_sha256": "a" * 64,
        "requested_image": "docker.sii.shaipower.online/inspire-studio/relay2:v2",
        "scheduler_image_id": "image-7330d118-df9d-4e2e-82b1-c2543e831eb4",
        "observed_image_digest": None,
        "authorized_gpu_type": "NVIDIA_H100_SXM_80G",
        "requested_nodes": 1,
        "requested_gpus_per_node": 8,
        "requested_gpus": 8,
        "maximum_runtime_seconds": 1_800,
        "maximum_gpu_hours": 4,
        "maximum_job_submissions": 1,
        "maximum_automatic_retries": 0,
    }
    if override is not None:
        field, value = override
        payload[field] = value
    return ControllerReceipt.model_validate(payload).to_binding_evidence().model_dump(
        mode="json"
    )


def _success_evidence(
    rank: int,
    elapsed_seconds: float = 1.0,
    *,
    nonce: str = "a" * 32,
) -> JsonValue:
    return {
        "kind": "success",
        "start_utc": 1_789_171_200.0,
        "end_utc": 1_789_171_201.0,
        "elapsed_monotonic_seconds": elapsed_seconds,
        "rank": rank,
        "local_rank": rank,
        "world_size": 8,
        "gpu_inventory": [
            {"memory_mib": 81_559, "name": "NVIDIA H100 80GB HBM3", "uuid": f"GPU-{gpu}"}
            for gpu in range(8)
        ],
        "controller_binding": _controller_binding(nonce),
        "python_version": "3.12.3",
        "torch_version": "2.8.0a0+5228986c39.nv25.6",
        "fla_version": "0.5.0",
        "fla_core_version": "0.5.0",
        "transformers_version": "5.3.0",
        "safetensors_version": "0.5.3",
        "triton_version": "3.3.0+git96316ce52.nvinternal",
        "source_manifest_sha256": "a" * 64,
        "verified_source_files": 18,
        "checkpoint_step": 4_750,
        "checkpoint_sha256": "ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07",
        "checkpoint_size_bytes": 16_367_167_378,
        "checkpoint_state_tensors": 1_955,
        "geometry": {
            "head_dim": 64,
            "hidden_size": 2_560,
            "intermediate_size": 10_240,
            "num_heads": 40,
            "num_hidden_layers": 32,
            "vocab_size": 65_536,
        },
        "parameters": {
            "backward_attention_numel": 934_049_280,
            "backward_attention_tensors": 829,
            "forward_attention_numel": 934_049_280,
            "forward_attention_tensors": 829,
            "fusion_numel": 209_797_120,
            "fusion_tensors": 64,
            "loop_numel": 1,
            "loop_tensors": 1,
            "shared_numel": 2_013_685_760,
            "shared_tensors": 230,
            "total_numel": 4_091_581_441,
            "total_tensors": 1_953,
        },
        "fla_causal_cache": {
            "atol": 0.05,
            "max_abs_delta": 0.0,
            "prefix_length": 32,
            "rtol": 0.0,
            "scope": "standalone_forward_rwkv7_attention",
            "sequence_length": 48,
        },
        "masked_logits_all_finite": True,
        "masked_logits_shape": [1, 48, 65_536],
        "memory": {
            "allocated_after_load_bytes": 8_500_000_000,
            "peak_allocated_bytes": 9_000_000_000,
            "peak_reserved_bytes": 10_000_000_000,
            "physical_bytes": 85_000_000_000,
        },
        "fullmodel_cachedprefix": "NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED",
        "loop_cachedprefix": "NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES",
    }


def _write_passing_rank(output_dir: Path, rank: int, nonce: str) -> None:
    payload: JsonValue = {
        "checks": list(REQUIRED_CHECKS),
        "detail": "all_bounded_checks_passed",
        "evidence": _success_evidence(rank, nonce=nonce),
        "nonce": nonce,
        "rank": rank,
        "schema_version": 1,
        "status": "PASSED",
    }
    (output_dir / f"rank-{rank}.json").write_text(
        json.dumps(payload, allow_nan=True), encoding="utf-8"
    )


def _controller_bound_success(rank: int, nonce: str) -> dict[str, JsonValue]:
    evidence = _success_evidence(rank, nonce=nonce)
    if not isinstance(evidence, dict):
        raise AssertionError("success evidence fixture must be an object")
    return evidence


def _write_controller_bound_rank(
    output_dir: Path,
    rank: int,
    nonce: str,
) -> None:
    evidence = _controller_bound_success(rank, nonce)
    payload: JsonValue = {
        "checks": list(REQUIRED_CHECKS),
        "detail": "all_bounded_checks_passed",
        "evidence": evidence,
        "nonce": nonce,
        "rank": rank,
        "schema_version": 1,
        "status": "PASSED",
    }
    (output_dir / f"rank-{rank}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_foreign_controller_rank(
    output_dir: Path,
    nonce: str,
    override: tuple[str, JsonValue],
) -> None:
    evidence = _controller_bound_success(7, nonce)
    evidence["controller_binding"] = _controller_binding(nonce, override)
    payload: JsonValue = {
        "checks": list(REQUIRED_CHECKS),
        "detail": "all_bounded_checks_passed",
        "evidence": evidence,
        "nonce": nonce,
        "rank": 7,
        "schema_version": 1,
        "status": "PASSED",
    }
    (output_dir / "rank-7.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_malformed_controller_rank(
    output_dir: Path,
    nonce: str,
    override: tuple[str, JsonValue],
) -> None:
    evidence = _controller_bound_success(7, nonce)
    binding = evidence["controller_binding"]
    if not isinstance(binding, dict):
        raise AssertionError("controller binding fixture must be an object")
    field, value = override
    binding[field] = value
    payload: JsonValue = {
        "checks": list(REQUIRED_CHECKS),
        "detail": "all_bounded_checks_passed",
        "evidence": evidence,
        "nonce": nonce,
        "rank": 7,
        "schema_version": 1,
        "status": "PASSED",
    }
    (output_dir / "rank-7.json").write_text(json.dumps(payload), encoding="utf-8")


def test_limits_reject_a_non_authorized_world_size() -> None:
    # Given: the only authorized eight-rank qualification shape.
    limits = QualificationLimits()

    # When / Then: a seven-rank request crosses the boundary.
    with pytest.raises(ValueError, match="expected_world_size"):
        limits.validate_world_size(7)


def test_probe_parser_returns_structured_error_for_overbudget_timeout(tmp_path: Path) -> None:
    # Given: an otherwise complete probe invocation exceeding the launcher cap.
    arguments = (
        "--output-dir", str(tmp_path / "qualification-unique"),
        "--nonce", "0" * 32,
        "--expected-world-size", "8",
        "--timeout-seconds", "1201",
    )

    # When: the CPU-only parser handles the input.
    result = parse_probe_config(arguments)

    # Then: it returns a stable non-GPU rejection, not a permissive config.
    assert result.status == "INVALID_ARGUMENT"
    assert result.detail == "timeout_seconds_exceeds_1200"


def test_deadline_budget_bounds_every_phase_and_nccl() -> None:
    # Given: the complete controller-to-aggregation qualification workflow.
    limits = QualificationLimits()

    # When: its independent phase and teardown budgets are inspected.
    phase_total = (
        limits.preflight_timeout_seconds
        + limits.runtime_timeout_seconds
        + limits.aggregation_timeout_seconds
    )

    # Then: no phase or NCCL wait can exhaust the 1650-second outer boundary.
    assert limits.workflow_timeout_seconds == 1_650
    assert limits.preflight_timeout_seconds == 300
    assert limits.runtime_timeout_seconds == 1_200
    assert limits.aggregation_timeout_seconds == 90
    assert limits.nccl_timeout_seconds == 120
    assert limits.controller_receipt_timeout_seconds == 60
    assert phase_total <= limits.workflow_timeout_seconds
    assert limits.kill_grace_seconds == 30


@pytest.mark.parametrize(
    "case",
    (
        (
            "--nccl-timeout-seconds",
            "121",
            "nccl_timeout_seconds_exceeds_120",
        ),
        (
            "--controller-receipt-timeout-seconds",
            "61",
            "controller_receipt_timeout_seconds_exceeds_60",
        ),
    ),
)
def test_probe_parser_rejects_subdeadline_overruns(
    tmp_path: Path,
    case: tuple[str, str, str],
) -> None:
    # Given: a complete invocation with one wait exceeding its independent cap.
    option, value, expected_detail = case
    arguments = (
        "--output-dir",
        str(tmp_path / "qualification-unique"),
        "--nonce",
        "0" * 32,
        "--manifest-sha256",
        "a" * 64,
        "--run-id",
        "qualification-20260912-01-deadbeef",
        "--controller-receipt",
        str(tmp_path / "controller-receipt.json"),
        option,
        value,
    )

    # When: CPU-only control parsing applies the nested deadline budget.
    result = parse_probe_config(arguments)

    # Then: an inner wait cannot consume its parent phase deadline.
    assert result.status == "INVALID_ARGUMENT"
    assert result.detail == expected_detail


def test_probe_parser_accepts_controller_receipt_path_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: controller-bound values injected as job environment variables.
    receipt_path = tmp_path / "controller-receipt.json"
    monkeypatch.setenv("QUALIFICATION_JOB_RECEIPT", str(receipt_path))
    monkeypatch.setenv(
        "QUALIFICATION_RUN_ID", "qualification-20260912-01-deadbeef"
    )
    arguments = (
        "--output-dir",
        str(tmp_path / "qualification-unique"),
        "--nonce",
        "0" * 32,
        "--manifest-sha256",
        "a" * 64,
    )

    # When: no duplicate CLI path or run id is supplied.
    result = parse_probe_config(arguments)

    # Then: the expected absolute receipt path is preserved in the ready config.
    assert result.status == "READY"
    assert result.config is not None
    assert result.config.controller_receipt == receipt_path
    assert result.config.run_id == "qualification-20260912-01-deadbeef"


def test_launcher_applies_outer_and_phase_deadlines() -> None:
    # Given: the exact shell boundary eventually executed by the authorized job.
    launcher = (
        Path(__file__).parents[3]
        / "scale/experiments/nonlatent_iclr/qualification/run_qualification.sh"
    ).read_text(encoding="utf-8")

    # When/Then: explicit constants and timeout invocations cover the whole workflow.
    assert "readonly WORKFLOW_TIMEOUT_SECONDS=1650" in launcher
    assert "readonly PREFLIGHT_TIMEOUT_SECONDS=300" in launcher
    assert "readonly RUNTIME_TIMEOUT_SECONDS=1200" in launcher
    assert "readonly AGGREGATION_TIMEOUT_SECONDS=90" in launcher
    assert "readonly NCCL_TIMEOUT_SECONDS=120" in launcher
    assert "QUALIFICATION_WORKFLOW_INNER" in launcher
    assert '"${WORKFLOW_TIMEOUT_SECONDS}s" "${BASH_SOURCE[0]}" "$@"' in launcher
    assert '"${PREFLIGHT_TIMEOUT_SECONDS}s" "${PYTHON_BIN}"' in launcher
    assert '"${RUNTIME_TIMEOUT_SECONDS}s"' in launcher
    assert '"${AGGREGATION_TIMEOUT_SECONDS}s" "${PYTHON_BIN}"' in launcher


@pytest.mark.parametrize("value", ("nan", "inf", "-inf"))
def test_probe_parser_rejects_non_finite_memory_fraction(tmp_path: Path, value: str) -> None:
    # Given: a non-finite resource fraction that bypasses ordinary comparisons.
    arguments = ("--output-dir", str(tmp_path / "qualification-unique"), "--nonce", "0" * 32, f"--memory-fraction={value}")

    # When: the CPU-only parser handles the value.
    result = parse_probe_config(arguments)

    # Then: no non-finite control reaches CUDA allocation.
    assert result.status == "INVALID_ARGUMENT"
    assert result.detail == "memory_fraction_must_be_finite"


def test_aggregate_requires_every_fresh_rank_and_matching_nonce(tmp_path: Path) -> None:
    # Given: seven fresh passing rank records and one stale nonce.
    nonce = "a" * 32
    for rank in range(7):
        _write_passing_rank(tmp_path, rank, nonce)
    _write_passing_rank(tmp_path, 7, "b" * 32)

    # When: the owner aggregator consumes the result directory.
    outcome = aggregate_results(tmp_path, nonce=nonce)

    # Then: a stale rank record is a structured failure, never a partial pass.
    assert outcome.status == "FAILED"
    assert outcome.detail == "stale_or_foreign_rank_result:7"


def test_aggregate_rejects_missing_rank_and_runtime_failure(tmp_path: Path) -> None:
    # Given: a bounded outcome set that omits rank seven and includes a runtime failure.
    nonce = "c" * 32
    for rank in range(6):
        _write_passing_rank(tmp_path, rank, nonce)
    RankResult.failed(rank=6, nonce=nonce, detail="fla_runtime_failure").write_once(tmp_path)

    # When: aggregation checks exact rank coverage before completion.
    outcome = aggregate_results(tmp_path, nonce=nonce)

    # Then: missing coverage has priority and has a non-pass terminal state.
    assert outcome.status == "FAILED"
    assert outcome.detail == "missing_rank_results:7"


def test_aggregate_rejects_malformed_and_complete_runtime_failure(tmp_path: Path) -> None:
    # Given: malformed rank zero evidence followed by a complete runtime-failure set.
    nonce = "d" * 32
    (tmp_path / "rank-0.json").write_text("{", encoding="utf-8")
    for rank in range(1, 8):
        _write_passing_rank(tmp_path, rank, nonce)

    # When / Then: malformed evidence fails closed before a valid all-rank failure is reported.
    malformed = aggregate_results(tmp_path, nonce=nonce)
    assert malformed.detail == "malformed_rank_result:0"
    (tmp_path / "rank-0.json").unlink()
    RankResult.failed(rank=0, nonce=nonce, detail="memory_ceiling_exceeded").write_once(tmp_path)
    failed = aggregate_results(tmp_path, nonce=nonce)
    assert failed.detail == "rank_runtime_failure:0:memory_ceiling_exceeded"


def test_aggregate_rejects_empty_success_evidence(tmp_path: Path) -> None:
    # Given: syntactically passing records that prove none of the named checks.
    nonce = "e" * 32
    for rank in range(8):
        payload: JsonValue = {
            "checks": list(REQUIRED_CHECKS),
            "detail": "all_bounded_checks_passed",
            "evidence": {},
            "nonce": nonce,
            "rank": rank,
            "schema_version": 1,
            "status": "PASSED",
        }
        (tmp_path / f"rank-{rank}.json").write_text(json.dumps(payload), encoding="utf-8")

    # When: owner aggregation validates the evidence schema.
    outcome = aggregate_results(tmp_path, nonce=nonce)

    # Then: empty success evidence cannot certify the qualification.
    assert outcome.status == "FAILED"
    assert outcome.detail == "malformed_rank_result:0"


def test_aggregate_rejects_duplicate_named_checks(tmp_path: Path) -> None:
    # Given: eight detailed records, one claiming a required check twice.
    nonce = "f" * 32
    for rank in range(8):
        _write_passing_rank(tmp_path, rank, nonce)
    payload: JsonValue = {
        "checks": [*REQUIRED_CHECKS, REQUIRED_CHECKS[0]],
        "detail": "all_bounded_checks_passed",
        "evidence": _success_evidence(0, nonce=nonce),
        "nonce": nonce,
        "rank": 0,
        "schema_version": 1,
        "status": "PASSED",
    }
    (tmp_path / "rank-0.json").write_text(json.dumps(payload), encoding="utf-8")

    # When: owner aggregation compares the ordered fixed check contract.
    outcome = aggregate_results(tmp_path, nonce=nonce)

    # Then: set equality cannot hide duplicated or reordered claims.
    assert outcome.status == "FAILED"
    assert outcome.detail == "malformed_rank_result:0"


def test_aggregate_rejects_non_finite_typed_evidence(tmp_path: Path) -> None:
    # Given: otherwise complete evidence with one non-finite nested measurement.
    nonce = "1" * 32
    for rank in range(1, 8):
        _write_passing_rank(tmp_path, rank, nonce)
    payload: JsonValue = {
        "checks": list(REQUIRED_CHECKS),
        "detail": "all_bounded_checks_passed",
        "evidence": _success_evidence(0, float("nan"), nonce=nonce),
        "nonce": nonce,
        "rank": 0,
        "schema_version": 1,
        "status": "PASSED",
    }
    (tmp_path / "rank-0.json").write_text(
        json.dumps(payload, allow_nan=True), encoding="utf-8"
    )

    # When: owner aggregation parses the nested timing evidence.
    outcome = aggregate_results(tmp_path, nonce=nonce)

    # Then: JSON extensions such as NaN fail closed.
    assert outcome.status == "FAILED"
    assert outcome.detail == "malformed_rank_result:0"


def test_aggregate_reports_honest_partial_scope_for_complete_core_checks(tmp_path: Path) -> None:
    # Given: exactly eight fresh, fully evidenced supported-check records.
    nonce = "2" * 32
    for rank in range(8):
        _write_passing_rank(tmp_path, rank, nonce)

    # When: owner aggregation reaches its sole successful terminal state.
    outcome = aggregate_results(tmp_path, nonce=nonce)

    # Then: core validity never becomes a whole-model cache qualification claim.
    assert outcome.status == "PARTIAL_QUALIFICATION"
    assert outcome.detail == "corevalid/fullmodelcacheunqualified"


@pytest.mark.parametrize(
    "case",
    (
        (
            "job_id",
            "job-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            True,
            "cross_rank_controller_binding_mismatch:7",
        ),
        (
            "submitted_request_sha256",
            "7" * 64,
            True,
            "cross_rank_controller_binding_mismatch:7",
        ),
        (
            "spec_id",
            "7166bd2e-6cbe-4bd9-be38-762d11003e70",
            False,
            "malformed_rank_result:7",
        ),
        (
            "scheduler_image_id",
            "image-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            False,
            "malformed_rank_result:7",
        ),
        (
            "observed_image_digest",
            "6" * 64,
            False,
            "malformed_rank_result:7",
        ),
    ),
)
def test_aggregate_requires_all_ranks_to_share_controller_resource_image_binding(
    tmp_path: Path,
    case: tuple[str, JsonValue, bool, str],
) -> None:
    # Given: seven ranks bound to one receipt and rank seven bound to foreign evidence.
    nonce = "3" * 32
    for rank in range(7):
        _write_controller_bound_rank(tmp_path, rank, nonce)
    field, foreign_value, is_valid_foreign_receipt, expected_detail = case
    if is_valid_foreign_receipt:
        _write_foreign_controller_rank(tmp_path, nonce, (field, foreign_value))
    else:
        _write_malformed_controller_rank(tmp_path, nonce, (field, foreign_value))

    # When: terminal aggregation compares controller, resource, and image identity.
    outcome = aggregate_results(tmp_path, nonce=nonce, manifest_sha256="a" * 64)

    # Then: every independently varied binding fails at the divergent rank.
    assert outcome.status == "FAILED"
    assert outcome.detail == expected_detail
