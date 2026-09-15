from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import cast

import pytest

from scale.experiments.nonlatent_iclr import schema


REQUIRED_TUPLES = (
    ("R1", "README.md", "document"),
    ("R1", "ARCHITECTURE.md", "document"),
    ("R1", "TRAINING_HISTORY.md", "document"),
    ("R1", "report.md", "document"),
    ("R2", "results/gate_analysis_s4750_20260910.txt", "raw_record"),
    ("R2", "results/ability_curve_m4loop.csv", "raw_record"),
    ("R2", "results/ability_curve_m4mhc.csv", "raw_record"),
    ("R3", "scripts/wl_gate_analysis.py", "code"),
    ("R4", "code/eval/sampler_eval_offline.py", "code"),
    ("R5", "code/models/birwkv7_diffusion.py", "code"),
    ("R5", "code/models/residual_streams.py", "code"),
    ("R5", "code/train/test_backbone_loop.py", "code"),
    ("R6", "code/eval/lm_eval.py", "code"),
    ("R6", "code/train/train_birwkv_diffusion.py", "code"),
    ("R6", "specs/m4_loop_32gpu_half.json", "manifest"),
    ("R6", "JOBS.md", "document"),
)
PANEL_PATHS = (
    "cap_lm_m4loop_s4750",
    "cap_lm_n2_s4000",
    "cap_lm_n2_s6000",
    "cap_lm_m4loop_s4750_reps2",
    "cap_lm_m4loop_s4750_reps3",
    "cap_lm_m4loop_s4750_reps4",
    "sampler_gate_m4_loop_s4750",
    "sampler_gate_m4_n2_s4000",
    "sampler_gate_m4_n2_s6000",
    "sampler_gate_m4_loop_s4750_reps2",
    "sampler_gate_m4_loop_s4750_reps3",
    "sampler_gate_m4_loop_s4750_reps4",
    "m2_baseline_triangle/ability_curve_m4loop.csv",
)
METADATA_PATHS = (
    "outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/meta.json",
    "outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00006000_probe_copy/meta.json",
    "models/RWKV7-Goose-World3-2.9B-HF/tokenizer_config.json",
)
GATE_RECORD = "results/gate_analysis_s4750_20260910.txt"
LM1B_RECORD = "cap_lm_m4loop_s4750/lm1b/merged/merged_m4loop_endpoint_ckpt_nll-ppl-bits.json"
WIKITEXT_RECORD = "cap_lm_m4loop_s4750/wikitext103/merged/merged_m4loop_endpoint_ckpt_nll-ppl-bits.json"
SAMPLER_RECORD = "sampler_gate_m4_loop_s4750/merged/merged_m4loop_endpoint_ckpt_em-tau-residue-max_run_frac-distinct_frac.json"
LOOP_METADATA = "outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/meta.json"
ABILITY_RECORD = "m2_baseline_triangle/ability_curve_m4loop.csv"
CLAIM_BINDINGS = (
    ("lm1b_loss_ppl", "R3", (LM1B_RECORD, GATE_RECORD), ("record_count", "records[].metrics.nll", "records[].metrics.ppl")),
    ("wikitext103_loss_ppl", "R3", (WIKITEXT_RECORD, GATE_RECORD), ("record_count", "records[].metrics.nll", "records[].metrics.ppl")),
    ("causal_path_comparison", "R3", (LM1B_RECORD, WIKITEXT_RECORD), ("records[].arm", "records[].document_id", "records[].metrics.nll")),
    ("capacity_4_98b", "R6", (LOOP_METADATA, ABILITY_RECORD), ("step", "tokens_seen", "csv.tokens_seen", "csv.score")),
    ("max_run_frac_direction", "R3", (SAMPLER_RECORD, GATE_RECORD), ("records[].arm", "records[].metrics.max_run_frac", "delta", "ci95")),
    ("masked_token_accuracy", "R3", (SAMPLER_RECORD, GATE_RECORD), ("records[].arm", "records[].metrics.em", "delta", "ci95")),
    ("tau_commit_order", "R3", (SAMPLER_RECORD, GATE_RECORD), ("records[].arm", "records[].metrics.tau", "delta", "ci95")),
    ("r100_fully_masked", "R3", (SAMPLER_RECORD, GATE_RECORD), ("records[].arm", "records[].metrics", "delta", "ci95")),
)


def _artifact(source: str, path: str, kind: str) -> dict[str, object]:
    return {
        "source": source,
        "origin": "snapshot",
        "path": path,
        "kind": kind,
        "identity": "sha256",
        "sha256": "a" * 64,
        "bytes": 1,
        "file_count": 1,
        "reason": None,
    }


def _external_artifact(source: str, path: str, kind: str) -> dict[str, object]:
    artifact = _artifact(source, path, kind)
    artifact["origin"] = "external"
    return artifact


def valid_ledger() -> dict[str, object]:
    artifacts = [_artifact(*item) for item in REQUIRED_TUPLES]
    artifacts.extend(_external_artifact("R3", path, "raw_record") for path in PANEL_PATHS)
    artifacts.extend(_external_artifact("R6", path, "metadata") for path in METADATA_PATHS)
    return {
        "schema_version": 2,
        "attempt": "a" * 32,
        "source_roots": {
            "snapshot": "/audit/snapshot",
            "external": "/audit/external",
            "live": "/audit/live",
            "staged": "/audit/staged",
        },
        "artifacts": artifacts,
        "claims": [
            {
                "claim_id": claim_id,
                "source": source,
                "state": "claimed",
                "detail": f"historical assertion {claim_id}; interpretation requires task2",
                "records": list(records),
                "fields": list(fields),
            }
            for claim_id, source, records, fields in CLAIM_BINDINGS
        ],
        "views": [
            {"source": "snapshot", "state": "observed", "detail": "snapshot root present"},
            {"source": "live", "state": "unresolved", "detail": "live root absent"},
            {"source": "staged", "state": "unresolved", "detail": "staged root absent"},
        ],
        "accounting_note": "Packed capacity is never counted as unique tokens.",
    }


def test_versioned_machine_contract_is_accepted() -> None:
    # Given: exactly 32 artifacts and eight claims with stable machine IDs.
    ledger = valid_ledger()
    # When: the versioned ledger crosses the parser boundary.
    error = schema.validate(ledger)
    # Then: the complete contract is accepted.
    assert error is None


def test_complete_closed_contract_is_accepted() -> None:
    assert schema.validate(valid_ledger()) is None


@pytest.mark.parametrize("missing_path", (PANEL_PATHS[0], METADATA_PATHS[0]))
def test_external_artifact_cannot_disappear_from_complete_contract(missing_path: str) -> None:
    # Given: all 32 required artifact tuples are represented.
    ledger = valid_ledger()
    artifacts = cast(list[dict[str, object]], ledger["artifacts"])
    # When: one required external tuple disappears.
    ledger["artifacts"] = [item for item in artifacts if item["path"] != missing_path]
    # Then: the boundary rejects incomplete coverage.
    assert schema.validate(ledger) == "invalid required inventory"


def test_extra_external_artifact_is_rejected() -> None:
    # Given: a complete task-01 ledger.
    ledger = valid_ledger()
    # When: an unversioned external tuple is appended.
    cast(list[object], ledger["artifacts"]).append(
        _external_artifact("R3", "unexpected-panel", "raw_record")
    )
    # Then: the task-01 contract rejects the extra tuple.
    assert schema.validate(ledger) == "invalid required inventory"


def test_wrong_external_artifact_tuple_is_rejected() -> None:
    # Given: a complete task-01 ledger.
    ledger = valid_ledger()
    artifact = next(
        item for item in cast(list[dict[str, object]], ledger["artifacts"])
        if item["path"] == PANEL_PATHS[0]
    )
    # When: a required panel is relabeled as metadata.
    artifact["kind"] = "metadata"
    # Then: tuple identity mismatch is rejected.
    assert schema.validate(ledger) == "invalid required inventory"


def test_required_claims_cannot_be_replaced_by_one_generic_claim() -> None:
    # Given: a complete task-01 ledger.
    ledger = valid_ledger()
    # When: all historical bindings are replaced by one shape-valid claim.
    ledger["claims"] = [cast(list[dict[str, object]], ledger["claims"])[0]]
    # Then: claim coverage is rejected.
    assert schema.validate(ledger) == "invalid claim coverage"


def test_lm1b_claim_binding_cannot_disappear() -> None:
    # Given: all eight historical bindings.
    ledger = valid_ledger()
    # When: the LM1B binding disappears.
    ledger["claims"] = cast(list[dict[str, object]], ledger["claims"])[1:]
    # Then: exact claim coverage is rejected.
    assert schema.validate(ledger) == "invalid claim coverage"


def test_claim_classification_cannot_bypass_task_two_block() -> None:
    # Given: all eight historical bindings are blocked.
    ledger = valid_ledger()
    # When: one claim is fabricated as observed.
    cast(list[dict[str, object]], ledger["claims"])[0]["state"] = "observed"
    # Then: task-01 cannot report the fabricated classification as valid.
    assert schema.validate(ledger) == "invalid claim classification"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    (
        (lambda value: value["artifacts"][0].__setitem__("source", "R6"), "invalid required inventory"),
        (lambda value: value["artifacts"].append(copy.deepcopy(value["artifacts"][0])), "duplicate artifact"),
        (lambda value: value.__setitem__("claims", []), "missing claims"),
        (lambda value: value.__setitem__("views", []), "invalid views coverage"),
        (lambda value: value["views"][0].__setitem__("source", "live"), "invalid views coverage"),
        (lambda value: value["source_roots"].pop("staged"), "invalid source roots"),
    ),
)
def test_semantic_coverage_is_exact(
    mutate: Callable[[dict[str, object]], None], expected: str
) -> None:
    ledger = valid_ledger()
    mutate(ledger)
    assert schema.validate(ledger) == expected


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__("schema_version", True),
        lambda value: value.__setitem__("attempt", 1),
        lambda value: value["source_roots"].__setitem__("external", "relative/root"),
        lambda value: value["artifacts"][0].__setitem__("bytes", True),
        lambda value: value["artifacts"][0].__setitem__("file_count", True),
        lambda value: value["claims"][0].__setitem__("fields", "loss"),
        lambda value: value["views"][0].__setitem__("state", "unknown"),
    ),
)
def test_scalar_and_collection_types_are_strict(
    mutate: Callable[[dict[str, object]], None]
) -> None:
    ledger = valid_ledger()
    mutate(ledger)
    assert schema.validate(ledger) is not None


def test_duplicate_json_object_keys_are_rejected() -> None:
    payload = json.dumps(valid_ledger()).replace(
        '"schema_version": 2,', '"schema_version": 2, "schema_version": 2,', 1
    )
    with pytest.raises(ValueError, match="duplicate key"):
        schema.load_json_object(payload)


def test_receipt_contract_has_exact_fields_and_types() -> None:
    receipt: dict[str, object] = {
        "schema_version": 1,
        "attempt": "a" * 32,
        "ledger_sha256": "b" * 64,
        "status": "AUDIT_COMPLETE",
    }
    assert schema.validate_receipt(receipt, expected_attempt="a" * 32) is None
    receipt["status"] = "PREPARING"
    assert schema.validate_receipt(receipt, expected_attempt="a" * 32) == "invalid receipt status"
    receipt["status"] = "AUDIT_COMPLETE"
    receipt["extra"] = False
    assert schema.validate_receipt(receipt, expected_attempt="a" * 32) == "invalid receipt envelope"
