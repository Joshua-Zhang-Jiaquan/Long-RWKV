from __future__ import annotations

from pathlib import Path
import os
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from scale.experiments.nonlatent_iclr.architecture_checks import ContractViolation, validate_contract_json
from scale.experiments.nonlatent_iclr.architecture_contract import contract_json, contract_payload
from scale.experiments.nonlatent_iclr.architecture_task import prepare

ROOT: Final = Path(__file__).parents[3]
CAMPAIGN: Final = ROOT / '.omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260914-08-live'
WORKER: Final = Path(os.environ.get(
    "NONLATENT_WORKER_ROOT",
    Path(__file__).resolve().parents[3] / "external/nonlatent_iclr_qualification")) / "qualification-20260912-01-3257f249-eace-4a74-841f-0d7d8e0e32e2"


def test_v8_closes_scoped_acceptance_when_actual_evidence_is_available() -> None:
    # Given / When: canonical ingestion of actual reviewed v8 records.
    payload = contract_payload(ROOT)
    evidence = TypeAdapter(dict[str, JsonValue]).validate_python(payload['semantics_v8_evidence'])
    acceptance = TypeAdapter(dict[str, str]).validate_python(payload['task3_acceptance'])
    # Then: both empirical gaps close, without whole-architecture promotion.
    assert evidence['review_sha256'] == '4996f460a0ac6befe2f42e6a17b15a555d938f65d352faebe6534087b923b6f0'
    assert evidence['ranks'] == list(range(8))
    assert acceptance['active_optimizer_membership_and_gradient_policy'] == 'satisfied_loaded_model_two_freeze_configs_stage_a_only'
    assert acceptance['active_input_output_mask_loss_semantics'] == 'satisfied_tested_corruption_rank_local_actual_logit_loss_backward'
    assert payload['readiness'] == {'blocked_claims': 4, 'whole_architecture_ready': False}


@pytest.mark.parametrize('target', [
    CAMPAIGN / 'runtime-review/verdict.json',
    CAMPAIGN / 'runtime-review/evidence-sha256.txt',
    CAMPAIGN / 'worker-artifacts.sha256',
    WORKER / 'rank-7.json',
    WORKER / 'lifecycle-7.json',
    WORKER / 'model-semantics-7.json',
])
@pytest.mark.parametrize('missing', [True, False])
def test_v8_damage_rejects_publication(monkeypatch: pytest.MonkeyPatch, target: Path, missing: bool) -> None:
    # Given: change reads only; immutable evidence stays untouched.
    original = Path.read_bytes

    def damaged(path: Path) -> bytes:
        if path == target:
            if missing:
                raise FileNotFoundError(path)
            return original(path) + b' '
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', damaged)
    # When / Then: neither omission nor tampering allows publication.
    assert prepare(ROOT).status in {'MALFORMED', 'SOURCE_UNAVAILABLE'}


@pytest.mark.parametrize(('before', 'after'), [
    (b'job-cd3c6850', b'job-dd3c6850'),
    (b'3257f249-eace', b'3257f249-eacf'),
    (b'a9750fd783cc98d7', b'b9750fd783cc98d7'),
    (b'bae0fa5a71aca185', b'cae0fa5a71aca185'),
    (b'312d37aff1b89318', b'412d37aff1b89318'),
    (b'ffaa464dabb3291c', b'aaaa464dabb3291c'),
])
def test_wrong_v8_binding_rejects_validation(monkeypatch: pytest.MonkeyPatch, before: bytes, after: bytes) -> None:
    # Given: valid contract followed by a changed real semantics sidecar read.
    canonical = contract_json(ROOT)
    original = Path.read_bytes
    target = WORKER / 'model-semantics-0.json'
    assert before in original(target)

    def damaged(path: Path) -> bytes:
        data = original(path)
        return data.replace(before, after) if path == target else data

    monkeypatch.setattr(Path, 'read_bytes', damaged)
    # When / Then: exact-hash identity binding rejects substitution.
    with pytest.raises(ContractViolation):
        validate_contract_json(canonical, ROOT)


@pytest.mark.parametrize(('before', 'after'), [
    ('"whole_architecture_ready": false', '"whole_architecture_ready": true'),
    ('NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED', 'QUALIFIED'),
    ('NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES', 'QUALIFIED'),
])
def test_v8_does_not_allow_cache_or_whole_architecture_promotion(before: str, after: str) -> None:
    # Given: the canonical bounded publication.
    canonical = contract_json(ROOT)
    assert before in canonical
    # When / Then: v8 cannot authorize broader promotion by editing claims.
    with pytest.raises(ContractViolation):
        validate_contract_json(canonical.replace(before, after), ROOT)


@pytest.mark.parametrize(('before', 'after'), [
    ('"each_trainable_once":true', '"each_trainable_once":false'),
    ('"eligibility_exact":true', '"eligibility_exact":false'),
    ('"corruption_exact":true', '"corruption_exact":false'),
    ('"empty_loss":0.0', '"empty_loss":1.0'),
    ('"unselected_logit_grad_zero":true', '"unselected_logit_grad_zero":false'),
    ('"unchanged_when_ungated":true', '"unchanged_when_ungated":false'),
    ('"before":"none","after":"none"', '"before":"none","after":"zero"'),
])
def test_reparsed_semantics_failure_rejects_independently_of_hash_guard(before: str, after: str) -> None:
    from scale.experiments.nonlatent_iclr.architecture_evidence_io import EvidenceError
    from scale.experiments.nonlatent_iclr.architecture_semantics import read_semantics_records, validate_semantics_records

    # Given: hash-verified real records with one schema-valid observation damaged.
    records = read_semantics_records(ROOT)
    sidecar = records.semantics[0]
    data = sidecar.model_dump_json()
    assert before in data
    altered = type(sidecar).model_validate_json(data.replace(before, after))
    damaged = records.model_copy(update={'semantics': (altered, *records.semantics[1:])})
    # When / Then: the derived predicate independently rejects the record.
    with pytest.raises(EvidenceError, match='semantics_derived_predicate_failed'):
        _ = validate_semantics_records(damaged)


@pytest.mark.parametrize('field', ['ranks', 'lifecycle', 'semantics'])
@pytest.mark.parametrize('reorder', [True, False])
def test_v8_missing_or_reordered_inventory_rejects_after_parsing(field: str, reorder: bool) -> None:
    from scale.experiments.nonlatent_iclr.architecture_evidence_io import EvidenceError
    from scale.experiments.nonlatent_iclr.architecture_semantics import read_semantics_records, validate_semantics_records
    from scale.experiments.nonlatent_iclr.architecture_semantics_models import SemanticsRecords

    # Given: actual records, with only one inventory shortened or reordered.
    records = read_semantics_records(ROOT)
    data = TypeAdapter(dict[str, JsonValue]).validate_json(records.model_dump_json())
    entries = TypeAdapter(list[JsonValue]).validate_python(data[field])
    data[field] = list(reversed(entries)) if reorder else entries[:-1]
    import json
    damaged = SemanticsRecords.model_validate_json(json.dumps(data))
    # When / Then: exact rank coverage and ordering are independently enforced.
    with pytest.raises(EvidenceError, match='semantics_rank_inventory_or_order_mismatch'):
        _ = validate_semantics_records(damaged)


def test_v8_lifecycle_failure_rejects_after_parsing() -> None:
    from scale.experiments.nonlatent_iclr.architecture_evidence_io import EvidenceError
    from scale.experiments.nonlatent_iclr.architecture_semantics import read_semantics_records, validate_semantics_records

    # Given: a real lifecycle sidecar with an unsuccessful comparison.
    records = read_semantics_records(ROOT)
    sidecar = records.lifecycle[0]
    data = sidecar.model_dump_json()
    assert '"exact_equal":true' in data
    altered = type(sidecar).model_validate_json(data.replace('"exact_equal":true', '"exact_equal":false'))
    damaged = records.model_copy(update={'lifecycle': (altered, *records.lifecycle[1:])})
    # When / Then: valid semantics cannot compensate for failed lifecycle evidence.
    with pytest.raises(EvidenceError, match='semantics_lifecycle_predicate_failed'):
        _ = validate_semantics_records(damaged)


# The artifacts these tests exercise are not part of the repository: they are
# large, rights-gated, and produced on a cluster. A clone must SKIP rather than
# fail, so the suite reports honestly what it can verify without them. Point
# NONLATENT_WORKER_ROOT at a prepared root to run them.
pytestmark = pytest.mark.skipif(
    not WORKER.exists(),
    reason=f"external artifact absent: {WORKER} -- set the matching "
           f"NONLATENT_* environment variable to a prepared root")
