from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from scale.experiments.nonlatent_iclr.architecture_checks import ContractViolation, validate_contract_json
from scale.experiments.nonlatent_iclr.architecture_contract import contract_json, contract_payload
from scale.experiments.nonlatent_iclr.architecture_task import prepare

ROOT: Final = Path(__file__).parents[3]
CAMPAIGN: Final = ROOT / '.omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260914-07-live'
WORKER: Final = Path(os.environ.get(
    "NONLATENT_WORKER_ROOT",
    Path(__file__).resolve().parents[3] / "external/nonlatent_iclr_qualification")) / "qualification-20260912-01-462d8c9d-eb9b-41fd-9992-d1d44528bc3a"


def test_v7_observations_when_real_review_is_ingested() -> None:
    # Given / When: only the actual accepted review and worker records.
    payload = contract_payload(ROOT)
    # Then: independently identifiable v7 evidence closes bounded lifecycle only.
    lifecycle = TypeAdapter(dict[str, JsonValue]).validate_python(payload['lifecycle_v7_evidence'])
    assert lifecycle['review_sha256'] == '64a1a457191561e0576afa80aa84caff1b80210585cb74e24533dcc427325bd2'
    assert lifecycle['ranks'] == list(range(8))
    assert lifecycle['model_calls_per_rank'] == 12
    assert lifecycle['synchronizations_per_rank'] == 12
    assert payload['readiness'] == {'blocked_claims': 4, 'whole_architecture_ready': False}


@pytest.mark.parametrize('target', [
    CAMPAIGN / 'runtime-review/verdict.json',
    CAMPAIGN / 'runtime-review/evidence-sha256.txt',
    CAMPAIGN / 'worker-artifacts.sha256',
    WORKER / 'rank-7.json',
    WORKER / 'lifecycle-7.json',
])
@pytest.mark.parametrize('missing', [True, False])
def test_rejects_v7_damage_when_publishing(
    monkeypatch: pytest.MonkeyPatch, target: Path, missing: bool,
) -> None:
    # Given: damage reads only; accepted evidence is never edited.
    original = Path.read_bytes

    def damaged(path: Path) -> bytes:
        if path == target:
            if missing:
                raise FileNotFoundError(path)
            return original(path) + b' '
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', damaged)
    # When / Then: publication rejects instead of ignoring v7 or downgrading.
    assert prepare(ROOT).status in {'MALFORMED', 'SOURCE_UNAVAILABLE'}


@pytest.mark.parametrize(('before', 'after'), [
    (b'job-5b99b0c6', b'job-6b99b0c6'),
    (b'462d8c9d-eb9b', b'462d8c9d-eb9c'),
    (b'859969188464fd48', b'959969188464fd48'),
    (b'a38e0558a721ab9a', b'b38e0558a721ab9a'),
    (b'ef1d88b0219d4470', b'ff1d88b0219d4470'),
    (b'ffaa464dabb3291c', b'aaaa464dabb3291c'),
    (b'4091581441', b'4091581442'),
])
def test_rejects_wrong_v7_identity_when_validating(
    monkeypatch: pytest.MonkeyPatch, before: bytes, after: bytes,
) -> None:
    # Given: a valid contract, then a mutated real rank read.
    canonical = contract_json(ROOT)
    original = Path.read_bytes
    target = WORKER / 'rank-0.json'
    assert before in original(target)

    def damaged(path: Path) -> bytes:
        data = original(path)
        return data.replace(before, after) if path == target else data

    monkeypatch.setattr(Path, 'read_bytes', damaged)
    # When / Then: no identity substitution is promotable.
    with pytest.raises(ContractViolation):
        validate_contract_json(canonical, ROOT)


def test_rejects_rank_reordering_when_validating(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: valid real records with only the aggregate ordering reversed on read.
    canonical = contract_json(ROOT)
    original = Path.read_bytes

    def reordered(path: Path) -> bytes:
        data = original(path)
        if path == WORKER / 'aggregate.json':
            aggregate = TypeAdapter(dict[str, JsonValue]).validate_json(data)
            ranks = TypeAdapter(list[JsonValue]).validate_python(aggregate['rank_results'])
            aggregate['rank_results'] = list(reversed(ranks))
            return json.dumps(aggregate).encode()
        return data

    monkeypatch.setattr(Path, 'read_bytes', reordered)
    # When / Then: aggregate order is part of the immutable evidence contract.
    with pytest.raises(ContractViolation):
        validate_contract_json(canonical, ROOT)


@pytest.mark.parametrize(('before', 'after'), [
    ('"exact_equal":true', '"exact_equal":false'),
    ('"synchronizations":12', '"synchronizations":11'),
    ('"model_calls":12', '"model_calls":11'),
    ('"completed":true', '"completed":false'),
])
def test_derived_predicate_rejects_when_reparsed_observation_fails(before: str, after: str) -> None:
    from scale.experiments.nonlatent_iclr.architecture_evidence import EvidenceError
    from scale.experiments.nonlatent_iclr.architecture_lifecycle import (
        read_lifecycle_records, validate_lifecycle_records,
    )

    # Given: real hash-verified records, then a schema-valid semantic failure.
    original = read_lifecycle_records(ROOT)
    sidecar = original.sidecars[0]
    data = sidecar.model_dump_json()
    assert before in data
    altered = type(sidecar).model_validate_json(data.replace(before, after))
    records = original.model_copy(update={'sidecars': (altered, *original.sidecars[1:])})
    # When / Then: a separate semantic guard rejects even without the hash guard.
    with pytest.raises(EvidenceError, match='lifecycle_derived_predicate_failed'):
        validate_lifecycle_records(records)


@pytest.mark.parametrize('field', ['ranks', 'sidecars'])
@pytest.mark.parametrize('reorder', [True, False])
def test_rejects_semantic_inventory_damage_when_hashes_already_checked(field: str, reorder: bool) -> None:
    from scale.experiments.nonlatent_iclr.architecture_evidence import EvidenceError
    from scale.experiments.nonlatent_iclr.architecture_lifecycle import (
        read_lifecycle_records, validate_lifecycle_records,
    )

    records = read_lifecycle_records(ROOT)
    entries = records.ranks if field == 'ranks' else records.sidecars
    altered = tuple(reversed(entries)) if reorder else entries[:-1]
    with pytest.raises(EvidenceError, match='lifecycle_rank_inventory_or_order_mismatch'):
        validate_lifecycle_records(records.model_copy(update={field: altered}))


# The artifacts these tests exercise are not part of the repository: they are
# large, rights-gated, and produced on a cluster. A clone must SKIP rather than
# fail, so the suite reports honestly what it can verify without them. Point
# NONLATENT_WORKER_ROOT at a prepared root to run them.
pytestmark = pytest.mark.skipif(
    not WORKER.exists(),
    reason=f"external artifact absent: {WORKER} -- set the matching "
           f"NONLATENT_* environment variable to a prepared root")
