from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
import os
from typing import Final

import pytest
from pydantic import TypeAdapter

from scale.experiments.nonlatent_iclr.architecture_contract import (
    SOURCE_FILES, contract_json, contract_payload,
)
from scale.experiments.nonlatent_iclr.architecture_checks import (
    ContractViolation, validate_contract_json,
)
from scale.experiments.nonlatent_iclr.architecture_task import prepare, verify
from scale.experiments.nonlatent_iclr.architecture_evidence_models import RuntimeQualification

ROOT: Final = Path(__file__).parents[3]
CAMPAIGN: Final = Path('.omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260913-05')
WORKER: Final = Path(os.environ.get(
    "NONLATENT_WORKER_ROOT",
    Path(__file__).resolve().parents[3] / "external/nonlatent_iclr_qualification")) / "qualification-20260912-01-91f32832-f583-4638-aedc-be4ac9887ab3"


@pytest.fixture
def isolated(tmp_path: Path) -> Path:
    for relative in SOURCE_FILES.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_bounded_counts_when_accepted_evidence_is_present() -> None:
    # Given / When: the actual accepted campaign, not fabricated runtime fixtures.
    payload = contract_payload(ROOT)
    # Then: counts are runtime-backed while architecture readiness stays blocked.
    runtime = RuntimeQualification.model_validate_json(json.dumps(payload['runtime_evidence']))
    assert runtime.checkpoint_step == 4750
    assert runtime.checkpoint_state_tensors == 1955
    assert runtime.parameter_numel == 4091581441
    assert runtime.ranks == tuple(range(8))
    assert payload['readiness'] == {'blocked_claims': 4, 'whole_architecture_ready': False}
    assert runtime.masked_logits_shape == (1, 48, 65536)
    assert runtime.masked_logits_all_finite is True
    assert runtime.standalone_fla_cache_max_abs_delta == 0.0
    assert runtime.observed_image_digest is None
    assert runtime.long_context_claim is False
    assert runtime.performance_or_goodput_claim is False


def test_cpu_only_when_isolated_repository_has_no_campaign(isolated: Path) -> None:
    # Given / When: source-only isolated repositories retain engineering controls.
    payload = contract_payload(isolated)
    # Then: no runtime claim is inferred from external files alone.
    assert payload['runtime_evidence'] is None
    claims = TypeAdapter(dict[str, str | dict[str, int]]).validate_python(payload['claims'])
    assert claims['full_checkpoint_count'] == 'unavailable'


@pytest.mark.parametrize('relative', [
    'runtime-review-superseding-01/verdict.json',
    'submission/monitoring-v6/worker-output-hashes.sha256',
    'submission/monitoring-v6/evidence-sha256.txt',
])
@pytest.mark.parametrize('missing', [False, True])
def test_rejects_campaign_damage_when_evidence_is_bound(
    isolated: Path, relative: str, missing: bool,
) -> None:
    # Given: exact real campaign copies; only the isolated copy is damaged.
    for directory in ('runtime-review-superseding-01', 'submission/monitoring-v6'):
        _ = shutil.copytree(ROOT / CAMPAIGN / directory, isolated / CAMPAIGN / directory)
    damaged = isolated / CAMPAIGN / relative
    if missing:
        damaged.unlink()
    else:
        _ = damaged.write_bytes(damaged.read_bytes() + b' ')
    # When / Then: publication fails closed, never silently downgrades to CPU mode.
    assert prepare(isolated).status in {'MALFORMED', 'SOURCE_UNAVAILABLE'}


@pytest.mark.parametrize('mutation', [
    (WORKER / 'rank-0.json', b'"rank": 0', b'"rank": 1'),
    (WORKER / 'rank-0.json', b'91f32832-f583', b'91f32832-f584'),
    (WORKER / 'rank-0.json', b'ffaa464dabb3291c', b'aaaa464dabb3291c'),
    (WORKER / 'rank-0.json', b'a41af12290a62f', b'b41af12290a62f'),
    (WORKER / 'rank-0.json', b'9271a2036e84c3', b'8271a2036e84c3'),
    (WORKER / 'rank-0.json', b'55df1dfa714dd5', b'65df1dfa714dd5'),
    (WORKER.parent / 'controller-receipts' / (WORKER.name + '.json'), b'"schema_version":1', b'"schema_version":2'),
])
def test_rejects_altered_runtime_bytes_when_validating(
    monkeypatch: pytest.MonkeyPatch, mutation: tuple[Path, bytes, bytes],
) -> None:
    # Given: mutate only reads, never historical evidence on disk.
    canonical = contract_json(ROOT)
    original = Path.read_bytes
    target, before, after = mutation
    assert before in original(target)

    def altered(path: Path) -> bytes:
        data = original(path)
        return data.replace(before, after) if path == target else data

    monkeypatch.setattr(Path, 'read_bytes', altered)
    # When / Then: exact digests prevent rank/run/checkpoint/receipt substitution.
    with pytest.raises(ContractViolation):
        validate_contract_json(canonical, ROOT)
    assert verify(ROOT).status in {'MALFORMED', 'SOURCE_UNAVAILABLE'}


@pytest.mark.parametrize(('before', 'after'), [
    ('"whole_architecture_ready": false', '"whole_architecture_ready": true'),
    ('NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED', 'QUALIFIED'),
    ('NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES', 'QUALIFIED'),
])
def test_rejects_promotion_when_bounded_claims_are_edited(before: str, after: str) -> None:
    # Given: the canonical bounded claims.
    canonical = contract_json(ROOT)
    assert before in canonical
    # When / Then: neither whole architecture nor either complete cache can promote.
    with pytest.raises(ContractViolation):
        validate_contract_json(canonical.replace(before, after), ROOT)


@pytest.mark.parametrize('case', ['prepare', 'happy', 'failure'])
def test_cli_when_canonical_campaign_is_available(case: str) -> None:
    # Given / When: the owned Task3 CLI uses the real publication service.
    result = subprocess.run(
        [sys.executable, '-m', 'scale.experiments.nonlatent_iclr.architecture_task',
         '--case', case, '--repo-root', str(ROOT)],
        capture_output=True, text=True, check=False,
    )
    # Then: each supported publication/verification path succeeds.
    assert result.returncode == 0, result.stderr


def test_cli_rejects_when_campaign_is_incomplete(isolated: Path) -> None:
    # Given: a campaign directory without its required immutable evidence.
    (isolated / CAMPAIGN).mkdir(parents=True)
    # When: the publication CLI attempts ingestion.
    result = subprocess.run(
        [sys.executable, '-m', 'scale.experiments.nonlatent_iclr.architecture_task',
         '--case', 'prepare', '--repo-root', str(isolated)],
        capture_output=True, text=True, check=False,
    )
    # Then: rejection exits nonzero instead of publishing a CPU fallback.
    assert result.returncode == 1
    assert not (isolated / 'DAN/nonlatent_iclr/architecture_contract.json').exists()


@pytest.mark.parametrize('target', [
    WORKER / 'rank-7.json',
    ROOT / CAMPAIGN / 'runtime-review-superseding-01/evidence-sha256.txt',
    ROOT / CAMPAIGN / 'submission/exact_job_spec.json',
])
def test_validation_rejects_when_bound_record_disappears(
    monkeypatch: pytest.MonkeyPatch, target: Path,
) -> None:
    # Given: a valid contract followed by an unavailable evidence record.
    canonical = contract_json(ROOT)
    original = Path.read_bytes

    def missing(path: Path) -> bytes:
        if path == target:
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', missing)
    # When / Then: freshness validation fails closed on absence.
    with pytest.raises(ContractViolation, match='runtime_evidence_unavailable'):
        validate_contract_json(canonical, ROOT)


# The artifacts these tests exercise are not part of the repository: they are
# large, rights-gated, and produced on a cluster. A clone must SKIP rather than
# fail, so the suite reports honestly what it can verify without them. Point
# NONLATENT_WORKER_ROOT at a prepared root to run them.
pytestmark = pytest.mark.skipif(
    not WORKER.exists(),
    reason=f"external artifact absent: {WORKER} -- set the matching "
           f"NONLATENT_* environment variable to a prepared root")
