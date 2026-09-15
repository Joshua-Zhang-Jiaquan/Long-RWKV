from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from scale.experiments.nonlatent_iclr.architecture_contract import contract_payload
from scale.experiments.nonlatent_iclr.architecture_task import prepare

ROOT: Final = Path(__file__).parents[3]


def test_cpu_claims_when_source_bound_units_are_available() -> None:
    # Given / When: the real CPU unit sources, not GPU measurements.
    payload = contract_payload(ROOT)
    claims = TypeAdapter(dict[str, JsonValue]).validate_python(payload['cpu_source_evidence'])
    # Then: clocks and initialization are CPU scoped; active trainer remains open.
    assert claims['scope'] == 'source_hash_bound_cpu_tests_not_gpu_measurements'
    assert claims['active_model_optimizer_membership'] is False
    assert claims['active_model_mask_loss_semantics'] is False
    assert claims['base_block_calls'] == 32
    assert claims['recycled_block_calls'] == 16


@pytest.mark.parametrize('name', ['trainer_semantics.py', 'initialization_loop.py', 'test_initialization_loop.py'])
def test_rejects_cpu_source_drift_when_publishing(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    # Given: damage only reads of the real approved source/test unit.
    target = ROOT / 'scale/experiments/nonlatent_iclr' / name
    original = Path.read_bytes

    def altered(path: Path) -> bytes:
        data = original(path)
        return data + b' ' if path == target else data

    monkeypatch.setattr(Path, 'read_bytes', altered)
    # When / Then: no stale CPU claims may be published.
    assert prepare(ROOT).status == 'MALFORMED'


@pytest.mark.parametrize('missing', [True, False])
def test_rejects_cpu_receipt_damage_when_publishing(monkeypatch: pytest.MonkeyPatch, missing: bool) -> None:
    from scale.experiments.nonlatent_iclr.architecture_cpu_evidence import CPU_RECEIPT

    original = Path.read_bytes

    def damaged(path: Path) -> bytes:
        if path == ROOT / CPU_RECEIPT:
            if missing:
                raise FileNotFoundError(path)
            return original(path) + b' '
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', damaged)
    assert prepare(ROOT).status in {'MALFORMED', 'SOURCE_UNAVAILABLE'}
