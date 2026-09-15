from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scale.experiments.nonlatent_iclr.architecture_checks import (
    CanvasStateOwner,
    ContractViolation,
    TinyArchitectureHarness,
    validate_contract_json,
    validate_receipt_json,
)
from scale.experiments.nonlatent_iclr.architecture_contract import (
    blocked_claim_count,
    contract_payload,
    contract_json,
    receipt_json,
)
from scale.experiments.nonlatent_iclr.architecture_task import analyze, prepare, verify


REPO_ROOT = Path(__file__).parents[3]


def test_contract_rejects_changed_source_hash() -> None:
    # Given: a canonical contract whose model identity was made stale.
    stale = contract_json(REPO_ROOT).replace('"sha256": "', '"sha256": "0', 1)

    # When / Then: source identity validation rejects it rather than trusting markers.
    with pytest.raises(ContractViolation, match="canonical_contract_mismatch"):
        validate_contract_json(stale, REPO_ROOT)


def test_contract_rejects_wrong_version_and_block_count() -> None:
    # Given: independently malformed schema version and readiness count claims.
    canonical = contract_json(REPO_ROOT)
    wrong_version = canonical.replace('"contract_version": 2', '"contract_version": 1')
    blocked_count = blocked_claim_count(REPO_ROOT)
    wrong_count = canonical.replace(f'"blocked_claims": {blocked_count}', '"blocked_claims": 0')

    # When / Then: neither can be promoted to a current architecture contract.
    with pytest.raises(ContractViolation, match="canonical_contract_mismatch"):
        validate_contract_json(wrong_version, REPO_ROOT)
    with pytest.raises(ContractViolation, match="canonical_contract_mismatch"):
        validate_contract_json(wrong_count, REPO_ROOT)


def test_contract_rejects_false_cached_runtime_claim_and_removed_blocker() -> None:
    # Given: a contract that upgrades source inspection and deletes a needed asset.
    false_runtime = contract_json(REPO_ROOT).replace('"cache_runtime": "unavailable"', '"cache_runtime": "runtime_verified"')
    decoded = json.loads(contract_json(REPO_ROOT))
    decoded["blockers"] = decoded["blockers"][1:]
    removed_blocker = json.dumps(decoded, indent=2, sort_keys=True)

    # When / Then: exact source-derived availability and blockers are enforced.
    with pytest.raises(ContractViolation, match="canonical_contract_mismatch"):
        validate_contract_json(false_runtime, REPO_ROOT)
    with pytest.raises(ContractViolation, match="canonical_contract_mismatch"):
        validate_contract_json(removed_blocker, REPO_ROOT)


def test_receipt_rejects_partial_and_stale_contract_identity() -> None:
    # Given: a receipt for current bytes, then a receipt missing its required source binding.
    contract = contract_json(REPO_ROOT)
    partial = receipt_json(REPO_ROOT, contract).replace('"source_manifest_sha256": ', '"removed_source_manifest_sha256": ')
    stale = receipt_json(REPO_ROOT, contract).replace('"contract_sha256": "', '"contract_sha256": "0', 1)

    # When / Then: neither receipt validates against fresh sources and contract bytes.
    with pytest.raises(ContractViolation, match="receipt_schema"):
        validate_receipt_json(partial, contract, REPO_ROOT)
    with pytest.raises(ContractViolation, match="receipt_mismatch"):
        validate_receipt_json(stale, contract, REPO_ROOT)


def test_public_task_outputs_are_blocked_with_nonzero_readiness() -> None:
    # Given: the real staged source and a fresh task-3 publication.
    prepared = prepare(REPO_ROOT)
    verified = verify(REPO_ROOT)
    analyzed = analyze(REPO_ROOT)

    # When / Then: CPU controls never describe the whole architecture as complete.
    for result in (prepared, verified, analyzed):
        assert result.status == "BLOCKED_EXTERNAL"
        assert result.blocked_claims > 0
        assert "ENGINEERING_ONLY" in result.detail


def test_tiny_harness_is_an_engineering_control_not_fla_cache_proof() -> None:
    # Given: a CPU torch test harness with copied but separate parameters.
    torch.manual_seed(7)
    harness = TinyArchitectureHarness(width=4, layers=2)

    # When: repeated calls reuse its exact module instances.
    paired = harness.pair_at(0)
    output = harness(torch.ones(1, 4), outer_nfe=3)

    # Then: identity and loop-count controls are real torch checks, not cache proof.
    assert paired.attn_fwd is not paired.attn_bwd
    assert paired.attn_fwd.weight is not paired.attn_bwd.weight
    assert torch.equal(paired.attn_fwd.weight, paired.attn_bwd.weight)
    assert harness.block_passes == 18
    assert output.shape == (1, 4)
    assert '"cache_runtime": "unavailable"' in contract_json(REPO_ROOT)


def test_snapshot_loop_runs_with_explicit_test_only_fake_block_loader() -> None:
    # Given: the snapshot loop suite and its missing type-only module seam.
    train_root = REPO_ROOT / "DAN/v7_arch_round/code/train"
    runner = """
import runpy
import sys
import types
import torch
from pathlib import Path
train_root = Path(sys.argv[1])
code_root = train_root.parent
models = types.ModuleType("models")
models.__path__ = [str(code_root / "models")]
sys.modules["models"] = models
typed = types.ModuleType("models.state_hijacking_dit_torch_types")
typed.TypedTorchModule = torch.nn.Module
sys.modules["models.state_hijacking_dit_torch_types"] = typed
sys.path.insert(0, str(train_root))
runpy.run_path(str(train_root / "test_backbone_loop.py"), run_name="__main__")
"""

    # When: its original class-body loop executes through the explicit fake loader.
    completed = subprocess.run([sys.executable, "-c", runner, str(train_root)], capture_output=True, text=True, check=False)

    # Then: the Python loop seam passes, still without claiming FLA kernel proof.
    assert completed.returncode == 0, completed.stderr


def test_state_owner_rejects_cross_session_stale_canvas_and_prefix() -> None:
    # Given: a valid prefix from session A at revision zero.
    owner = CanvasStateOwner()
    owner.capture(session_id="session-a", canvas_revision=0, prefix_end=2, value=(1, 2))

    # When / Then: unrelated ownership or any invalidation boundary is rejected.
    with pytest.raises(ContractViolation, match="session_mismatch"):
        owner.carry(session_id="session-b", canvas_revision=0, prefix_end=2)
    with pytest.raises(ContractViolation, match="prefix_mismatch"):
        owner.carry(session_id="session-a", canvas_revision=0, prefix_end=3)
    owner.edit_canvas()
    with pytest.raises(ContractViolation, match="stale_canvas"):
        owner.carry(session_id="session-a", canvas_revision=0, prefix_end=2)
