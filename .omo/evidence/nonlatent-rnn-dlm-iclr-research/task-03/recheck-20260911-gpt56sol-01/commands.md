# Task-3 repaired CPU-subset recheck commands

All Python commands used `CUDA_VISIBLE_DEVICES=""` where Torch was involved and `PYTHONDONTWRITEBYTECODE=1`. Pytest disabled its cache provider. No command used qz, GPU execution, network, installation, git, or checkpoint loading.

## Freshness preflight

```text
python -c '<call architecture_task.verify(Path(".")); require BLOCKED_EXTERNAL, ENGINEERING_ONLY, blocked_claims > 0>'
exit: 0
{"blocked_claims": 6, "detail": "ENGINEERING_ONLY CPU controls are fresh; whole architecture remains blocked", "status": "BLOCKED_EXTERNAL"}
```

This read-only preflight ensured the owned test's `prepare(REPO_ROOT)` call would take its already-fresh no-write branch.

## Eight owned tests

```text
python -m pytest scale/tests/nonlatent_iclr/test_architecture_contract.py -q -p no:cacheprovider
exit: 0
8 passed in 9.93s
```

## Isolated adversarial copy

The probe copied all five `SOURCE_FILES` into a temporary root below this evidence directory, generated a matching contract/receipt, mutated one condition at a time, and removed the root in `finally`.

```text
exit: 0
baseline=BLOCKED_EXTERNAL|blocked=6|ENGINEERING_ONLY CPU controls are fresh; whole architecture remains blocked
stale_version=MALFORMED|blocked=6|canonical_contract_mismatch
wrong_source_hash=MALFORMED|blocked=6|canonical_contract_mismatch
false_runtime_verified=MALFORMED|blocked=6|canonical_contract_mismatch
deleted_blocker=MALFORMED|blocked=6|canonical_contract_mismatch
source_drift=MALFORMED|blocked=6|canonical_contract_mismatch
wrong_receipt_contract_hash=MALFORMED|blocked=6|receipt_mismatch
wrong_receipt_source_hash=MALFORMED|blocked=6|receipt_mismatch
cleanup=removed:adversarial-tyksgu76
```

## Public task functions

```text
exit: 0
prepare=BLOCKED_EXTERNAL|blocked=6|ENGINEERING_ONLY CPU controls already published; external runtime proof remains blocked
verify=BLOCKED_EXTERNAL|blocked=6|ENGINEERING_ONLY CPU controls are fresh; whole architecture remains blocked
analyze=BLOCKED_EXTERNAL|blocked=6|ENGINEERING_ONLY; <three helper paths>; <two checkpoint paths>; <unexecuted FLA runtime>
```

`prepare` used the fresh-publication branch; no product artifact was archived, replaced, or created.

## Owned standalone module interface

```text
python -m scale.experiments.nonlatent_iclr.architecture_task --case happy --repo-root .
exit: 0
{"status":"BLOCKED_EXTERNAL","detail":"ENGINEERING_ONLY CPU controls are fresh; whole architecture remains blocked","blocked_claims":6}

python -m scale.experiments.nonlatent_iclr.architecture_task --case failure --repo-root .
exit: 0
{"status":"EXPECTED_FAILURE_CONFIRMED","detail":"canonical_contract_mismatch"}
```

The failure case is an adversarial QA result, not a readiness result; it intentionally has no blocker-count field. Root CLI integration is not part of this partial interface.

## Readiness consistency

The first shell rendering attempt had a quoting-only Python `SyntaxError` and exited 1 before reading or writing anything. The corrected command exited 0:

```text
contract_status=BLOCKED_EXTERNAL|cpu=ENGINEERING_ONLY|blocked=6|ready=False
receipt_status=BLOCKED_EXTERNAL|cpu=ENGINEERING_ONLY|blocked=6
```

## Current binding and no-GPU classification

```text
exit: 0
canonical_contract=True
canonical_receipt=True
receipt_contract_hash=True
receipt_source_manifest_hash=True
all_staged_sources_match=True
all_fla_sources_match=True
fla_availability=installed_source_available|version=0.5.0|sources=3
cache_runtime=unavailable|state_owner=engineering_control_only|masks=unavailable
fla_imported=False
cuda_initialized=False
```

## Snapshot loop seam

The explicit in-memory `TypedTorchModule = torch.nn.Module` shim ran the unchanged snapshot test, which uses the real model class body but `_FakeBlock` for every RWKV block:

```text
exit: 0
BACKBONE-LOOP SUITE: ALL PASS
```

Classification: test-only fake / engineering control. It is not an FLA kernel, checkpoint, optimizer, caller-state, or cache-equivalence execution.

## LSP

Pyright diagnostics on `architecture_contract.py`, `architecture_checks.py`, `architecture_task.py`, and `test_architecture_contract.py`: no diagnostics, all four.
