# Commands and exits

## Direct four-defect repros

The prior same-length mutations were rerun against production `generate_exact_task`, `public_evidence`, and `validate_exact_gold`.

```text
exact_tasks_sha256=cd29f3ab5cc155357c37686ad43651083e9860631adea1868eaa0a33eca50ef9
state_a_gold=11997661/40547627
state_b_expected_complement=28549966/40547627
a_plus_b=1
state_b_with_old_a_gold_rejected=true
state_b_with_complement_validates=true
nonbinary_same_length=true
nonbinary_rejected=true
double_block_same_length=true
double_block_rejected=true
```

An isolated copy of the four tracked implementation sources was hashed into a receipt, checked current, changed only in temporary `tasks/exact_tasks.py`, checked stale, and removed. The named production receipt was also checked directly; an existing legacy receipt without `implementation_source_hashes` was checked ineligible.

```text
named_receipt_current=true
actual_source_hashes_match_named_receipt=true
isolated_receipt_current_before_drift=true
isolated_receipt_current_after_exact_source_drift=false
legacy_receipt_has_source_hashes=false
legacy_receipt_current=false
temporary_source_copy_removed=true
```

## Owned tests

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  scale/tests/nonlatent_iclr/test_exact_tasks.py \
  scale/tests/nonlatent_iclr/test_task_registry.py
```

```text
Running 19 items in this shard
................... [100%]
19 passed in 5.50s
OWNED_TEST_EXIT=0
```

## Temporary standalone CLI

Both commands targeted a fresh `/tmp/opencode/task4-final-recheck-cli.*` output/evidence root; default assets remained the real absent project paths.

```text
CLI_STATUS=BLOCKED
CLI_BLOCKED_COUNT=5
CLI_SUCCESSFUL_CELLS=689
CLI_INFEASIBLE_CELLS=31
CLI_REGISTRY_SHA256=75c487cb77ecb58b698b6fc9c032543528fd67dc9cb395784ce6a2f099d5531d
CLI_RECEIPT_REGISTRY_HASH_MATCH=True
CLI_RECEIPT_SOURCE_HASH_COUNT=4
CLI_FAILURE_TRUE_COUNT=4
HAPPY_VERIFY_EXIT=2
FAILURE_VERIFY_EXIT=0
TEMP_CLEANUP_EXIT=0
```
