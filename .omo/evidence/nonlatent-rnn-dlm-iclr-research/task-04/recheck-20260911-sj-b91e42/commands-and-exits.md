# Focused command receipts

## Owned tests

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  scale/tests/nonlatent_iclr/test_exact_tasks.py \
  scale/tests/nonlatent_iclr/test_task_registry.py
```

```text
Running 16 items in this shard
................ [100%]
16 passed in 5.44s
OWNED_TEST_EXIT=0
```

## Representative matrix

One direct loop constructed only seed 101/index 0 for each of the 720 lazy declarations, caught `InfeasibleTaskError`, and applied production `validate_exact_gold` plus exact length/position checks.

```text
success=689
infeasible=31
gold_failures=0
length_failures=0
position_failures=0
materialized_instances=0
representative_scan_seconds=0.294687
full_verify_seconds=1.105048
length_8192_cells=144
length_8192_success=134
length_8192_infeasible=10
length_8192_scan_seconds=0.032997
```

## Standalone CLI

The production module ran with the real missing asset root and fresh `/tmp/opencode/task4-recheck-cli.*` output/evidence paths.

```text
CLI_STATUS=BLOCKED
CLI_BLOCKED_COUNT=5
CLI_SUCCESSFUL_CELLS=689
CLI_INFEASIBLE_CELLS=31
CLI_REGISTRY_SHA256=75c487cb77ecb58b698b6fc9c032543528fd67dc9cb395784ce6a2f099d5531d
CLI_RECEIPT_COUNT=2
CLI_RECEIPT_HASH_MATCHES=2
CLI_FAILURE_TRUE_COUNT=4
PREPARE_EXIT=2
HAPPY_VERIFY_EXIT=2
FAILURE_VERIFY_EXIT=0
TEMP_CLEANUP_EXIT=0
```

## Manifest and staleness controls

- A temporary structurally complete fake set contained hash-bound RULER/LongBench bytes, exactly 200 unique repository records with bound source bytes, bound isolation evidence, and bound tokenizer results. All five statuses remained false with `no trusted qualification adapter`, and the tree was removed.
- The current registry was regenerated in memory from current source and compared byte-for-byte. Current and regenerated hashes matched. Seven historical receipts were parsed: three bind the current hash, four lack `registry_sha256`, and none bind implementation source hashes.
- `source-hashes.sha256` binds every reviewed source/test plus current registry, registry Markdown, and archived prior registry.
