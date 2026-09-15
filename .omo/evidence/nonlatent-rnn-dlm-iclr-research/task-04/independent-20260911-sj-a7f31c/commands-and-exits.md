# Commands and exits

All commands ran from the repository root unless noted. `PYTHONDONTWRITEBYTECODE=1` and pytest's disabled cache provider prevented local bytecode/test-cache artifacts.

## Owned tests

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  scale/tests/nonlatent_iclr/test_exact_tasks.py \
  scale/tests/nonlatent_iclr/test_task_registry.py
```

```text
Running 8 items in this shard
........ [100%]
8 passed in 1.03s
OWNED_TEST_EXIT=0
```

Non-authoritative setup attempt from `scale/`: 0 tests ran; both modules failed collection with `ModuleNotFoundError: scale`; exit 2. The command was corrected by using the repository-root import context.

## Standalone CLI with absent assets

Each command used fresh `mktemp -d /tmp/opencode/task4-independent-cli.XXXXXX` paths for `--asset-root`, `--output-root`, and `--evidence-root`; the tree was removed afterward.

```text
CLI_STATUS=BLOCKED
CLI_BLOCKED_COUNT=5
CLI_EVIDENCE_ATTEMPTS=2
CLI_REJECTIONS={"answer_field_rejected": true, "fork_leakage_rejected": true, "silent_truncation_rejected": true, "test_tampering_rejected": true}
PREPARE_EXIT=2
HAPPY_VERIFY_EXIT=2
FAILURE_VERIFY_EXIT=0
TEMP_CLEANUP_EXIT=0
```

After monkeypatching one in-memory rejection predicate to return false, failure verification returned 2. Its temporary output tree was removed.

```text
PLANTED_REJECTION_FALSE_EXIT=2
FAILURE_EXIT_GUARD_PROBE=0
TEMP_CLEANUP_EXIT=0
```

## Manifest authorization probes

The probe created only JSON under a fresh `/tmp/opencode/task4-independent-assets.XXXXXX`, called `verify_registry`/standalone happy verification, and removed the tree. It never executed an evaluator or referenced a container runtime.

```text
empty: ready=false, blocked_count=5, exit_code=2
bare count=200 and isolation=true: ready=false, blocked_count=5, exit_code=2
fake empty version/license/sha256 keys plus isolation="external_vm": ready=true, blocked_count=0, exit_code=0
FAKE_HAPPY_VERIFY_EXIT=0
FAKE_REGISTRY_STATUS=READY
FAKE_BLOCKED_COUNT=0
MANIFEST_CREATE_EXIT=0
ASSET_PROBE_EXIT=0
TEMP_CLEANUP_EXIT=0
```

## Numeric probes

- Generated and directly inspected four fixed public-evidence examples; independent arithmetic is captured in `controls.json`.
- Enumerated the 720 registry cells over five seeds at instance 0 (3,600 prompts) for source visibility and exact character length.
- Enumerated all `4 * 5 * 200 = 4,000` family identities for split counts.
- Compared all load values and all three distractor prompt hashes for every exact family.
- Generated 12 tiny requests (four families times declared lengths 0/1/16), and tested a public-prompt-redaction mutation plus undeclared seed/index requests.
- Bound reviewed inputs and preserved registry outputs with `sha256sum`; see `source-hashes.sha256`.
