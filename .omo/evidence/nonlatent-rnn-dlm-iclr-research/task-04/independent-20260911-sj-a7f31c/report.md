# Task 04 independent CPU-subset verification

## Verdict

**NEEDS FIX; keep Task 4 BLOCKED.** The four sampled exact answers are correct, current generated evidence is public, the declared matrix/splits are deterministic, and the eight owned tests pass. The partial registry is not fail-closed: five structurally fake manifests with empty provenance values make `verify_registry` and standalone happy verification report `READY`/exit 0. Boundary and load-axis defects also remain.

## Correctness controls

- Authoritative owned-test run from repository root: 8 collected, 8 passed, exit 0. An earlier launch from `scale/` was a harness-path error (0 collected, import failure, exit 2), not a product result.
- Matrix arithmetic is exact: `4 * 5 * 3 * 4 * 3 = 720` unique lazy cells; 200 instances and five seeds declare 720,000 condition requests if 200 applies per seed/cell. There are 4,000 source-family identities and 180 condition views per identity. The implementation does not enforce the declared seed or index bounds.
- Across 4,000 family IDs, deterministic family-first assignment produced train/validation/confirmation = 3212/422/366. Generated instance-0 groups each had 180 views and zero split violations. Two byte-distinct paraphrase controls sharing `source_family=repo-family-7` both mapped to train. No real paraphrase axis is implemented despite the test description.
- Associative sample: public `k6=v821;QUERY k6` gives `v821`, matching encoded gold.
- Overwrite sample: public `k6=v9521`, then `k6=v1207`, then final `k6=v5264;QUERY k6` gives `v5264`, matching encoded gold.
- HMM sample `11111111`: independent posterior path is `1/2, 13/35, 41/127, 713/2335, 2581/8627, 47413/159635, 175121/591127, 3240113/10946935, 11997661/40547627`; final fraction matches encoded gold.
- Dataflow sample computes `3+5+7+2+4+7+6+8=42`, matching encoded gold.
- All none/random/similar variants changed prompt hashes for each of four families and preserved gold. These are implemented prompt mutations, not labels only, although `similar` always uses key/value text regardless of family.
- Across 3,600 normal cell/seed instance-0 requests, every `source_bytes` string occurred in `public_prompt`; every prompt had exactly the declared **character** count. Public serialization keys excluded private answer fields. Registry/condition labels explicitly say `logical_character_fixture`, not RWKV tokens.

## Defects requiring repair before accepting the engineering subset

1. `external_assets._inspect` checks only key presence. Empty `version`, `license`, and `sha256` values are accepted; hashes are not bound to bytes; suites need no records; repository `count=200` needs no 200 records/rights/families/hashes; tokenizer needs no implementation/procedure; evaluator needs only the string `external_vm`. Five such fake files yield `ready=true`, status `READY`, and happy exit 0.
2. `validate_exact_gold` recomputes from evaluator-held `source_bytes`, not from `public_prompt`. A prompt with all evidence removed still validates true. Current generated prompts are visible, but the readiness check does not prove the no-hidden-input property.
3. Tiny declared lengths 0/1/16 do not fail: 12 probes returned 155-223 characters. For seed 101/instance 0, 26 of 720 registered cells cannot place the complete evidence/distractor block at the requested 10%/50% location; generation silently clamps prefix padding instead of rejecting an impossible request.
4. HMM and dataflow do not implement the declared load endpoints: requested load 1 becomes 2 and 128 becomes 32; load-32 and load-128 source hashes/gold are identical for the no-distractor controls. Associative honors 1/8/32/128; overwrite emits load distractor writes plus two target writes.
5. Registry limits are declarations only: seed 999 and instance index 200 are accepted. There is no sealed iterator/validator proving exactly the declared 200 instances and five seeds.
6. Standalone failure exit aggregation is correct (normal four-true fixture exits 0; forcing one predicate false exits 2), but its four rejection helpers are hard-coded comparisons rather than calls through production registry/manifest validation. It therefore does not falsify defect 1 or 2.

## Current external blockers (all absent at the bound registry paths)

1. `DAN/nonlatent_iclr/task4_assets/ruler/development_manifest.json`: genuine pinned RULER development suite, version, nonempty license, source records, and hashes bound to local bytes.
2. `DAN/nonlatent_iclr/task4_assets/longbench/development_manifest.json`: genuine pinned LongBench-family development suite with the same provenance/license binding.
3. `DAN/nonlatent_iclr/task4_assets/repository_tasks/manifest.json`: 200 rights-cleared executable task records, repository family/time split lineage, per-source hashes, and licenses.
4. `DAN/nonlatent_iclr/task4_assets/repository_evaluator/isolation_manifest.json`: a real externally managed VM/container evaluator identity and provenance; this QA did not execute it.
5. `DAN/nonlatent_iclr/task4_assets/tokenizer/rwkv_tokenizer_manifest.json`: local RWKV tokenizer identity plus an executable, hash-bound token-length qualification procedure/results.

## CLI exits and cleanup

- Missing-assets standalone prepare: exit 2; happy verify: exit 2; status BLOCKED with five blockers.
- Planted-failure verify: exit 0 with four reported rejections; forced one-false control: exit 2.
- Empty manifests: exit 2. Bare `count=200` / `isolation=true`: exit 2. Empty-string key-only fake manifests: API exit 0, happy CLI exit 0, status READY (fail-open defect).
- Every `/tmp/opencode/task4-independent-*` probe tree was removed (`TEMP_CLEANUP_EXIT=0`). No network, container, qz, GPU, installation, git, untrusted evaluator/code execution, production edit, plan edit, or Boulder edit occurred. Existing registry artifacts were never command output targets.
