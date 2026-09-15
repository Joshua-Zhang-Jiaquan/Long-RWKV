# Adapter-v6 compliant CPU QA commands

The successful attempt used cache prefix `/tmp/adapter-v6-qa-cache.84kajX`. Every Python invocation had:

```text
PYTHONDONTWRITEBYTECODE=1
PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX
PYTHONSAFEPATH=1
/usr/bin/python -P
```

Durations were measured around each command with shell `date +%s%N`; no Python timing helper was used.

## Candidate and release tests

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX PYTHONSAFEPATH=1 QUALIFICATION_RELEASE_UNDER_TEST="$V6" PYTHONPATH="$V6" /usr/bin/python -P -m pytest -q "$STAGING/test_adapter_candidate_v6.py"
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX PYTHONSAFEPATH=1 QUALIFICATION_RELEASE_UNDER_TEST="$V6" PYTHONPATH="$V6" /usr/bin/python -P -m pytest -q "$STAGING/test_adapter_release_v6.py"
```

Results: 5 passed in 18.54s, exit 0; 3 passed in 0.81s, exit 0.

## Approved regression scope

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX PYTHONSAFEPATH=1 QUALIFICATION_RELEASE_UNDER_TEST="$V6" PYTHONPATH="$V6" /usr/bin/python -P -m pytest -q "$ROOT/scale/tests/nonlatent_iclr/test_full_canvas.py" "$ROOT/scale/tests/nonlatent_iclr/test_full_canvas_ownership.py" "$ROOT/scale/tests/nonlatent_iclr/test_qualification_masked_forward.py" "$ROOT/scale/tests/nonlatent_iclr/test_qualification_parameter_evidence.py"
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX PYTHONSAFEPATH=1 QUALIFICATION_RELEASE_UNDER_TEST="$V6" PYTHONPATH="$V6" /usr/bin/python -P -m pytest -q "$ROOT/scale/tests/nonlatent_iclr/test_qualification_manifest.py" "$ROOT/scale/tests/nonlatent_iclr/test_architecture_contract.py"
```

Results: 55 passed in 4.56s, exit 0; 11 passed in 14.14s, exit 0. The scopes are disjoint: 66/66.

## Inherited controls

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX PYTHONSAFEPATH=1 QUALIFICATION_RELEASE_UNDER_TEST="$V5" PYTHONPATH="$V5" /usr/bin/python -P -m pytest -q "$V5_EVIDENCE/test_bytecode_isolation_v5.py" "$V5_EVIDENCE/test_release_freshness_v5.py"
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX PYTHONSAFEPATH=1 QUALIFICATION_RELEASE_UNDER_TEST="$V6" PYTHONPATH="$V6" /usr/bin/python -P -m pytest -q "$V4_EVIDENCE/test_origin_enforcement_v4.py::test_owned_root_is_promoted_when_already_later_in_sys_path" "$V4_EVIDENCE/test_origin_enforcement_v4.py::test_foreign_preloaded_models_package_is_rejected_before_construction" "$V4_EVIDENCE/test_origin_enforcement_v4.py::test_foreign_preloaded_models_submodule_is_rejected_before_construction" "$V4_EVIDENCE/test_origin_enforcement_v4.py::test_owned_preloaded_model_origins_are_accepted" "$V4_EVIDENCE/test_origin_enforcement_v4.py::test_import_rejects_loaded_model_helper_missing_from_manifest" "$V4_EVIDENCE/test_origin_enforcement_v4.py::test_wrong_original_cwd_resolves_owned_models_with_launcher_environment"
```

Results: 11 passed in 2.05s, exit 0; 6 passed in 1.82s, exit 0. Total inherited controls: 17/17.

## Strict preflight and environment

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX PYTHONSAFEPATH=1 PYTHONPATH="$V6" /usr/bin/python -P -m scale.experiments.nonlatent_iclr.qualification.cli validate --output-dir "$TMP/out" --nonce eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee --run-id qualification-20260912-01-adapter-v6-cpu-review-20260913 --controller-receipt "$TMP/controller-unused.json" --controller-receipt-timeout-seconds 60 --expected-world-size 8 --timeout-seconds 1200 --nccl-timeout-seconds 120 --memory-fraction 0.90 --staged-root "$V6/model_source/scale" --checkpoint-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/m2_baseline_triangle/m4loop_endpoint_ckpt --model-dir /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF --manifest "$V6/scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json" --manifest-sha256 692d1cc88d6b2c6335736d47492aa933bf57b9a23a4d0dcc9e5e3eeda165cf32 --preflight-receipt "$TMP/preflight.json"
env -C "$V6" PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/adapter-v6-qa-cache.84kajX PYTHONSAFEPATH=1 PYTHONPATH="$V6" /usr/bin/python -P "$V6/scale/experiments/nonlatent_iclr/qualification/runtime_environment.py" --manifest "$V6/scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json" --output "$TMP/runtime_environment.json" --release-root "$V6" --staged-root "$V6/model_source/scale" --checkpoint-root /inspire/hdd/global_user/zhangjiaquan-253108540222/m2_baseline_triangle/m4loop_endpoint_ckpt --model-root /inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF
```

Results: `PREFLIGHT_VERIFIED`, 111 files, exit 0; environment `MATCH`, 8/8 package sources, 6/6 versions, no runtime model modules, exit 0.

## Shell-only identity and cleanup checks

The deterministic tree command used GNU tar with sorted names, epoch mtime, numeric root ownership, deleted pax atime/ctime, and SHA-256. Before and after tree SHA-256 was `5b86080fba3c7ed11c65c9b7045349f0ad074973921ff04c22347ac4db8690f9`; manifest SHA-256 was `692d1cc88d6b2c6335736d47492aa933bf57b9a23a4d0dcc9e5e3eeda165cf32`.

Shell path scans returned zero writable paths, symlinks, and bytecode paths before and after. The external cache contained zero entries before removal and was absent afterward; the temporary output directory was also removed.
