# bootstrap-20260913-01 — worker bootstrap diagnostic (candidate, NOT submitted)

Campaign: nonlatent-rnn-dlm-iclr-research. Diagnoses the <5s no-output failure of
qualification job `e0c9f792`. This is a bootstrap environment probe — NOT a model
qualification and NOT a paper result.

## Contract (source/runtime observability)

1. Payload starts WITHOUT the project mount: the job `command` carries the whole
   payload inline (single-line base64; decoded to `/tmp` on the worker and run).
2. Stdout is captured immediately (tee) so GetJobLog shows progress even on a
   fast death; watchdog TERM@120s + KILL@+5s, per-exec timeout <=15s.
3. Durable publish is best-effort into unique children
   `bootstrap-<RUN_ID>-<NONCE>/` of two KNOWN roots; write errors are recorded
   verbatim (PUBLISH_*_FAIL lines), never fixed (no chmod).
4. Secrets policy: env/config/token VALUES are never printed or written — only
   key presence + approved-value match booleans.
5. Checks: identity (uid/gid/host/time), executable paths, python3/python -VV,
   importlib.metadata versions of torch/fla/pydantic (no package import, no CUDA),
   stat/readable/executable for project payload path + staged model dir +
   optional metadata-file/receipt (unset -> not_configured), approved
   qualification env keys, nvidia-smi names/counts (only if binary exists;
   CPU-side driver query, no compute).

## Files

- `bootstrap_diag.sh`   — reviewed payload (single source of truth)
- `build_spec.sh`       — regenerates `job_spec_candidate.json`, inlining the payload
- `job_spec_candidate.json` — spec candidate, placeholder `logic_compute_group_id`
  (`lcg-71b971a7-PENDING-...`) makes it non-submit-able by construction
- `test_local_fixture.sh` — CPU fixture + spec round-trip QA (local only, not a node proof)

## Before submit (by orchestrator)

1. `POOL_ID=lcg-<full-uuid-of-H200-pool> bash build_spec.sh`  (known fragment: `lcg-71b971a7-...`)
2. Optional: add `BOOT_META_PATH` env once the checkpoint metadata FILENAME is
   confirmed (metadata only, never weights). Leave receipt unset.
3. `qz train CreateJob --dry-run --data "$(cat job_spec_candidate.json)"` then submit.

## Caps baked into the spec

1 node x 8 H200 (spec_id 7166bd2e-...), image relay2:v2, priority 4,
`max_running_time_ms: 180000` (3 min), `auto_fault_tolerance: false`,
`fault_tolerance_max_retry: 0`, reserve_* omitted, project `project-160ccb20-...`,
workspace `ws-9dcc0e1f-...`.

## Where to read results after submit

- GetJobLog (full stdout incl. `DIAG_SUMMARY` and `PUBLISH_*` lines)
- `/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/logs/bootstrap-<RUN_ID>-<NONCE>/diag.json`
- `/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/bootstrap-<RUN_ID>-<NONCE>/diag.json`
  (RUN_ID/NONCE = the values baked into the spec `envs`; also in the job name)

Interpretation: `project_payload` MISSING -> project source path unmounted on
worker (hypothesis 1); qualification_env all absent under a spec that injects
them -> env injection broken (hypothesis 2); python/python3 missing, PKGS errors,
or PUBLISH_*_FAIL with permission errors -> runtime/permission failure (hypothesis 3).
