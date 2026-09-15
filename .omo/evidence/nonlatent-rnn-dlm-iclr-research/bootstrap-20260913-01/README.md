# bootstrap-20260913-01 — worker bootstrap diagnostic v2 (FROZEN, not submitted)

Campaign: nonlatent-rnn-dlm-iclr-research. Diagnoses the 5s/no-output failure of
job-e0c9f792 (classified INCONCLUSIVE_PRE_PAYLOAD_OR_LAUNCHER_START). This is a
bootstrap environment probe — NOT a model qualification and NOT a paper result.

## Frozen request (do not regenerate; guard exits 7)

- job_name: `bootstrap-diag-p1-h100-20260913T073456Z-1fe0b3fe` (identity pinned across re-freeze; no job ever submitted)
- run_id: `20260913T073456Z`, nonce: `1fe0b3fe`
- request_sha256: `7e3f86759618e7ac3f3c92a4e6c7bae7b986927ce9a232c21fc429cffe2be7a1` (sha256 of job_spec_candidate.json, v3)
- payload_sha256: `b5d3b688aee1d1a477910fcb1f64b508ca9458487de741df90cfe853be554531` (bootstrap_diag.sh, unchanged since v2 freeze)
- wrapper: command = `/usr/bin/bash -lc <shlex.quote(inner)>`; inner decodes payload to
  `/tmp/bootstrap_diag.$$.sh`, trap-removes it on EXIT, preserves payload exit status,
  echoes `BOOTSTRAP_DIAG_EXIT=$rc` (shell-explicit; no reliance on scheduler shell parsing)
- History: `archive/v1-h200-placeholder/` (H200/LCG placeholder) ·
  `archive/v2a-stale-freeze-after-payload-edit/` (QA hash check caught it) ·
  `archive/v2b-echo-unwrapped-command/` (command began `echo <b64> |` — wrapped in v3)
- Resources: `lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e`, project `project-160ccb20-…`,
  ws `ws-9dcc0e1f-…`, spec `7166bd2e-…`, image relay2:v2, **NVIDIA H100 SXM 80G** x8, 1 node
- Caps: scheduler 180000 ms, priority 4, FT false / retries 0, reserve keys ABSENT;
  in-job watchdog TERM@120s + KILL@+5s, per-probe ≤15s

## Channel separation (env-injection hypothesis)

- Command channel: payload inline (single-line base64) + ALL control values inline
  (BOOT_RUN_ID / BOOT_NONCE / BOOT_OUT_ROOTS (colon-sep) / probe paths /
  BOOT_API_SENTINEL_EXPECTED) — diagnostic is complete even if the API env channel
  delivers nothing.
- API env channel: ONLY `BOOT_API_SENTINEL`. If missing on the pod → recorded
  `present=false` → pod-level proof that scheduler env injection fails (GetJob
  `envs=[]` is NOT pod proof).

## Key observation already made (host-side, to be confirmed on the node)

- Project launcher `scale/experiments/nonlatent_iclr/qualification/run_qualification.sh`
  is `root:root 0700` and controller-receipt-04 is `root:root 0600` — a non-root pod
  gets exec/read denial within seconds. The probe records uid + x/r bits so the node
  confirms or refutes this.

## Files

- `bootstrap_diag.sh` — reviewed payload v2 (single source of truth)
- `build_spec.sh` — regenerates+froze spec; refuses rebuild (exit 7) unless FORCE_REBUILD=1
- `job_spec_candidate.json` — FROZEN request candidate; `request_frozen.json` — freeze record
- `test_local_fixture.sh` — CPU fixture + frozen-request QA (local only, NOT a node proof)

## Pre-submit (root delegates preflight, then operator submits)

1. Independent preflight verifies `request_frozen.json` hashes vs files, spec caps,
   and no duplicate live job with the same name.
2. Submit exactly: `qz train CreateJob --data "$(cat job_spec_candidate.json)" -o yaml`
   (no dry-run requirement recorded here; keep raw creation response before parsing).

## Read paths after submit

- GetJobLog: `DIAG_SUMMARY`, `PATH …` lines, `SENTINEL {…}`, `PUBLISH_*` lines
- `/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/logs/bootstrap-20260913T073456Z-1fe0b3fe/diag.json`
- `/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/bootstrap-20260913T073456Z-1fe0b3fe/diag.json`

## Interpretation map

- `project_launcher.executable=false` (or uid mismatch) → permission denial at exec (likely root cause)
- `api_sentinel.present=false` → scheduler env injection broken
- `project_payload.exists=false` → project mount unavailable on worker
- `python`/`usr_bin_python` missing or PKGS errors → runtime/dependency failure
- `PUBLISH_*_FAIL` with Permission denied → output-root permission issue (recorded, not fixed)

## Explicit unknowns

- 43-byte checkpoint metadata file: NOT located anywhere searched (scale dir depth-3,
  size-43c scan, qualification evidence). `BOOT_META_PATH` ships unset →
  `checkpoint_metadata_file: not_configured`. Configure inline before submit if found.
- Whether job pods run as root or a non-root user (probe answers this via uid).
- Whether the scheduler delivers `envs` to the pod at all (sentinel answers this).
