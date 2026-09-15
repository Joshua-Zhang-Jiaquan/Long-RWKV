# Startup investigation — job-e0c9f792 (attempt 04, job_failed @ 5 s)

Date: 2026-09-13. Read-only lane, 3 of <=4 qz API calls (schema GetJobLog; GetJob; GetJobLog). Local repo-file read for launcher logic (no edits). No CreateJob/dry-run/StopJob/credentials/broad listing.

## API evidence
- `qz schema train.GetJobLog` → documented params: `job_id` (string), `page_num`/`page_size` (int32, optional), `filter`, `sorter`, `search_after`; response `total` + `logs[]`
- `qz train GetJob {"job_id":"job-e0c9f792-8afb-4902-bb24-5dd8872927e1"}` → exit 0:
  - name `qualification-20260912-01-0ff9241e-460a-4647-95b3-af4c0fba16ce`, status `job_failed`
  - running_time_ms=`5000` (exact 5 s); max_running_time_ms=`1800000` correctly stored
  - timeline (epoch ms): created 1789280986000 → resource_prepared 1789281030000 → run 1789281033000 → finished 1789281038000 (≈47 s scheduling, 5 s runtime)
  - resource tuple as authorized: instance_count=1, gpu_count=8, image relay2:v2, LCG 开发区-H100-cuda13.2版本-183核; task_priority=0
  - **`envs` key present but EMPTY array (length 0)** — server preserved zero environment variables
  - **command** = `/usr/bin/bash <repo>/scale/experiments/nonlatent_iclr/qualification/run_qualification.sh` (179 chars, single script call, **zero** QUALIFICATION_*/NONCE/MANIFEST/RECEIPT tokens)
  - No faults/instance-exit/log-location fields exposed in GetJob response shape (same schema as reference job)
- `qz train GetJobLog {"job_id":"...","page_num":0,"page_size":100}` (documented shape) → `InternalError: internal server error`. **API path stopped per instruction**; pod stderr remains unobservable this pass.

## Local launcher logic (scale/experiments/nonlatent_iclr/qualification/run_qualification.sh, read-only)
- `set -Eeuo pipefail`; fail-closed guard sequence BEFORE any output dir is created:
  - line 44-46: `QUALIFICATION_RUN_ID is required` → exit 2 if unset
  - lines 48-67: RUN_ID format, NONCE 32-hex, MANIFEST_SHA256 64-hex, JOB_RECEIPT absolute-path guards → exit 2
  - only AFTER guards (lines 82-95): `mkdir` of SHARED_OUTPUT_ROOT/RUN_ID output dir, manifest snapshot, receipts

## Concrete failure determination (evidence-backed, not speculation)
**The pod executed with none of the required QUALIFICATION_* variables** (server-persisted envs=[], command carries no exports). The launcher's first guard (RUN_ID, line 44) fails → `exit 2` within the 5-second runtime window, before any model load or shared-output-dir creation — consistent with the observed "no exact shared output dir". All four guard inputs were absent; the FIRST failure deterministically is the RUN_ID check.
- Where the loss happened (controller sent envs that the platform dropped vs envs never attached to request 04) is NOT distinguishable from read APIs: GetJob stores what it stores; no submit-receipt endpoint exists. Flag: earlier failed attempt 03 also carried these envs in its request design; platform storage behavior for envs on this LCG/pool is only observed as "empty" for this job (prior reference job job-83690070 also stored no envs, while a different historical job class does store envs — i.e., empty storage here is not proof of stripping).
- Residual unknown: pod-side user/permissions on the mode-700 root-owned script; primary explanation (guard exit) requires only that bash read the script, which the 5 s runtime is consistent with; an immediate permission failure cannot be fully excluded without pod logs (GetJobLog InternalError).

## Resource/recycle observations
- GPU burn actually observed: 8 GPUs x 5 s = 40 GPU-seconds (~0.011 GPU-h) — measured from stored timeline, not invented.
- No faults field, no retry consumed (auto_fault_tolerance path untouched; running round n/a). Job is terminal `job_failed`; nothing to stop.
- GetJobLog path: valid documented schema exists (job_id/page_num/page_size) but currently returns InternalError for this job — retry belongs to a later pass, not this one.

## Fix direction (for the controller worker; no action taken here)
Re-attach the four QUALIFICATION_* variables via the job `envs` array (or inline exports in command, reference-job style) AND verify they persist; consider a trivial pre-submission self-check that the created job's GetJob echoes non-empty envs before model stages are allowed.
