# Qualification 04 runtime result

- Job: `job-e0c9f792-8afb-4902-bb24-5dd8872927e1`
- Scheduler terminal status: `job_failed`
- Qualification classification: `FAILED_NO_RUNTIME_PROOF`
- Failure cause: `INCONCLUSIVE_PRE_PAYLOAD_OR_LAUNCHER_START`

## Timing and resource accounting

- Created: `2026-09-13T06:29:46Z`
- Resource prepared: `2026-09-13T06:30:30Z`
- Running: `2026-09-13T06:30:33Z`
- Finished: `2026-09-13T06:30:38Z`
- Scheduler running time: `5000 ms`
- Shape: 1 node, 8 × `NVIDIA_H100_SXM_80G`
- Observed GPU-hours: `8 * 5000 / 3600000 = 0.011111111111111112`

The GPU-hour value uses the scheduler's `running_time_ms`; it is not a billing claim. The broader resource-prepared-to-finished window was 8 seconds (`0.017777777777777778` GPU-hours), also not a billing claim.

## Runtime proof

The configured image, H100 shape, pool/spec, 1,800,000 ms cap, disabled fault tolerance, and zero retries matched. The GPFS run root did not exist more than 20 minutes after scheduler finish. Consequently no runtime manifest snapshot, preflight, launcher, aggregate, or rank records were available; zero of eight required nonce/controller/manifest-bound rank records were observed and none of the seven core checks can be credited.

`GetJobLog` was attempted once because GPFS evidence was absent. It returned `InternalError: internal server error` with no usable log lines. Therefore the scheduler failure is proven, but the exact payload or infrastructure failure trace is unavailable.

The healthy target `PARTIAL_QUALIFICATION (corevalid/fullmodelcacheunqualified)` was not reached. Full-model cache behavior was not evaluated.

## Safety and uncertainty

- No retry, new submission, or StopJob occurred.
- No resource mismatch or over-cap runtime was observed.
- Retention fields and resource-recycle fields were absent from terminal GetJob; retention and post-terminal resource release remain unknown. No zero-retention inference was made, and no visible terminal resource hold was reported.

Primary evidence: `status-snapshots/001-20260913T065112Z.json`, `terminal-getjob.json`, `gpfs-artifact-inventory.json`, `getjoblog.stdout.redacted`, `check-results.json`, and `resource-accounting.json`.
