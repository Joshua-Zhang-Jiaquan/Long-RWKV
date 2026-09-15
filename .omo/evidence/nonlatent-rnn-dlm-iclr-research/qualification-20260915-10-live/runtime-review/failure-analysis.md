# Calibration job v10: outcome analysis (author analysis, NOT an independent review)

Job `job-84988b46-749e-4d04-9597-a2cf3284e3ab`, run
`qualification-20260912-01-1a8a87d7-9e47-4602-89f8-a27bbbff9a45`, payload v10
(`manifest_sha256 fcbd2a9195f1b4f8a97df16c1df3d564f294edd9b5b6354841169d03fd9891c1`, 131 files).

## What worked
- The v9 startup defect is fixed: the launcher cleared its own guards, and `preflight.json` is
  `PREFLIGHT_VERIFIED` with 131 source files and all package versions matching the manifest.
- The runtime stage started: `runtime_environment.json` records `status: MATCH`.
- Five of eight ranks reached the point of publishing a terminal record.

## What failed
- Ranks 1, 2, 3, 4 and 5 published `FAILED` sidecars, all with
  `QualificationInputError: no_canvas_could_be_measured` — i.e. not a single canvas produced a row.
- Ranks 0, 6 and 7 published **nothing at all**. No sidecar, no stderr.
- `aggregate.json` is `{"detail": "missing_calibration_rank:0", "status": "FAILED"}`.
- `launcher.json`: `status FAILED`, `launcher_status 1`, `aggregate_status 1`, `preflight_status 0`.
- Scheduler: `job_failed` after `running_time_ms 316000`.

**No measurement from this job is usable.** Every rank that reported failed the same way, and the
remaining three produced no record. Nothing here supports any performance, throughput or
capability claim.

## Root cause of the five ranked failures (identified and fixed)
`calibration_runtime` called the device-scoped CUDA counters with **no device argument**:

    cuda.reset_peak_memory_stats(); cuda.synchronize(); cuda.max_memory_allocated(); cuda.max_memory_reserved()

Called without a device, these report the *process's current device*, which defaults to 0. A rank
whose model sits on `cuda:r` (r != 0) therefore reads zero allocations for its own work, the row
is classified `measurement_not_positive`, the ladder stops at its first canvas, and the rank raises
`no_canvas_could_be_measured`. That is exactly the signature of ranks 1-5.

The accepted v8 probe does not have this bug: `runtime_checks.py` passes `device` to
`reset_peak_memory_stats`, `memory_allocated`, `synchronize` and `max_memory_allocated`. The
calibration runtime was written without that detail.

Fix: the CUDA surface in `calibration_runtime` is now device-explicit end to end (`measure_canvas`
and `run_ladder` take a `device`), and `calibration.py` passes the device its model is on. Two CPU
tests pin it: one asserts every device-scoped call names the device, one asserts the ladder
forwards it to every canvas.

## Unresolved: ranks 0, 6 and 7
- Rank 0 runs the same 0.4B class of model as ranks 1-5. Under the device bug it is the one rank
  whose default device matches its work, so it should have measured rather than failed; nothing
  was published for it, and the reason is not recoverable from the available evidence.
- Ranks 6 and 7 load the nominal-2.9B weights from a 16 GB checkpoint on a single GPU, after a
  preflight that hashes 22 GB. They published nothing.

The blocker to resolving this is the deployment, not the payload: `GetJobLog` returns business
`InternalError`, so stdout/stderr never reach a reviewer. Mitigation now in place: a failed rank
writes `calibration-rank-<rank>.stderr.txt` (or `...-unknown.stderr.txt`) into the output directory
with its traceback, and `no_canvas_could_be_measured` now carries the first unavailability reasons
in its message. Together those make the next attempt diagnosable from the sidecars alone.

## Consequence
Two bounded submissions have now been spent: v9 failed on a launcher startup defect (fixed, with a
local startup probe added to staging), and v10 failed on the device-accounting defect above (fixed,
with CPU regression tests). Both defects are in calibration code written this wave and neither was
observable on a CPU-only host. A further submission is not authorized by the current instruction
("one bounded re-attempt ... only after a documented root-cause fix") and has not been made.
