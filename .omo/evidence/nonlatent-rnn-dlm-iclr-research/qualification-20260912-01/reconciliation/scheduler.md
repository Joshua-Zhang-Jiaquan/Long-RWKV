# Scheduler reconciliation — qualification-20260912-01 (ambiguous CreateJob 2026-09-12T08:30:43Z)

Read-only pass: 4 of <=6 qz calls. No CreateJob/dry-run/StopJob, no config/credentials, no broad job listing (control query exposes only the already-known reference job).

## Prior invalid attempts (context, not repeated)
- `page`/`page_num` pagination keys → `InvalidParameter` ("page or page_size too large" / "parameter error"). Root cause: schema pagination key is **`PageNumber`** (int32) + `page_size`.

## Queries this pass (all exit 0)
1. `qz schema train.ListJobs` — filters: keyword, workspace_id, project_ids[], job_ids[], status_list[], created_at_begin/end; pagination `PageNumber`+`page_size`
2. `ListJobs {"keyword":"7bef6cf8-fda4-46b9-84ff-d7ca4fe5b892","workspace_id":"ws-9dcc0e1f-...","project_ids":["project-160ccb20-..."],"page_size":10,"PageNumber":1}` → `Result: {"jobs":[],"total":0}`
3. `ListJobs {"keyword":"qualification-20260912-01","workspace_id":"ws-9dcc0e1f-...","page_size":10,"PageNumber":1}` (project filter dropped to rule out scoping) → `{"jobs":[],"total":0}`
4. **Control**: `ListJobs {"keyword":"m4-loop-2p9b","workspace_id":"ws-9dcc0e1f-...","page_size":10,"PageNumber":1}` → total=6, includes reference `job-83690070-e8cf-4bd7-a3db-91d6a6977c56` — proves the search path + pagination + scope work for this identity

## Authoritative finding
- **No job exists matching the reserved request name or its unique UUID nonce** under workspace ws-9dcc0e1f (with and without project scoping), via a demonstrably functional keyword search.
- Therefore the 08:30:43Z CreateJob attempt did NOT materialize as a schedulable job: no job_id was ever returned, and the server now confirms zero jobs carry the reserved name/nonce.
- Status: `ambiguous_no_retry` resolves to **no-live-job / no-submit** with high confidence. No orphaned GPU allocation under the reserved name; nothing to stop.
- GetJob verification of "the actual job" is impossible — there is no actual job to fetch (no id exists).

## Residual limits (honest unknowns)
- A job created under a *different* name (server-side rename) cannot be found by name search; schema offers no list-by-request-hash or created-window+nonce search short of enumerating jobs in the window (rejected: broad-listing prohibition). Risk accepted as negligible because the controller sent the exact reserved name (unique nonce suffix) and name search is the index the API exposes.
- Whether the server ever accepted the request (e.g., async validation failure) is not observable through available read APIs — no submit-receipt endpoint in `qz spec`. The empty result is the strongest available evidence.

## Safety posture going forward
- Reservation `qualification-20260912-01-7bef6cf8-fda4-46b9-84ff-d7ca4fe5b892` remains unconsumed; any resubmission decision belongs to the user/parent (authorization allows maximum_job_submissions=1 — treating the failed attempt as consuming the submission slot is a parent-policy decision, flagged here, not decided).
- Raw outputs in temp only, deleted; only the matching-record fields above (plus control reference job id) are persisted. No unrelated job data retained.
