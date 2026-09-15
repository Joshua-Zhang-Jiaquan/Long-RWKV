# Retention/reservation resolution — qualification-20260912-04

Date: 2026-09-13. Read-only, exactly 3 qz API calls (schema CreateJob; GetTrainScheduleConfig; GetJob reference). No CreateJob/dry-run/StopJob/credentials.

## Rejected attempt (context, not repeated)
Attempt 03 (priority 4) passed explicit `reserve_on_fail_ms='0'` + `reserve_on_success_ms='0'` → API rejected `InvalidParameter: 'retention time must be positive'`. No job admitted.

## Authoritative findings
1. **CreateJob schema (call 1)**: `reserve_on_fail_ms` and `reserve_on_success_ms` are optional (required:false) **string** fields, like `max_running_time_ms`. The schema exposes **name/jsonField/type/required only — no description text** for these fields, so their semantics (resource hold vs artifact retention, allowed range, minimum) are NOT documented in any readable authoritative source. Billing meaning: UNKNOWN — do not infer from field names.
2. **GetTrainScheduleConfig (call 2)**: workspace train config exposes `auto_recycle_train`, `auto_recycle_train_ruleset` (recycle train jobs <40% GPU util for 3 h), `timed_recycle_train`, `recycle_train_day/hour/minute`, `train_enable_slow_detect`, `train_enable_vccl`, `default_fault_tolerance_max_retry/interval_sec` — **no reserve/retention default field exists**. There is no documented workspace default to substitute for a '0'.
3. **Reference job job-83690070 (call 3, succeeded 2026-09-10)**: stored spec contains **max_running_time_ms="86400000"** and **reserve_on_fail_ms / reserve_on_success_ms ABSENT** — proof the platform accepts CreateJob with both fields omitted and applies server-side defaults invisibly.
4. **Minimum positive value (e.g. 1000 ms)**: NOT verifiable read-only — schema carries no min constraint and no business doc is reachable via available read APIs. Treating 1000 ms as valid would be logical inference, not fact.

## Recommendation (evidence-backed)
- **Fix: OMIT both `reserve_on_fail_ms` and `reserve_on_success_ms` from the request entirely** — this exactly matches the reference job's accepted-and-succeeded stored spec and avoids the positivity violation. Do not normalize "absent" to '0' (the priority mistake); absence and zero are different on this API.
- Confidence: high for acceptance-by-omission (direct precedent). Low/none for any explicit positive value.
- If a future requirement demands explicit retention, its validity needs either a business-doc source or a sanctioned test submission — outside this read-only pass.

## Unknowns on record
- Actual semantics (hold-after-fail vs result/artifact retention), server defaults applied on omission, billing interaction — all UNKNOWN from authoritative sources.
- Idle-burn guard at campaign scale: the workspace auto-recycle ruleset (3 h at <40% GPU util) is the documented protection; the per-job runtime cap remains 30 min for qualification jobs.
- Raw outputs in temp only and deleted; no credentials or raw API dumps persisted.
