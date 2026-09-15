# Resource resolution — qualification nonlatent-h100-qualification-20260912-01

Date: 2026-09-12. Read-only discovery, 6 qz API calls (bound was <=8). No CreateJob/dry-run/StopJob, no config dump, no credentials.

## Commands executed (all exit 0)
1. `qz spec`
2. `qz schema train.CreateJob`
3. `qz schema workspace.GetLogicComputeGroupNodeSpecs`
4. `qz schema train.GetTrainScheduleConfig`
5. `qz workspace GetLogicComputeGroupNodeSpecs --data '{"workspace_id":"ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6","logic_compute_group_id":"lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"}' -o json`
6. `qz train GetTrainScheduleConfig --data '{"workspace_id":"ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"}' -o json`

## Resolved spec (live, evidence-backed)
- **spec_id candidate: `7166bd2e-6cbe-4bd9-be38-762d11003e7f`** ("8卡160核": gpu_count=8, cpu_count=160, memory_size=1800)
- Field source: `train.GetTrainScheduleConfig` → `Result.predef_train_spec` (JSON string), entry `logic_compute_group_ids` explicitly includes **`lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e`** (plus 10 other pools; spec id is pool-shared — pool is selected by `logic_compute_group_id`, not the spec id)
- Workspace enforces predef specs: `Result.use_predef_train_spec = 1`
- Live LCG node catalog (call 5): 70 entries for this LCG incl. **8-GPU `NVIDIA_H100_SXM_80G` (80 GB)** nodes at cpu_count=183 / ~1888 GiB, tagged `support_job_type=distributed_training` (also interactive/tensorboard/serving variants). This listing is a shape catalog only — it carries **no spec ids**.
- Historical cross-check: reference job job-83690070 (same project/workspace/LCG) observed per-node shape 8 GPU / 160 CPU / mem_gi 1800 / shm_gi 1800 — exactly matches predef spec 7166bd2e. (GetJob does not echo spec_id; match is by shape.)

## CreateJob request shape (schema-verified, NOT dry-run)
- Required: `logic_compute_group_id`, `name`, `project_id`, `workspace_id`, `framework`, `command`, `framework_config` (array)
- `framework_config[].spec_id` (string), `image` (string), `image_type` (string), `instance_count` (int32), `shm_gi` (float64)
- **`max_running_time_ms` type = string** → authorized cap passes as `"1800000"` (reference job used string `"86400000"`, confirming practice)
- **`auto_fault_tolerance` type = bool** → explicit `false`; workspace `default_auto_fault_tolerance = false` and `default_fault_tolerance_max_retry = 0` already match the no-autoretry requirement
- Priority via `task_priority` (int32, optional)

## Unknown / residual risks (not blockers, but unverified)
- **Quota headroom not checked** (GetWorkspaceQuota / ListProjectQuotas not called) — submit-time quota exhaustion remains possible; fail once, no resubmit loop per authorization.
- Spec→GPU-type binding is inferred at LCG level (LCG name "开发区-H100-cuda13.2版本-183核" + live catalog NVIDIA_H100_SXM_80G); the predef spec entry's own `gpu_type` field is an empty string.
- Image `relay2:v2` (SOURCE_PRIVATE) not re-validated against the image service this turn; it is the same project/image the reference job ran.
- 30-min runtime interacts with workspace auto-recycle ruleset (recycles train jobs <40% GPU util after 3 h) — irrelevant at 30 min, noted for completeness.

## Conclusion
No admission blocker found at schema/discovery level. Exact authorized tuple is expressible: LCG lcg-71b971a7 + project-160ccb20 + ws-9dcc0e1f + image relay2:v2 + framework_config=[{spec_id: 7166bd2e-6cbe-4bd9-be38-762d11003e7f, image relay2:v2, image_type SOURCE_PRIVATE, instance_count 1, shm_gi 1800}] + max_running_time_ms "1800000" + auto_fault_tolerance false. Raw API outputs held in temp memory only and deleted; no command/env/credential contents persisted.
