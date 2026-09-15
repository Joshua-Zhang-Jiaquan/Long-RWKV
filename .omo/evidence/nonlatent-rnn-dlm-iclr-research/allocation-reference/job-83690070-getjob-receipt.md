# Sanitized allocation reference — job-83690070

- Inspected (UTC): 2026-09-12, read-only `qz train GetJob --data '{"job_id":"job-83690070-e8cf-4bd7-a3db-91d6a6977c56"}' -o json` (exit 0, single call, no retries, no other qz actions).
- job_id: `job-83690070-e8cf-4bd7-a3db-91d6a6977c56`
- name: `m4-loop-2p9b-32h100-half-r2_copy`
- status: `job_succeeded` (current_running_round=1, no fault fields present in response)
- framework: `pytorch`; image `docker.sii.shaipower.online/inspire-studio/relay2:v2` (SOURCE_PRIVATE)
- Resource shape: instance_count=4 nodes x gpu_count=8/node = 32 GPUs, `NVIDIA_H100_SXM_80G` (80 GB), cpu=160/node, mem_gi=1800, shm_gi=1800
- Quota scope IDs: project `project-160ccb20-98ab-4538-a847-01d1f83d5b0f`; workspace `ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6`; logic_compute_group `lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e` (name indicates H100 pool, cuda13.2)
- Priority: task_priority=0 (priority_name="4", priority_level=NORMAL)
- Runtime: max_running_time_ms=86400000 (24 h); actual running_time_ms=27352000 (~7.60 h) => ~243 GPU-hours consumed (32 GPUs x 7.60 h)
- Timeline (UTC): created 2026-09-10 01:02:39; resources prepared 01:03:27; run start 01:03:29; finished 2026-09-10 08:39:21
- Launcher (basename only, from private command inspection): `launch_birwkv_diffusion.sh`; model tokens observed: RWKV7-Goose-World3-2.x (2.9B-class), rwkv_diffusion — consistent with nonlatent bi-RWKV diffusion training arm
- spec_id: not returned by GetJob for this job (only pricing metadata); GPU type confirmed via `instance_spec_price_info.gpu_info.gpu_type`, not inferred from name
- Sanitization: full API output held in temp memory only; command/env contents, created_by PII, and credentials not recorded; temp files deleted after extraction
- Caveat: this job's completed runtime is historical usage only — not a spending authorization or budget ceiling for new runs
