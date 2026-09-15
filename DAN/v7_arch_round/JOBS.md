# Job registry — v7 architecture round

## mHC arm (training; owner: integrate-latent-guidance)
- m4-mhc-2p9b-16h100-r1  job-a9c5c8c3-… (silent stop at 21min; resubmitted)
- m4-mhc-2p9b-16h100-r2  (36h wall-clock cap kill; auto-resumed)
- r3/r4 resume rounds — specs in specs/m4_mhc_16gpu_r{3,4}.json
- endpoint evals: decode-gate-mhc-s9500 job-2f3ecb94-ea89-45db-bb9a-9513b4d7593d;
  decode-gate-n2-s9500 job-a6d3a79a-f76b-4bac-ab…; panels cap_lm_m4mhc_s{2000..9500}

## Loop arm (training; m4-loop-2p9b, 4750 steps, 4×8 H100 low lane)
- r1/r2 fault-tolerance rounds (specs m4_loop_32gpu_half{,_r2}.json)
- r3 FINAL: job-83690070-e8cf-4bd7-a3db-91d6a6977c56 — completed 2026-09-10 08:39Z
- MMLU probes (prober daemon): s500 job-cc21a18a · s1000 job-40f4c5f9 · s1500
  job-226c3d03 · s2500 job-8e866808 · s3000 job-669c0cd1 · s3500 job-d874cb95 ·
  s4500 job-b817f008
- s4000 tracker wave: probe job-057c4913, caplm job-3325dc98
- s4750 endpoint MMLU probe (manual; 4750 not a %500 boundary): job-d2a20266

## Loop endpoint battery (coordinator session, 2026-09-10)
- cap_lm panels (H100, all succeeded ~37min): r1 job-b8f89afd-400b-427a-9e66-706a057235ae ·
  r2 job-e60fbf50-d9e1-423c-89b5-a05806aeb220 · r3 job-2e258ae7-2b6a-4ea0-b87f-a4d96edaaa24 ·
  r4 job-87180397-363b-4306-94db-90e161134590
- decode gate v1 (H200 lcg, STUCK queued → stopped): job-855c2767 · job-bdf30451 ·
  job-35093a07 · job-20b60b3b · job-d302c4e1 · job-b6c8f1b2
- decode gate v2 (H100 resubmit, -h100): r1 job-da5cd2d2-9419-4655-a6ef-cc57ca1a56f6 ·
  r2 job-a1fbd06e-bd77-4530-9e15-04502e18b9f4 · r3 job-421d78a7-a6b5-49e6-a9ce-07bfd8edfcb7 ·
  r4 job-2902abf0-b61f-4ed8-8691-d46012d032d0 · n2-s4000 job-a7e5346d-ecb9-41dd-af0f-de5b50b3a74d ·
  n2-s6000 job-ff21bd8e-eff5-4f48-b3f2-50841a9e1286

## Key checkpoints on disk
- mHC endpoint: outputs_birwkv_diffusion/m4-mhc-2p9b/step_00009500 (+ neutral copy
  m2_baseline_triangle/m4mhc_endpoint_ckpt)
- loop endpoint: outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750 (+ neutral copy
  m2_baseline_triangle/m4loop_endpoint_ckpt — model.pt+meta.json, prober-neutral)
- N2 references: n2-knowpt-2p9b/step_00002..95000_probe_copy + cap_lm_n2_s{2000,4000,6000,8000,9500}

## Full-budget extension (user-approved follow-on, 2026-09-11)
- m4-loop-2p9b-32h100-ext job-21820297-02f9-4f0e-b79a-094e4f4724a9 (4×8 H100 low lane,
  prio 3): resume step_00004750 -> 9500 steps = 10B tokens, N2-matched mid-cosine lr
  (spec specs/m4_loop_ext_9500_job.json). Matched-budget gate: PRIMARY PPL vs N2
  s6000/s8000/s9500 panels; decode gate at s9500 endpoint; tracker/probers fire
  automatically on %500/%2000 boundaries.
- Pruning executed per results/ckpt_pruning_ledger_20260910.md + W-L1 closed-lane
  leftovers (141G; preserved: wl1 s2000_probe_copy + e_target manifest). Disk 222G free.
- prio-3 attempt job-21820297 TERMINAL job_failed (low-lane preemption ~18min in: NCCL up, first forward, silent stop — mHC r1 signature). REPLACED by prio-4 resubmit:
- m4-loop-2p9b-16h100-ext-p4 job-85faa154-1887-4663-aca8-01462ecf6d81 (2026-09-11, 2x8 H100, mb4xga4=256 rows/step, walltime 36h) — resume s4750 -> 9500.

## s9500 extension endpoint battery (2026-09-13)
- caplm r1 job-bbc8636b · reps2 job-fc222a1f · reps3 job-cc94cd00 · reps4 job-c7bd009c
  (outdir cap_lm_m4loop_s9500{,_reps{2,3,4}})
- decode r1 job-13e56892 · r2 job-f5790bec · r4 job-5c600c42 (sampler_gate_m4_loop_s9500*)
- MMLU probe job-02ad6dd7 (94/200 = 47.0); N2-s9500 decode reused from the mHC round
- Endpoint preserved: m2_baseline_triangle/m4loop_ext_endpoint_ckpt (neutral) + raw step_00009500

## v8 follow-on (2026-09-13, user: "go on and may use the dclm dataset")
- B-pilot chat SFT: b-pilot-loop-sft-16h100-p4 job-de56cde1-ae66-484b-b7b2-38db81a9f392
  (loop s9500 warm start, loop flags KEPT, b3-sft recipe, 2000 steps). NOTE: s2_traj
  STEM pack EXCLUDED — npz format, incompatible with FineWebPackedPickleDataset.
- DCLM data build: mlfoundations/dclm-baseline-1.0 via ModelScope DIRECT (no proxy,
  user directive — HF is network-blocked, proxy dead; ModelScope reachable). Builder
  scale/data/build_dclm_pack.py: 52 pinned files (global-shard_01, local 0+1, rev
  edf5473d), ~20B tokens -> pt_training_data/dclm_gpkgfs/dclm_4096_packed, exact
  packed-pkl convention (mask=ids!=0, 64-align, no row crossing). Build pid 2998200,
  log /tmp/dclm_build.log.
- A0 done: m4-loop s8500/s9000 pruned (s9500 + neutral copy kept).
- Lane A 20-30B: m4-loop-2p9b-30b-ext-p4 (spec m4_loop_30b_ext_job.json) — knowledge x1 +
  DCLM x2 (14.2B pool, 130 files, convention-verified) + rev4_final x1; resume s9500 -> s25000;
  queues behind B-pilot on the 16-GPU high lane.
- DCLM pool COMPLETE: 130 files / 14.2B tokens / 83G, zero download failures
  (base 52 + extend 78; manifests in pt_training_data/dclm_gpkgfs/dclm_4096_packed/dclm_manifest.json).

## v8 campaign — Lane A relaunch (2026-09-15, user-approved plan)
- **Lane A (the 16B loop extension): `job-fd80e224-e5dd-4cc8-ac42-f8125e3ac2df`**
  name m4-loop-2p9b-30b-ext-p4 · 2x8 H100 lcg-71b971a7 · prio 4 · 36h cap · FT retries 3.
  Resume m4-loop-2p9b/step_00009500 (optimizer intact) -> STEPS=25000 = +15,500 steps
  ~= 16.3B tokens (26.2B cumulative). Submitted 2026-09-15 ~09:55Z.
- TWO PRIOR ATTEMPTS DIED AT STEP 0, both on this same spec; both root causes fixed here:
  1. TOKEN_DIR entry 4 was `packed_rev4_final_gpkgfs` = **0 *.pkl**; the 142 shards live in
     its `pt_mixture_4096_packed/` subdir. Fixed in the spec (this was Lane A's only blocker).
  2. `qz/launch_birwkv_diffusion.sh` preflight validated **only dir #0**, so it printed
     PREFLIGHT OK while entry 4 was dead. Now loops over every comma-separated entry.
     VERIFIED to catch the old bug (old code: OK; new code: "TOKEN_DIR[4] has no pkl shards").
- Parity predicate fix (`train/train_birwkv_diffusion.py`): the loop extrapolation bar is an
  INIT-TIME contract but was keyed on `start_step == 0`, and a weights-only warm start
  (`--resume-from`) reports step 0 while carrying trained gates (loop.gates_raw=5.57e-3) --
  that is what failed `job-de56cde1` (rel=1.489e-02, all 3 retries, 0 steps). Now keyed on
  "were the loop gates loaded from a checkpoint", derived in `_merge_latent_keys` where
  "present in ckpt" is already distinguished from "backfilled zero-init".
  Lane A did NOT need this (its RUN_NAME collides with the existing step_00009500, so
  find_resume wins and start_step=9500); the SFT lane does (fresh RUN_NAME).
- Measured runtime: **23 s/step** at this shape (job-85faa154: 109,197,000 ms / 4750 steps)
  => ~99h / ~4.1 days for 15,500 steps, ~1580 H100-h across ~4 FT rounds. The v8 plan's
  "5.1 s/step" is STALE — do not budget from it.
- Disk: 240G free / 98% full. keep_last=3 (a non-CLI function default) => 147G steady state,
  peak 196G during a save => ~43G headroom. save-every kept at 500 (keep_last is COUNT-based,
  so the footprint is identical at 1000; 500 halves how much an FT restart loses).
- Pool at submit time: 16 H100 already busy on lcg-71b971a7 from project-632c8db8's two
  `v3-baselora-rollout` jobs (NOT ours), plus our own 30-min ICLR qualification job.
  Note `ListJobs` WORKS as of today (it did not before); `GetJob <id>` still fine.
- CORRECTION to the 2026-09-13 entry above: "HF is network-blocked" **no longer holds**.
  Verified 2026-09-15: huggingface.co API 200s and real byte-range downloads succeed with
  proxies stripped (the same pattern build_dclm_pack.py:50-63 already uses), for all three
  chat corpora + 8 MC-QA datasets (all gated=false). ModelScope's REST API returns 405 for
  these paths and the package is not installed. The chat corpus therefore comes from HF.

## v8 campaign — benchmark evals on the loop s9500 endpoint (2026-09-15)
- **A1 OBQA/RACE baseline (the missing reading), 1-GPU: `job-836ba887-4752-4d31-8a6a-086de2b355fb`**
  name m4-probe-a1-baseline-s9500-obqa-race-1g · spec m4_probe_a1_baseline_s9500_1gpu.json.
  No OBQA/RACE reading existed for ANY m4-loop checkpoint, so the extension had nothing to
  compare against. Protocol copied byte-for-byte from the recorded precedents
  (tasks_f2_s14000_obqa_race / tasks_rwkv7goose_obqa_race): obqa,race / max_samples 500 /
  max_new_tokens 32 / steps 16 / commit_group_size 4 / temperature 0.2 / chat_frame 0 / seed 42.
  New alias `tasks_m4loop_step9500_obqa_race`. Comparable against b3 54.80/48.00 (s1000),
  b3 45.40/45.20 (s8000), f2 58.40/54.80, rwkv7goose 46.40/31.80.
- **DUPLICATE, 8-GPU, needs stopping: `job-82259622-0ec0-43d3-a253-73524c37b8dc`**
  Submitted first on the 8-GPU shape (7166bd2e) before I checked the pricing API; it sat in
  job_queuing 12h (Lane A held the pool) then STARTED once capacity freed. The probe uses only
  CUDA_VISIBLE_DEVICES=0, so this burns 8 GPUs on 1 GPU of work and writes the SAME files
  (`cola_proxy_generic.py:128` opens "w" and writes per-sample, so the two jobs share the fd
  target). Both are byte-identical deterministic runs, so they should converge, but the output
  MUST be verified for interleaving/duplicates before use. StopJob was denied to me 3x by the
  auto-mode classifier; the user was asked to run it.
- **THE SHAPE LESSON (cost 12h of wall clock).** spec_id 7166bd2e is only the 8-GPU SHAPE
  (160 cpu / 1800 GiB). My lcg-71b971a7 also publishes 1/2/4-GPU H100 shapes, read from
  `qz resource-price GetLogicComputeGroupResourceSpecPrices`:
    1 GPU  79fe954a-be92-4772-ac0b-94ad8a79b7bb  (20 cpu, 200 GiB)   <- use this for probes
    2 GPU  26ef0d6e-330d-4650-a18a-7e1fbe8f3717  (40 cpu, 400 GiB)
    4 GPU  8a53ac21-299a-4dee-85e9-9c04a544cf8d  (80 cpu, 900 GiB)
    8 GPU  7166bd2e-6cbe-4bd9-be38-762d11003e7f  (160 cpu, 1800 GiB)
  Every 1-GPU probe must use 79fe954a. The 1-GPU resubmit cleared the queue in 56 seconds.
- **Pre-SFT generation baseline: `job-ae6d7ad5-7ce6-414b-ae89-2dfa95bb12d4`**
  name m4-gen-baseline-s9500-gsm8k-humaneval · 1-GPU · spec m4_gen_baseline_s9500.json.
  GSM8K + HumanEval via qz/launch_capability_eval.sh with COMMIT_GROUP=4 PINNED (the launcher
  defaults it to 0 = the legacy multi-commit mode that CAUSES repetition degeneracy) and
  CONDITIONS=iter16 to match the MC probes' --steps 16. This is the denominator for B4's rule
  "HE/GSM8K below the plain-model pre-SFT numbers => SFT not transferring" (v8 plan: plain
  model 0.61 HE / 1.21 GSM8K; b3-sft 1.22/0.68 at s500).
- **capability_eval_data was MISSING and was rebuilt.** `vendor_real_tasks.py` needs
  `capability_eval_data/raw/`, which did not exist, so no generation benchmark could run at all.
  Re-downloaded anonymously from HF and vendored with pins:
  gsm8k_tasks.jsonl 1319 rows sha256 03d4d909… · humaneval.jsonl 164 rows sha256 2e730932… ·
  math500_tasks.jsonl 500 rows sha256 f2513472…  (mbpp skipped: its raw file is still absent).
- **acc_calc.py OVERWRITES accuracy_summary.csv** (`write_summary_csv` line 545, `open(...,"w")`)
  by re-walking $O with RECURSIVE_SEARCH=True and EVAL_DIR_PREFIX="tasks_" (114 dirs intact, so
  the rewrite is faithful). A timestamped backup was taken before the first A1 run:
  `accuracy_summary.csv.pre-A1-baseline.bak`. Diff it against the result.
- Lane A mid-run health at step 11440: `val_mask_ce` 3.8056/3.8148/3.8162 at steps 10000/10500/
  11000, inside the lineage's stable 3.81-3.83 band (old run: 3.8181/3.8280/3.8170/3.8173/3.8167
  at 2500-4500). Train mask_ce's apparent jump 1.9 -> 3.8 at step ~9920 is a DATA-MIXTURE
  artifact (the sampler moving off the easy knowledge shards into DCLM/rev4 = 92% of the blend),
  NOT degradation. No pre-registered kill rule fires.


