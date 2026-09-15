# v7 Architecture Round — standalone project folder

**Question:** do architecture levers (mHC, weight-tied loop) improve the 2.9B
BiRWKV masked-diffusion model over the N2 knowledge recipe at matched budget?

**Method:** each lever replays the N2 recipe EXACTLY (F2 s14000 warm start,
knowledge_interim pack, lr 1e-5, 256 rows/step) so architecture is the only
changed variable; every N2 metric is a paired control (per-document deltas,
paired-t ci95).

**Arms:**
- **mHC** (manifold hyperconnections, n=4 streams) — full 9500 steps (10B tokens).
  VERDICT: refuted at this scale (see `report.md`).
- **Loop** (weight-tied extra passes, layers [16,32), reps=1, NFE 48) — half
  budget 4750 steps (4.98B tokens). PRIMARY PPL gate MET vs both N2 brackets
  (4/5 corpora better vs N2-s6000 at half its token budget; lm1b −0.07…−0.11).
  Decode gate pending at folder creation.

**Gates (pre-registered):** PRIMARY = 5-corpus PPL panel beats N2 CI-disjoint on
≥2 corpora at matched tokens, no erosion >0.03 nats. CO-PRIMARY DECODE = paired
sampler harness improve-or-not-regress. SECONDARY = blend-val CE ≤ N2 (kill if
>0.02 worse twice). MMLU tracked, never gating. W-A5 = loop depth-extrapolation
probe (reps beyond trained depth; no collapse found).

## Contents
- `report.md` — the round's verdict document (canonical; the old
  `DAN/v7_arch_round_report.md` path is a stub pointer).
- `ARCHITECTURE.md` — the model in detail: the BiRWKV masked-diffusion base,
  the mHC and weight-tied-loop mechanisms, trainer/eval integration, and the
  full file map.
- `TRAINING_HISTORY.md` — both arms' training history (restarts, quota
  incidents, wall-clock) and ALL performance curves: val-CE, MMLU, the PPL
  gate tables, decode gates, W-A5 depth curves.
- `code/` — the round's usable code, snapshotted: model mechanisms
  (models/), trainer + CPU safety cases (train/), eval harnesses (eval/),
  job launchers (qz/). Live copies remain in scale/ of both trees.
- `results/` — ability-curve CSVs (mHC, loop), the full gate-analysis
  snapshot, and the checkpoint-pruning ledger. Panels/sampler outputs stay in
  place on GPFS (167M each): `cap_lm_m4{mhc,loop}_s*`, `sampler_gate_m4_*`,
  `cap_lm_n2_s*` under `/inspire/hdd/global_user/zhangjiaquan-253108540222/`.
- `specs/` — every job spec shape used: training (mHC 16gpu r1-r4, loop
  32gpu half), eval templates (probe, caplm, sampler-highmask).
- `scripts/` — the round's automation: per-500 MMLU prober, 2000-boundary
  ability tracker, endpoint battery submitter, paired gate analysis.
- `JOBS.md` — registry of every job ID in the round.

## Checkpoints retained (user directive: best-performed + last only)
- `outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750` — loop endpoint
  (last = best-gated; model+optim) + neutral copy at
  `m2_baseline_triangle/m4loop_endpoint_ckpt`
- `outputs_birwkv_diffusion/m4-mhc-2p9b/step_00009500` — mHC endpoint (last)
  + neutral copy at `m2_baseline_triangle/m4mhc_endpoint_ckpt`
- `outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00006000_probe_copy` (best
  performed: MMLU 48.0, pre-registered bracket) + `step_00009500` (last)
- Deletions recorded in `results/ckpt_pruning_ledger_20260910.md`

## Code (lives in the shared trees — not copied)
- `scale/models/birwkv7_diffusion.py` — `attach_residual_streams` (mHC),
  `attach_backbone_loop` + `loop_reps_override` forward param (loop;
  extrapolation reuses the last trained gate). In BOTH
  `DiffRWKV-RELAY/scale/` and `qz_stage_traj4096_v7/scale/` (jobs run staged).
- `scale/models/residual_streams.py` — `BackboneLoopControl`, mixer.
- `scale/eval/capability/birwkv_diffusion_model.py` — loader sniffs
  `residual_streams.*` / `loop.*` ckpt keys; `LOOP_REPS_OVERRIDE` env wraps
  forward for W-A5 (works for cap_lm AND sampler harness).
- `scale/train/test_backbone_loop.py`, `scale/train/test_residual_streams.py`
  — the mechanisms' CPU safety cases (identity at init, grad flow, weight-tying).
- `scale/qz/{launch_lm_eval,launch_sampler_eval}.sh` — eval launchers.

## Ops lessons (round-specific; full ledger in report.md)
- Quota lanes are per-project: 16 GPU high / 32 GPU low running caps on
  project-160ccb20 regardless of free capacity.
- The `birwkv_sampler_highmask_job.json` template carries the H200 train-pool
  lcg (`lcg-df089db8`) — eval jobs built from it queue behind training. The
  H100 dev pool (`lcg-71b971a7`) admits in seconds; the endpoint decode jobs
  were resubmitted there (-h100 names).
- ListJobs is broken (all page_size/keyword combos error) — track job IDs and
  use GetJob. See also qz-listjobs-broken memory.
- 4750 is not a %500 boundary — the prober never fires on it; endpoint probes
  at non-round steps must be submitted manually.

**Ownership:** mHC arm + all training/prober automation by session
`integrate-latent-guidance` (jobId 6f32d845); loop-arm endpoint battery, W-A5,
and final report by the coordinator session (takeover 2026-09-10, user
directive "go on the non-latent version's loop and mhc arm").

## Follow-on plan
`../v8_loop_scale_chat_plan.md` — the 20–30B loop extension (Lane A) and the
chat/instruct post-train lane (B), with pre-registered gates and budgets.
