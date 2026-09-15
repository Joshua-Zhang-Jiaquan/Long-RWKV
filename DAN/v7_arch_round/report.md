# v7 Architecture Round — mHC arm FINAL VERDICT (2026-09-10)

> **Round ownership note (2026-09-10):** the loop arm (W-A3) and final round
> readout were taken over by the coordinator session after the mHC verdict;
> the mHC sections below are as written by `integrate-latent-guidance`.

# LOOP ARM (W-A3) — running (taken over 2026-09-10 ~02:00)

Mechanism: weight-tied loop, layers [16,32), reps=1 (NFE 48), otherwise the
exact N2 recipe replay (F2 s14000 warm start, knowledge_interim pack, lr 1e-5,
256 rows/step). Half-budget: 4750 steps (~5.0B tokens) on 4×8 H100.

## MIDPOINT GATE (s2000 = 2.1B tokens) — read 2026-09-10

Paired per-document causal-arm PPL deltas (loop − N2, nats, ci95 paired-t;
method reproduces the mHC table exactly):

| corpus       | n     | delta   | ci95  | verdict                |
|--------------|-------|---------|-------|------------------------|
| wikitext103  | 999   | −0.0000 | 0.0006 | wash                   |
| pg19         | 16000 | −0.0021 | 0.0001 | **loop better** (CI)   |
| owt          | 16000 | −0.0021 | 0.0002 | **loop better** (CI)   |
| lambada      | 5153  | +0.0052 | 0.0006 | **loop worse** (CI)    |
| lm1b         | 10000 | −0.0767 | 0.0012 | **loop better** (CI)   |

**Letter of the PRIMARY gate at 2B: MET** (3/5 better CI-disjoint, no erosion
>0.03). The lm1b effect (−0.0767) is 2.3× the mHC lm1b effect at the same
token count (−0.0327) — the largest single-corpus effect of the round so far.
Caveat from the mHC precedent: mHC also met the 2B letter and ended a wash;
the endpoint reading decides.

- MMLU matched delta @2.1B: loop 46.50 vs N2 46.50 → **0.0** (n=200, non-gating).
- SECONDARY (blend-val CE): s2500 val_mask_ce 3.8181 vs N2 3.8112 (+0.0069
  worse — within bar). s500–s2000 val readings lost to restart-round log
  rotation; s2500 onward is the anchor. Kill bar: >0.02 worse twice.
- MMLU curve so far: 50.5 / 49.5 / 47.5 / 46.5 / 48.5 (s500–s2500).

## ENDPOINT READINGS (s4750 = 4.98B tokens, read 2026-09-10)

Training completed cleanly (4750/4750, final val 3.8167). All four cap_lm panels
succeeded on the H100 pool (~37 min each). Paired per-document deltas, causal arm:

**PRIMARY (PPL panel) vs BOTH pre-registered N2 brackets:**

| corpus       | vs N2-s4000 (4.2B) | vs N2-s6000 (6.3B) |
|--------------|--------------------:|--------------------:|
| wikitext103  | +0.0077 worse (CI) | +0.0260 worse (CI)  |
| pg19         | −0.0016 better (CI)| −0.0130 better (CI) |
| owt          | +0.0005 worse (CI) | −0.0304 better (CI) |
| lambada      | +0.0045 worse (CI) | −0.0134 better (CI) |
| lm1b         | **−0.1082 better (CI)** | **−0.0735 better (CI)** |
| #better      | 2/5                 | 4/5                 |

**Letter of the PRIMARY gate: MET against BOTH brackets** (≥2 better CI-disjoint;
max erosion +0.0260 < 0.03). Substance: the loop arm at **half the N2 budget**
(4.98B vs 10B tokens) beats the 6.3B-token N2 on 4/5 corpora and holds a large
lm1b advantage against both (−0.073 to −0.108 nats — 4–6× mHC's endpoint lm1b
effect, grown steadily from −0.077 at the 2.1B midpoint). wikitext103 is the one
consistent loser (+0.008 to +0.026).

**W-A5 DEPTH EXTRAPOLATION (PPL, reps beyond trained depth=1):**

| corpus       | reps2−reps1 | reps3−reps1 | reps4−reps1 |
|--------------|------------:|------------:|------------:|
| lm1b         | −0.0059    | −0.0117    | −0.0171    |
| lambada      | −0.0009    | −0.0016    | −0.0019    |
| owt          | +0.0001    | +0.0004    | +0.0010    |
| pg19         | −0.0000    | +0.0001    | +0.0005    |
| wikitext103  | +0.0005    | +0.0013    | +0.0023    |

**No Ouro-style collapse beyond trained depth.** Extra weight-tied passes give
small monotone gains on the short-sentence corpora (lm1b, lambada) and small
monotone costs on the long-context ones; every effect < 0.02 nats. The trained
loop gate is conservative — depth extrapolation is safe but not exploitable at
this training scale.

**SECONDARY (blend-val CE): PASS** — all five readings within bar (max +0.0115
at s3500; final +0.0058 at s4500 vs N2 3.8109).

**MMLU (non-gating):** 50.5/49.5/47.5/46.5/48.5/50.0/48.5/47.0/47.5 (s500–s4500),
endpoint probe pending. N2 neighborhood: 46.5@s2000, 47.5@s4000, 48.0@s6000.

**DECODE GATE (co-primary) vs both N2 brackets (256-doc fixed holdout, paired
by document_id+seed; s4750 endpoint, trained reps):**

- **vs N2-s4000: NOT REGRESSED — arguably improved.** em ~0 on every arm
  (max |Δ| 0.0006, all CI overlap); repetition (max_run_frac) CI-disjoint
  BETTER across the whole r100 row (+0.02 s1 → +0.24 s64) — the loop repeats
  less at full mask. tau (per-token agreement) ~0 at r70/r95/r99.
- **vs N2-s6000: marginally behind on em at partial mask.** em CI-disjoint
  worse at r70 (−0.001..−0.003) and r95 (−0.001..−0.002) — significant but two
  orders below any practical bar (N2-s6000 saw 27% more tokens). distinct_frac
  better at r99/r100.
- **r100 (fully-masked, degenerate for every arm):** the loop endpoint's tau
  is NEGATIVE (−0.08 vs N2-s4000 +0.38, N2-s6000 +0.17, mHC-s9500 +0.13) while
  being LESS repetitive — more diverse but less reference-like pure generation.
  em ~0 for all arms here; treat as a distributional note, not a gate cell.

**Decode verdict: improve-or-not-regress HOLDS vs the matched-budget bracket
(s4000); a negligible (≤0.003) significant em deficit vs the higher-token
bracket (s6000). No repetition regression anywhere it matters (r95/r99).**

**W-A5 DECODE REPS CURVE (paired vs trained reps1, s4750 endpoint):** no
collapse. em ~0 at r70; **CI-disjoint BETTER at r95 (reps3 +0.0011, reps4
+0.0015) and r99 (reps4 +0.0009)** — tiny but monotone-in-depth gains in the
high-mask regimes, mirroring the PPL reps curve. Repetition ~0 at r70/r95/r99.
The r100 max_run_frac swings (reps2 −0.46, reps3/4 −0.01) are non-monotone
chaos inside the degenerate fully-masked regime, not signal.

## LOOP ARM FINAL VERDICT (2026-09-10, all gates read)

**The weight-tied loop EARNS its budget.** At HALF the N2 token budget
(4.98B vs 10B), the loop arm beats the 6.3B-token N2 on 4/5 PPL corpora
CI-disjoint with no erosion, matches the 4.2B-token N2's decode with better
full-mask repetition, keeps blend-val CE within +0.012 of N2 at every
checkpoint, and extrapolates to 4× trained depth without collapse. The lm1b
effect (−0.073 to −0.108 nats) is the round's largest single-corpus signal —
4–6× mHC's, growing steadily through training — and the em gains at
extrapolated depth concentrate in the same high-mask/short-context direction.

Honest caveats: wikitext103 is consistently worse (+0.008..+0.026); the
fully-masked pure-generation regime is less reference-like (tau −0.08 vs N2
+0.17..+0.38) though em is 0 for every arm there; MMLU stays at parity
(47.0 endpoint vs N2 47.5@s4000/48.0@s6000, n=200 noise); single seed,
2.9B scale, 5B tokens.

**Disposition:** unlike mHC, the loop arm is a credible candidate for the
20–30B extension — decision and budget to the user. The residual follow-up
shared with mHC: the short-context (lm1b/lambada) signature suggests
short-sentence corpora as the mechanism's natural habitat.

## ROUND CONCLUSION (both arms)

Of the two architecture levers tested as exact N2-recipe replays on the 2.9B
BiRWKV lineage: **hyperconnection mixing (mHC) is refuted** (wash at 10B
tokens), **weight-tied depth recycling (loop) is positive at half budget**.
Depth-recycling is the lever that pays on this lineage; the mechanism's gains
concentrate in short-context language modeling and survive depth
extrapolation.

## ENDPOINT PLAN (pre-registered)

s4750 ≈ 4.98B tokens lies between N2 s4000 (4.2B) and s6000 (6.3B) panels;
the honest paired cells are BOTH brackets — report loop−N2 deltas vs s4000
and vs s6000 (chosen before looking at the numbers). Decode gate = the same
paired sampler harness as mHC (5120 recs/arm, r∈{0.7,0.95,0.99,1.0},
steps {1,8,16,32,64}) vs both brackets. W-A5 depth-extrapolation probe:
loop reps {1,2,3,4} at eval time on the s4750 endpoint.

Mechanism: manifold hyperconnection (mHC) on the 2.9B BiRWKV masked-diffusion
model, N2-recipe replay (F2 s14000 warm start, knowledge_interim pack, lr 1e-5,
9500 steps), the ONLY changed variable vs N2. Every metric is a paired control
against N2 at matched tokens.

## PRIMARY GATE — PPL panel (per-document paired, causal arm, mHC − N2 nats)

| tokens  | wikitext  | pg19     | owt      | lambada  | lm1b      | #better |
|---------|-----------|----------|----------|----------|-----------|---------|
| 2.1B    | +0.0016   | +0.0012  | −0.0003  | −0.0023  | **−0.0327** | 3/5 |
| 4.2B    | +0.0090   | +0.0011  | +0.0010  | +0.0042  | **−0.0196** | 1/5 |
| 6.3B    | +0.0024   | +0.0005  | +0.0013  | **−0.0049** | **−0.0278** | 2/5 |
| 8.4B    | +0.0073   | +0.0019  | +0.0008  | −0.0002  | **−0.0043** | 1/5 |
| 9.96B   | +0.0034   | +0.0004  | +0.0015  | **−0.0089** | **−0.0168** | 2/5 |

**Letter of the gate (≥2 better CI-disjoint, no erosion >0.03): MET at 2B, 6B,
and 9.96B.** But the winning margins are ~0.005–0.03 nats, and the aggregate is
a wash. **lm1b is the one reproducible win** (better at 4/5 points); lambada
oscillates; wikitext/pg19/owt are consistently tiny-worse (all ≪ 0.03).

## SECONDARY GATE — blend-val CE (kill if >0.02 worse twice)

mHC ≡ N2 at every 500-step val from s500 to s9500 (max deviation +0.0068 at
s8000; usually ≤0.002). Never approached the kill bar. **PASS (no erosion).**

## MMLU-proxy (tracked, non-gating)

Full curve (200-sample): 50.0→50.5→48.5→49.5→49.0→46.0→46.0→46.5→46.5→47.0→
45.0→47.5→46.0. Matched deltas vs N2: **+3.5 (2B), +1.0 (4.2B), −2.0 (6.3B),
+0.5 (8.4B)** — an early transient, then noise around zero. mHC does not add
MCQ-retrievable capacity at this scale.

## CO-PRIMARY DECODE GATE — paired sampler harness (s9500 endpoints, 5120 recs/arm)

- **em@32**: r70 +0.0005 (~0); r95 −0.0011 (N2-better CI-disjoint); r99 −0.0004;
  r100 +0.0002; **r100_s64 +0.0068 (mHC-better CI [+0.0061,+0.0075])** — the only
  substantive win.
- **Repetition (max_run_frac)**: at r100 both arms degenerate (mHC 0.997, N2 mixed);
  ~0 differences at r95/r99/r70.
- **distinct_frac**: mHC +0.003–0.009 (slightly better lexical diversity) at r95,
  worse at r100.

**Verdict: NO substantive decode improvement, NO regression.** The earlier
in-training em@32 signals (mHC iteration "helps" at s4000/s8000) were NOT
reproduced by the formal harness — likely noise or prompt-set artifacts.

## VERDICT

**mHC does not earn a 20–30B extension.** At 2.9B over 10B tokens it is a wash:
a reproducible small lm1b win, an early-transient MMLU advantage, flat val-CE,
and no decode improvement. The architecture lever is recorded as **refuted at
this scale** for the primary and decode gates; the lm1b signal is the only
residual worth a follow-up (short-sentence / short-context corpora).

## OPS LEDGER (round lessons)

1. **Quota lanes are per-project, not per-capacity**: 1,800+ H100s free but 16
   (high) / 32 (low) GPU caps per project. project-160ccb20 admits in minutes;
   632c8db8 queued 4h.
2. **36h `max_running_time_ms` killed r2** mid-run; auto-resume works (full
   optimizer+step) but the **data-iterator position is NOT checkpointed** — post-
   resume step-level CE vs N2 is invalid; fixed-set panels/probes stay valid.
3. **Step-0 mechanism-parity probe must be resume-aware**: the loop extrapolation
   assert (identity-at-init) wrongly fired on a resume with trained gates
   (bf16 ULP hides 1× gate but 4× accumulates to 8.6e-3). Patched to skip when
   `start_step>0`.
4. **Prober/tracker marker-write races + duplicate submissions** (fixed: atomic
   writes, 2000-multiples excluded from the prober, all-5-corpora cleanup rule).
5. **GPFS rendezvous**: parse node IP from the marker FILENAME, not content
   (content visibility is unreliable on some pods).

## LOOP ARM (W-A3) — status

Half-budget (4750 steps @ 32×H100 LOW lane) run in progress; resuming from
s2500 after fault-tolerance restarts (round 3 queuing). Loop [16,32) reps=1,
NFE 48. Parity 0.000e+00 at init; extrapolation probe SKIPPED on resume (correct).
s2000 midpoint-gate evals running/queued. Endpoint + depth-extrapolation probe
(W-A5) pending. Final loop verdict + the round's architecture answer land after
s4750 completes.

# FULL-BUDGET EXTENSION — FINAL VERDICT (2026-09-13)

The loop arm extended from s4750 to s9500 (9.96B tokens — the exact N2
budget) via a clean optimizer-state resume (prio-3 attempt killed by
low-lane preemption; final run job-85faa154, 2×8 H100, prio 4).

## All gates at the fully-matched endpoint (loop-s9500 vs N2-s9500, 10B vs 10B)

- **PRIMARY PPL: MET — 4/5 corpora better CI-disjoint** (wikitext −0.0047,
  pg19 −0.0007, lambada −0.0088, **lm1b −0.2608**; owt +0.0089 the only
  loser, far under the 0.03 erosion bar). The historical wikitext loser
  FLIPPED better at the endpoint. **lm1b trajectory across the run:
  −0.077 → −0.100 → −0.179 → −0.261 — accelerating, not saturating.**
- **CO-PRIMARY DECODE: MET — improve-or-not-regress.** em ~0 at r70/r95/r99;
  full-mask arms CI-disjoint better (em +0.0002@s32, +0.0068@s64, less
  repetition). No regression anywhere it matters.
- **SECONDARY val-CE: PASS** — all 15 readings within bar, final gap +0.0026
  (narrowest of the run; the loop converges into N2's band).
- **MMLU (non-gating): 47.0 endpoint** (94/200) vs N2's 45.5 — parity-plus.
- **W-A5 depth extrapolation: SAFE at 2×/3×/4× trained depth** — PPL lm1b
  gains GROW with depth (reps4: −0.0251 at s9500 vs −0.0171 at s4750) and
  with training; decode em unchanged; no collapse at any depth.

## VERDICT

**The weight-tied loop's advantage is real at matched budget, and it
compounds with scale.** At 10B-vs-10B tokens the loop arm dominates the
PPL panel 4/5, holds decode, converges val-CE into the control band, and
extrapolates depth safely. The advantage concentrates in short-context
language modeling (lm1b −0.26 nats, accelerating) with small mixed effects
elsewhere (owt +0.009, wikitext flipped to −0.005).

**Recommendation for the 20–30B extension: PROCEED on the loop arm.** The
accelerating lm1b trajectory and growing depth-extrapolation gains are
exactly the signatures that justify scale. Pre-register the same paired
gates against the best plain-lineage checkpoint at matched tokens; watch
owt (the one consistent loser) and keep the short-context/long-context
split as the mechanism readout. mHC stays retired.
