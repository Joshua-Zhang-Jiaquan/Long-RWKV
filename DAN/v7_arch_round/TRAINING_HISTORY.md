# Training history & performance curves — v7 architecture round

## The control: N2 (knowledge recipe)

Both arms replay the N2 recipe EXACTLY — architecture is the only changed
variable, so every N2 metric is a paired control (per-document deltas,
paired-t ci95):

- warm start F2 s14000 (`f2-2p9b-ptcorpora`, the PT-corpora full-pretrain peak)
- data: `knowledge_interim_gpkgfs/pt_mixture_4096_packed` (7.6B-token knowledge pack)
- lr 1e-5 (warmup 200), global batch 256 rows/step (mb4 × 8GPU × ga4 on 2 nodes)
- 9500 steps ≈ 10.0B tokens; save/val every 500

**N2 reference curves** (used by both arms' gates):
- val_mask_ce: s500 3.8240 · s1000 3.8312 · s1500 3.8343 · s2000 3.8348 ·
  s2500 3.8112 · s3000 3.8202 · s3500 3.8055 · s4000 3.8087 · s4500 3.8109
- MMLU (200-sample): s2000 46.5 · s4000 47.5 · s6000 48.0 · s8000 46.5 · s9500 45.5
- Panels: cap_lm_n2_s{2000,4000,6000,8000,9500}; decode sampler_gate_m4_n2_s9500

## Arm 1 — mHC (m4-mhc-2p9b): full 9500 steps, REFUTED

History: 64-GPU submission blocked by per-project quota lanes (16 high / 32
low running caps despite 1800+ free H100s) → 16-GPU high-priority shape
(2×8 H100, mb4×ga4 = same 256 rows/step semantics). r1 died silently at 21
min (one-off, resubmitted); r2 hit the 36h max_running_time cap mid-run —
auto-resume recovered optimizer+step but NOT the data-iterator position (a
fixed-set evals stay valid; step-level train-CE comparisons do not). r3/r4
completed 9500 steps ≈ 9.96B tokens.

**val_mask_ce ≡ N2 at every 500-step checkpoint** (max deviation +0.0068 at
s8000; usually ≤0.002) — SECONDARY gate PASS trivially.

**MMLU curve (s500→s9500):** 50.0, 50.5, 48.5, 49.5, 49.0, 46.0, 46.0, 46.5,
46.5, 47.0, 45.0, 47.5, 46.0 — matched-token deltas vs N2: +3.5 (2B), +1.0
(4.2B), −2.0 (6.3B), +0.5 (8.4B): an early transient, then noise.

**PRIMARY PPL panel (mHC − N2 nats, per-doc paired):**

| tokens | wikitext | pg19 | owt | lambada | lm1b | #better |
|--------|----------|------|-----|---------|------|---------|
| 2.1B | +0.0016 | +0.0012 | −0.0003 | −0.0023 | **−0.0327** | 3/5 |
| 4.2B | +0.0090 | +0.0011 | +0.0010 | +0.0042 | **−0.0196** | 1/5 |
| 6.3B | +0.0024 | +0.0005 | +0.0013 | **−0.0049** | **−0.0278** | 2/5 |
| 8.4B | +0.0073 | +0.0019 | +0.0008 | −0.0002 | **−0.0043** | 1/5 |
| 9.96B | +0.0034 | +0.0004 | +0.0015 | **−0.0089** | **−0.0168** | 2/5 |

Met the ≥2-corpus letter at 3 of 5 points but margins ~0.005–0.03 nats;
lm1b the only reproducible win. Decode gate: no substantive change (only
r100_s64 +0.0068 em better). **VERDICT: refuted at this scale** — does not
earn the 20–30B extension.

## Arm 2 — weight-tied loop (m4-loop-2p9b): half budget 4750 steps, POSITIVE

History: half-budget by design (the mHC-first ordering left the low lane);
4×8 H100 low-priority lane, same 256 rows/step (mb4×ga2×16GPU). r1/r2 died
to fault-tolerance restarts (GPFS rendezvous fixed earlier in the round:
parse node IP from marker FILENAME, not content); r3 boot 2026-09-10 01:10Z,
completed 4750/4750 at 08:39Z (~4.98B tokens, 0.087-0.091 step/s).

**val_mask_ce (loop vs N2 at matched steps):** s2500 3.8181 (+0.0069) ·
s3000 3.8280 (+0.0078) · s3500 3.8170 (+0.0115) · s4000 3.8173 (+0.0086) ·
s4500 3.8167 (+0.0058) — SECONDARY gate PASS (kill bar was >0.02 twice;
s500–s2000 readings lost to restart log rotation).

**MMLU curve (s500→s4750):** 50.5, 49.5, 47.5, 46.5, 48.5, 50.0, 48.5, 47.0,
47.5, 47.0 — at parity with the N2 neighborhood (endpoint 47.0 vs N2 47.5@s4000
/ 48.0@s6000; n=200, non-gating).

**PRIMARY PPL panel at the s4750 endpoint (loop − N2 nats, per-doc paired):**

| corpus | vs N2-s4000 (4.2B) | vs N2-s6000 (6.3B) |
|--------|--------------------:|--------------------:|
| wikitext103 | +0.0077 worse | +0.0260 worse |
| pg19 | −0.0016 better | −0.0130 better |
| owt | +0.0005 worse | **−0.0304 better** |
| lambada | +0.0045 worse | −0.0134 better |
| lm1b | **−0.1082 better** | **−0.0735 better** |

MET vs both pre-registered brackets — at HALF the N2 token budget vs s6000.
The lm1b effect grew steadily: −0.077 (s2000) → −0.108 (endpoint vs s4000).

**Decode gate (256-doc fixed holdout, paired by doc+seed):** vs s4000 em ~0
everywhere, full-mask repetition CI-disjoint BETTER (+0.02..+0.24); vs s6000
negligible significant em deficit (−0.001..−0.003 at r70/r95). r100 (fully
masked, degenerate for all arms): loop tau −0.08 vs N2 +0.17..+0.38 — more
diverse, less reference-like pure generation (em 0 for all arms there).

**W-A5 depth extrapolation (eval at 2/3/4× trained depth):** PPL — no
collapse; monotone small gains lm1b (−0.0059 / −0.0117 / −0.0171), small
costs on long-context corpora, all <0.02 nats. Decode — em CI-disjoint
BETTER at r95/r99 for reps3/4 (+0.001..+0.0015). Safe, not exploitable.

**VERDICT: the loop EARNS its budget** — credible candidate for the 20–30B
extension (user decision). Full tables: report.md + results/.

## Job & incident registry

See JOBS.md for every job ID (training rounds, probes, panels, decode jobs,
the H200-pool queue stall and H100 resubmit). Ops lessons consolidated in
report.md §OPS LEDGER and README.md.

## Full-budget extension (2026-09-12 →, user-approved follow-on)

Prio-3 attempt (job-21820297, 4×8) died to low-lane preemption 18 min in;
replaced at priority 4 (user directive) as job-85faa154 (2×8 H100,
mb4×ga4 = 256 rows/step). Resumed from step_00004750 at 2026-09-12 03:55Z.

**val_mask_ce vs N2 (SECONDARY):** s5000 3.8035 (+0.0033) · s5500 3.8084
(+0.0063) · s6000 3.8091 (+0.0154) — all within the 0.02 kill bar.

**PRIMARY at s6000 (MATCHED budget + MATCHED tokens vs N2-s6000):**

| corpus | loop−N2 (nats) |
|--------|---------------:|
| wikitext103 | +0.0241 worse |
| pg19 | −0.0117 better |
| owt | −0.0293 better |
| lambada | −0.0128 better |
| lm1b | **−0.1001 better** |

4/5 better CI-disjoint — the round's strongest reading. The lm1b effect
continues growing (−0.077@s2000 → −0.074@s4750 → −0.100@s6000). Remaining:
s8000 panel, s9500 endpoint battery, decode gate, W-A5 reps.

**PRIMARY at s8000 (matched budget + tokens vs N2-s8000):** 3/5 better —
lm1b **−0.1794** (continues growing: −0.077@s2000 → −0.100@s6000 → −0.179@s8000),
owt −0.0004, pg19 −0.0002 (both CI-disjoint but ~zero); lambada +0.0033 and
wikitext103 +0.0088 worse. The advantage is concentrating in lm1b; the
mid-context margins (owt/pg19, strong at s6000) decayed to zero at s8000.
val-CE: s6500 +0.0126 · s7000 +0.0128 · s7500 +0.0114 · s8000 +0.0112 —
stable within bar.

## s9500 ENDPOINT (2026-09-13, training complete 9500/9500, 9.96B tokens)

**Final val: 3.7963 vs N2 3.7937 (+0.0026) — the narrowest gap of the run;
SECONDARY gate PASS on all 15 readings.**

**DECODE gate vs N2-s9500 (matched budget+tokens, trained reps):** em ~0 at
r70/r95/r99; full-mask arms CI-disjoint BETTER (em +0.0002@s32, +0.0068@s64;
repetition better: the loop repeats less). No repetition regression anywhere.
tau(r100) loop −0.066 vs N2 +0.205 — the diverse-but-less-reference-like
signature from the s4750 endpoint persists.

Pending: cap_lm PRIMARY panel + W-A5 reps{2,3,4} + decode reps{2,4} + MMLU.

**PRIMARY at s9500 — THE DECISIVE CELL (fully matched budget + tokens, 10B vs 10B):**

| corpus | loop−N2 (nats) |
|--------|---------------:|
| wikitext103 | −0.0047 better (historical loser FLIPPED) |
| pg19 | −0.0007 better |
| owt | +0.0089 worse (only loser, far under bar) |
| lambada | −0.0088 better (re-flipped from s8000) |
| lm1b | **−0.2608 better** |

4/5 better CI-disjoint, no erosion. **lm1b trajectory: −0.077 (s2000) →
−0.100 (s6000) → −0.179 (s8000) → −0.261 (s9500) — accelerating, not
saturating.** The matched-budget advantage is real and concentrates in
short-context language modeling.

**W-A5 decode at s9500:** depth extrapolation safe (em ~0 at 2×/4× depth;
reps4 em +0.0012 better at r95). No collapse.
