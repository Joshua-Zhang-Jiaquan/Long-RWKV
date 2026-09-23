# Theory MVP: results and research decision

## Decision

**Do not expand this version to a long-context sweep or claim a new theory contribution yet.** The exact experiments are correct and informative, but the candidate results have classical ingredients and the frozen-model competence gate fails. The MVP has completed its decision gates; a publishable theory result has not been obtained.

The most useful direction uncovered is **representation-aware parallel refinement**: the amount of retained information does not determine the cost of sampling from the retained posterior. A future contribution needs a new certified method or tight tradeoff for approximate learned representations, with evidence connecting it to an implemented model. The present oracle linear-code sampler is not that contribution.

## What was completed

- Formal proof and novelty audits for history access, restricted scanning, binary random access, and linear compression.
- All 46,233 permutations of chains with 1–8 hops: forward/alternating formulas and Eulerian distributions checked without mismatches; an off-path cache control exposes the restriction.
- Exact binary compression optima for 1–4 source bits: three source bits and one memory bit permit 75% optimal random-query accuracy, beating the 66.7% whole-bit-storage baseline.
- All 35 rank-two encoders on four bits: equal retained information, but one-round KL 0/1/2 bits for 6/16/13 encoders. An information-set two-round oracle is exact for all 35.
- All 24,592 feasible conditional-batch identities verified by direct finite distributions.
- Exact absorbing-schedule dynamic programs for 35 encoders × 8 stage budgets, with closed-form parity-pair controls and endpoint/path-bound checks.
- A higher-order-dependence counterexample: all 15 pairs independent yet global TC 2 bits and product-law TV 0.75.
- A 480-record natural-language lookup pilot on frozen F2 and causal RWKV, with complete input/output/provenance audit and portable raw outputs.

## Frozen-model pilot

| Model | One-hop exact | Two-hop exact | One-hop gold occurrence | Two-hop gold occurrence |
|---|---:|---:|---:|---:|
| F2 |22/120 (18.3%)|1/120 (0.8%)|57/120 (47.5%)|14/120 (11.7%)|
| Causal RWKV |63/120 (52.5%)|29/120 (24.2%)|110/120 (91.7%)|32/120 (26.7%)|

The frozen gate required 96/120 exact at each depth for each model. Every cell failed. The input histories contain 121–129 tokens (one-hop) or 209–219 tokens (two-hop), plus a 32-token output budget. There is no long filler. F2 uses matched-Bernoulli 8 stages, tau 0.7; RWKV uses greedy decoding. All outputs and confidence intervals are retained in `results/mvp_lookup_audit/`.

Gold occurrence anywhere is not exact-answer accuracy. In 31 of 57 reference-containing F2 one-hop outputs, other candidate colors also appear. R0's one-hop exact/occurrence gap suggests a substantial output-format issue; its two-hop failure remains under either metric. This is a new development diagnostic, not a retuned or substituted E3 confirmation endpoint. The runtime remains the recorded FLA implementation, without independent official-kernel parity.

## Compute and checks

- User cap enforced: 32 H100s queued/running; largest allocation used: 8.
- Exactly one GPU job: `job-736704dc-2211-4c31-a137-afdb08f952ed`; succeeded.
- Scheduler-reported running time: 316 seconds on 8 GPUs, approximately 0.70 H100-hours. This is not a provider billing estimate.
- No new pretraining or fine-tuning. No long-context expansion after the failed gate.
- Tests, independent audit, and final artifact hashes are recorded in `results/theory_mvp/completion_manifest.json`.

## Files

- `theory_mvp/RESEARCH_NOTE.md`: derivations and numerical interpretation.
- `results/theory_mvp/mvp_note.pdf`: standalone research note with figures and pilot results.
- `theory_mvp/THEORY_REVIEW.md` and `LINEAR_COMPRESSION_ADDENDUM.md`: proofs, prior-art audit, caveats.
- `results/theory_mvp/`: exact JSON outputs and standalone PDF/PNG figures.
- `results/mvp_lookup_audit/audit.json` and `portable_records.json.gz`: fully audited model observations.
- `theory_mvp/PROTOCOL.md`: scope, contracts and gates.

## Next research milestone

Before spending more GPUs, specify a dependency-aware procedure for approximate learned representations and a bound that remains informative after including estimation error, representation storage, and dependency-detection cost. Compare its actual assumptions and guarantee against coding/matroid results, conditional-independence samplers, and recent serial-depth theory. A purely pairwise dependence check is insufficient by the exact counterexample.

In parallel, a new explicitly labeled development study could qualify a more reliable natural-language interface or an independently checked runtime. It must use fresh examples, preserve this failed gate, and establish competence before any new length sweep. Neither future step is claimed complete here.
