# Exhaustive response audit: supported interpretation

Status: COMPLETE. All seven checkpoints, 672 conditions and 43,008 paired interventions validated. The 16-page PDF passed build and visual review; an independent raw-probability audit reproduces the primary estimates and collateral-error counts.

The new experiment is a post-study diagnostic prompted by the original fixed-history limitation. Its protocol, checkpoints, model implementation, complete panel and primary analysis were frozen before predictions. It reuses the original 32 problems and seven fixed checkpoints, so it is not independent replication. All 16 histories and four syndrome flips are exact inner expectations, not 64 independent samples.

## Supported by the six validated adaptation selectors

- Far paired-CE reductions are 1.99033349, 1.75013422 and 0.69308839 nats for adaptation seeds 20271011/12/13. Their conditional problem-bootstrap intervals are [1.66992977, 2.33527969], [1.52948948, 1.97762873], and [0.69306342, 0.69310807].
- The seed-averaged reduction is 1.47785204 nats, with conditional interval [1.32745064, 1.63708384]. This supports low average conditional error across histories, extending the original local diagnostic.
- Balanced far mean paired CEs are 0.0000278000, 0.00000121361 and 0.0000297298. These are symmetrized original/flipped-target errors, not original endpoint KLs.
- An exact summation identity connects paired CE to original and flipped-coordinate conditional errors. Nonnegativity yields the supplementary sufficient bound E_info <= E_first + 8 mean_pair_CE. It is elementary and conservative, applies to the declared target law, and does not certify the entire counterfactual prompt or natural-language performance.

## A stronger selectivity claim is contradicted

The frozen worst-case measurements expose large changes on unaffected targets. The largest changes across positions are 0.9338334, 0.99203795 and 0.19523327 for the three balanced models. The fixed all-zero probes did not reveal these exceptions.

An explicitly post hoc inspection counts three correct-to-incorrect changes among 55,296 unaffected-target comparisons: seed11 has one far and one middle event; seed12 has one middle event. These comparisons are dependent and confined to this enumerated panel. Mean full flipped second-round error remains below 0.002385 nats in every balanced model/position cell. Flipped first-round predictions were not collected, so this does not supply a full flipped-prompt endpoint KL.

The paper should claim improved average evidence use and mostly selective responses, with rare collateral failures. It should not claim uniform selectivity, invariance of every unaffected prediction, or identification of a unique internal circuit.

## Retained limits and corrections

All adaptations share the selected starting lineage. The problems and structure catalog were previously observed, public instruction/evidence blocks move together, and the benchmark is synthetic. The follow-up does not fix those limitations. No new latency or memory advantage is claimed.

The original manuscript called matched native prefixes exact-length. Recorded RWKV primary prefixes are 16,383 tokens; attention cost prefixes are 16,384 under the same requested 16K budget. The manuscript now states those actual lengths. Measurements and frozen inputs are unchanged; the completed prior delivery is preserved in a verified archive.

## Delivery checks

All archived metrics reconcile within 8.30e-5 (frozen tolerance 1e-4). Independent arithmetic differs by at most 7.11e-15. All 288 balanced model/problem/position cells pass the supplementary sufficient response bound on the original target law. Seven successful jobs consume 11.89 additional H100-hours. Peak reservation is 48 (primary 32 + extra 16), and the final campaign census is zero. Whole-job GPU busy is 84.0–88.0%; memory occupancy is 8.7–9.0%, under the later fastest-completion priority.
