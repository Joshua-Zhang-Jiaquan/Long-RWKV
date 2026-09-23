# Revision proposal: predict conditional-error differences on unseen dependency classes

Status: **draft for review, not a frozen GPU protocol**. Prepared in response to the 23 September advisory review of main commit `44fa1c813d07cb884ee27935a7fcf3596ba51754`. No new neural predictions or GPU jobs have been run for this proposal. Existing published evidence and the reviewed PDF remain unchanged.

## Contribution to pursue

A prospective, transferable decision rule for choosing a refinement schedule from calibration on other dependency classes. Keep the existing latency certificate as implementation-specific supporting evidence. This route matches the user's theory-driven mechanism focus; it does not require winning a broad deployment benchmark to succeed scientifically.

The revised scientific question is: **can source-calibrated conditional-error diagnostics choose between equally costly, equally dependence-preserving schedules on unseen posterior structures?** Success would add a prediction that the existing decomposition alone does not make. Failure would be informative and must not be relabeled a positive result.

The prior literature already separates learning and factorization errors; the fixed-partition equality is not the proposed novelty. See Lavenant and Zanella, Proposition2: https://arxiv.org/html/2510.25544v2#S2. The ICLR reviewer guide asks about new, relevant knowledge, not necessarily leaderboard performance: https://iclr.cc/Conferences/2027/ReviewerGuidelines.

## A sharper controlled comparison

Let U be uniform on four bits, let B be an invertible 4x4 binary matrix, and let P = B U XOR s for a public four-bit syndrome s. Output all eight bits, with public equations, randomized names, output order and row order. Both candidate four-coordinate bases must be recoverable from public prompt information; no private target or latent-generator label may enter selection.

Compare U-first and P-first two-round samplers. U and P each have four independent fair marginals. Once either is revealed, the remaining four bits are determined. Consequently **both schedules have D=0 and both use two complete context scans**. Their KL difference is entirely a learned-error difference. A dependence-only score cannot distinguish them.

The deterministic conditional functions use B in one direction and its inverse in the other. Their parity fan-in profiles can differ. This motivates a calibrated error predictor, but does not prove that more fan-in means larger neural error. First-round marginal error must also be included; large response margins alone are insufficient.

The CPU exploration exhausts all 20,160 invertible 4x4 matrices. It finds 25 distinct homogeneous-code weight enumerators, and 13,440 matrices with different sorted forward/reverse fan-in profiles. `feasibility.py`, `feasibility.json` and `test_feasibility.py` record this algebraic feasibility check. These numbers are not learned-model evidence.

Split source and held-out structures by the entire weight-enumerator class, using a deterministic hash rule fixed before training. Different weight enumerators certify that the homogeneous codes cannot be related by arbitrary output-coordinate permutation; equal enumerators do not prove equivalence. This guards against merely relabeling the old perfect-matching family. Candidate allocation is 15 source classes and10 held-out classes; final representatives and class allocation must be frozen before any neural evaluation. Never move a difficult held-out class into training.

## Prediction without held-out model evaluation

Train only on source classes. A disjoint source-calibration panel supplies oracle-to-model marginal errors, stratified using a small predeclared feature set: fair versus deterministic marginal, visible-set rank/size, minimal parity relation size, and context placement. Include first-round errors. Source calibration may fit bin means or one fixed low-dimensional model; the choice, smoothing, minimum support and fallback must be locked before viewing held-out outcomes.

For each held-out prompt, compute only its public structural features and analytic D. Predict each policy's score as D plus the sum of source-calibrated per-query errors. This is an empirical transfer hypothesis, not a theorem that source conditional errors generalize. The original context-inclusive coverage ratio is infinite across disjoint context families; it cannot justify a finite cross-family guarantee without additional assumptions.

Before running either held-out policy, write a hashed prediction artifact containing every policy score, chosen policy, uncertainty/unsupported-feature flag and fallback. No held-out logits, losses, confidence estimates or endpoint KL may be used to choose that prompt's policy. Then evaluate both complete endpoint laws, retaining all prompts, all model seeds and all failures. Unsupported predictions use a frozen fallback and remain in the primary denominator.

Primary comparison: paired held-out KL of a dependence-only policy with a predeclared content-hash tie-break versus the predicted policy. Also compare random choice in expectation, fixed policies, a source-calibrated unconditional preference, and a simple structural fan-in heuristic. This determines whether calibration adds value beyond a hand-built structural rule. The endpoint-best policy is a retrospective oracle reference, never a deployable baseline.

Primary inference should resample training lineages and held-out code classes, then prompts within classes; report individual lineage and class effects as well. Freeze a nontrivial effect-size floor before running. A proposed success gate is at least0.1 nat mean improvement, a positive conditional interval, useful decisions on at least80% of prompts, and an explicit absolute-quality requirement relative to the one-call product floor. These thresholds are proposals, not selected using outcomes. If both policies are already essentially exact, merely breaking ties accurately does not support a useful scheduling claim.

## Independent lineages and the 32+16 H100 cap

Use six fresh full training pipelines from public base weights, with independent data/corruption and newly introduced conditioning-parameter initialization seeds. They share a public pretrained backbone, but do not descend from selected seed71. Evaluate all six fixed terminal checkpoints without selecting competent survivors. Report the success fraction and total cost, including failures and qualification. Six lineages remain a small sample; avoid a precise universal success-rate claim.

A starting curriculum proposal is the existing short-task learning phases followed by fixed16K distance-balanced adaptation, adapted to the source code classes. Keep the complete terminal budget fixed before held-out inference. Do not assume the new tasks will learn in the old update count. A source-only runtime/numerical qualification must establish feasibility; it cannot tune on held-out quality.

Allocation: four eight-H100 jobs on the primary32 project and two on the extra16 project. Queued and running reservations together must never exceed32/16/48. First run a small source-only qualification, then freeze source hashes, recipe, update counts, seeds, masks, calibration model, held-out panel, estimands and retention rules. Six main jobs can then train concurrently. Release training GPUs after source calibration; launch held-out evaluation only after all prediction artifacts are frozen. Avoid idle GPU barriers and automatic retries.

The first-day milestones are a qualified immutable implementation, six terminal-or-live runs with measured completion estimates, frozen source-calibrated predictions when ready, and paper restructuring in parallel. Completion within24hours must be estimated from the qualification throughput, not promised from the GPU count. The existing fastest-completion preference governs utilization; no artificial memory filling or result-dependent experimental changes.

This new study would test prediction across lineages. It would not retroactively turn the existing six selected-seed71 adaptations into independent-lineage evidence for the old position-training treatment.

## Practical-baseline and reproducibility work

Keep a matched cached-causal and numerically qualified attention frontier as the alternative contribution route. If pursuing a practical speed/quality claim, these comparisons become essential. For the present mechanism route, do not imply that the existing1.5s certificate settles that practical question. A causal comparison must train/evaluate on the same posterior task and report full joint quality, not compare a timing-only lookup run against the parity result.

All seven required original checkpoints exist: selected initial plus six adaptations, each approximately5.086GiB, about35.6GiB total including optimizer state. A checkpoint release needs a storage/delivery choice and verification. To preserve the original recorded file hashes, publish lossless split files with a reconstruction index, or provide the complete original files on a suitable host. If publishing inference-only exports, give their new file hashes, model-tensor correspondence and an explicit mapping to original checkpoint hashes; do not label them byte-identical `resume.pt` files. A reconstruction workflow must include the initial1500-update parent and1000-update selected branch, not start from an unavailable already-adapted checkpoint. This proposal has not uploaded weights.

## Manuscript changes after evidence is available

1. Center the introduction on prospective schedule selection under conditional-error uncertainty. Present the old criterion as established accounting supporting a new testable prediction.
2. Add a compact diagram of a bidirectional recurrent scan, the answer canvas and the two candidate reveal bases. State why equal D and equal call count make the new comparison diagnostic.
3. Put held-out predictions, realized errors, selection regret and complete lineage outcomes in the main results. Keep negative results and the old position intervention available.
4. Move scheduler IDs, project allocations and utilization-target history to repository operations records; retain a compact compute-cost disclosure in the paper. No scientific outcome or failure may disappear during this edit.
5. Answer the review's four questions with measured outcomes, not a claim that algebra or tighter wording has already resolved significance.

Do not revise the abstract to claim transfer before the prospective test succeeds. If predictive selection fails, report that result and reassess the contribution rather than adding further matched prompts to the old family.
