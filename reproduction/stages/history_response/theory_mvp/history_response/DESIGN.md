# Exhaustive evidence-response follow-up

This follow-up is motivated by the completed distance study, whose numerical results were already observed. It is an explicitly post-study diagnostic, with this protocol and analysis frozen before new predictions. It does not create a new independent test sample or remove the selected-lineage limitation.

Use all seven existing selectors (original and near/balanced adaptation seeds 20271011/12/13), all 32 original primary problems, and all three 16K placements. For every condition, enumerate all 16 target-distributed information-set histories and flip each of the four parity syndromes separately. Hold visible bits, instructions, labels, distractors and token positions fixed. Each intervention changes exactly one native input token and one required hidden target. No retraining, model selection, new latency comparison, structure selection, or early stopping.

There are 672 checkpoint/problem/position conditions, 43,008 paired interventions, and 54,432 forward calls (one all-masked plus 16 original-history plus 64 flipped-history calls per condition). Cache originals only within each condition. Keep normalized log probabilities for all eight coordinates and all calls. Use the same serial FP32/IEEE implementation and repeat/cross-rank numerical screen as the completed study. Report every history and constraint; retain all negative outcomes.

Primary endpoint: per-problem mean paired correct target CE at 16K far, averaging 16 histories and four constraints. Primary contrast: near minus balanced, separately for each adaptation seed and then seed-averaged. Use 10,000 paired problem bootstrap draws, seed 20260923, retaining pairing across positions, checkpoints, histories and constraints. Histories and constraints are exact inner expectations, not independent statistical samples. Intervals are conditional on these checkpoints and this selected lineage. Reusing the 32 problems means this is not independent replication.

Report far-minus-near contrast, response-floor and centering-penalty contributions, both-variants-correct fraction, signed logit contrast, unaffected-target probability changes, worst-history original second-round error and worst paired CE. Report all seven selectors and all positions. Positive mean effects alone do not establish uniform history competence. The old all-zero/index-mod-four probe is recomputed and compared with the archived values; original first/second errors must also agree within 1e-4 nats. Differences above tolerance stop scientific reporting for investigation, without silently removing cells.

## Exact theoretical connection

For problem x, history h and constraint r let c(x,h,r) be the original correct CE of the target controlled by r, and c_flip(x,h,r) its CE after flipping r. Define L=(c+c_flip)/2 and orient logits to obtain L=softplus(-Delta/2)+G, G>=0. Average uniformly over all 16 histories and four constraints. Then

    4 mean(L) = (E_second_original + E_second_flipped_target)/2,
    mean(L) = mean(response_floor) + mean(centering_penalty).

Here E_second_flipped_target sums only the intervened target errors across four distinct counterfactual prompts, then averages histories. It is not the full four-target error of any single counterfactual prompt. The exact identity connects this exhaustive diagnostic to a symmetrized conditional-error law. It does not equate it to the original endpoint E_second; both quantities are reported separately. The original sufficient two-pass condition remains E_first + E_second_original < 4 ln 2.

Support requires a positive far paired-CE reduction with a positive conditional interval and matching direction in all three adaptation seeds, together with disclosed selectivity/worst-case diagnostics. No claim about a unique internal circuit, independent training lineages, instruction-distance isolation, natural text or general memory superiority follows. This exhaustive audit is most useful if it exposes exceptions to the original fixed-history account.

## Execution

One eight-H100 job per checkpoint, at most 48 queued+running GPUs, split primary 32 and extra 16; admission uses the existing locked capacity checker. Scheduler limit two hours/job; no automatic GPU retries. Expected duration is roughly 15–30 minutes/job based on the prior 16K forward latency, plus queue/load time, not a runtime guarantee. Record busy time, memory and scheduler hours; optimize fastest completion without artificial memory filling. Reuse the validated compiled kernel cache. Freeze source, checkpoint and prior raw-output hashes before submission. Preserve the completed paper and its delivery manifest; incorporate this follow-up only after complete validation.
