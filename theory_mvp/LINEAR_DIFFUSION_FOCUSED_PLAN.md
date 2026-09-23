# Active paper focus: affordable diffusion for long-context generation

User direction, 2026-09-22: use linear attention to reduce cost and diffusion to
improve long-context performance; make the argument theory driven and verify it
with MVP experiments. This supersedes centering the paper on antithetic history
training. Existing frozen experiments and adverse results remain preserved.

## Central hypothesis and boundary

At a fixed inference budget, a linear-cost recurrent denoiser can afford more
context-reading denoising passes than a quadratic-attention denoiser. Those
passes improve generation only if learned conditional estimation error is low
enough to realize the reduction in dependence lost by parallel token reveals.
The paper must verify both the cost advantage and the conditional-learning
requirement. Neither follows from calling the architecture linear attention.
RWKV-7 is the main generalized-delta recurrent realization, not an exact
implementation of softmax attention or every finite-feature linear kernel.

## Predictive mechanism

For fixed public context c and ordered reveal groups pi, the existing exact
identity is KL(P_c || Q_theta,pi,c) = D_pi(c) + E_theta,pi(c). For the same model,
a multi-call policy improves forward KL over one-call sampling exactly when
E_multi(c) - E_one(c) < D_one(c) - D_multi(c).
A stronger sufficient certificate is D_multi + E_multi < D_one, since any
one-call product sampler has KL at least D_one. This compares distributions;
it is not automatically a strict exact-match or downstream-accuracy guarantee.
Report validity and within-support diversity separately.

Let H be clamped context length and N generated target length. At fixed widths,
linear recurrent mixing costs O(K(H+N)) across K calls, whereas dense softmax
mixing costs O(K(H+N)^2). Projections, vocabulary heads, constants, batching,
precision, and temporary memory must be included in measured costs. Flash
attention's lower memory use does not remove its quadratic attention arithmetic.
An autoregressive RWKV comparator can cache state and must not be costed as
repeated full-prefix scans.

The existing independent-reveal dependence bounds scale with N, not H. They
cannot prove a long-context advantage just by substituting H for N. The missing
empirical link is E_theta,pi(H): whether the recurrent denoiser extracts and uses
distant evidence as H grows. Re-reading cannot recover information discarded
under a state-only access contract. This MVP uses retained full context.

## Minimal decisive experiment

Use a single controlled long-context task generator with disjoint keys
and examples across train/development/test. The local pair topology is fixed
across splits; this tests transfer to unseen keys and positions, not unseen
dependency structures. Pair each base problem
across context lengths and evidence locations. Include an evidence-only version
to separate failure to use evidence from failure to find distant evidence.
Train the 455M RWKV denoiser on the actual task, rather than assuming unrelated
pretraining or the binary posterior curriculum transfers to long contexts.

Two complementary task outcomes are necessary within that generator:
- A deterministic retrieval/copy control checks whether the relevant fact was
  extracted; its zero-entropy target does not itself test dependence cost.
- An evidence-conditioned multi-token posterior with known dependent answers
  permits exact small-target D/E accounting. Use a task with learnable local
  conditionals, and require development evidence of that competence before
  funding a long-context sweep. Preserve the current harder N8 parity failures.

Primary intervention: same trained RWKV checkpoint, same inputs and stochastic
output law, differing only in reveal schedule/call budget K. Measure one call,
a fixed two-call information-set partition, a two-call pair-preserving control,
a fixed random two-way partition, and eight sequential calls. Fix policies on
development data before test. Report both matched-call and matched-measured-cost
comparisons; give each method its own measured cost rather than equating K with
latency. Do not select a favorable K or checkpoint on test outcomes.

Primary practical cost comparator: a parameter- and training-budget-accounted
softmax attention denoiser on the same public task, objective, and corruption
law. The prepared Pythia405M comparator retains its native tokenizer and
pretraining; report native token lengths and distractor counts, and do not
interpret its accuracy gap as a controlled architecture effect. This choice
avoids replacing either released embedding vocabulary or claiming a matched
initialization that has not been trained. A strict same-tokenizer/initialization
architecture ablation remains outside the current MVP claim.
Primary practical comparator: causal RWKV with appropriate causal training and
cached generation. The same-checkpoint K comparison isolates the diffusion
schedule; comparisons to causal RWKV also change training/access and must be
identified as such. A smaller matched-width operator benchmark can verify the
scaling law, but cannot replace the trained accuracy comparison.

Initial length grid: evidence-only, 1K, 4K, and 16K tokens, with near/middle/far
placements. Expand to32K only after the competence gate and runtime permit it.
Use shared base problems across lengths; report the length interaction, not only
absolute long-context accuracy. Longer inputs without a short-context competence
anchor are not evidence of long-context ability. Freeze sample counts and seeds
from a development-based precision/runtime calculation before final testing.

## Decision gates and negative outcomes

1. Verify isolation, target leakage, gradients, and the sampler law.
2. Establish short-context conditional competence and partner sensitivity on
   development examples. If absent, diagnose representation/objective before
   scaling H; more long-context measurements cannot establish the mechanism.
3. Measure the same-checkpoint K intervention and D/E on a development panel.
4. Freeze the task, training recipe, checkpoints, evaluation panel, and cost
   accounting. Run the final matched long-context tests without outcome selection.
5. A positive paper needs a reproducible accuracy-cost improvement attributable
   to the tested mechanism, with uncertainty and adverse cases retained. If the
   gain fails, report a bounded failure of this implementation; do not replace
   the intended claim with a calibration-only success or oracle-only result.

## Manuscript organization

Main text: hypothesis; cost/error tradeoff; one task and RWKV implementation;
one primary K intervention; accuracy-versus-cost and length results; limitations.
Use the exact decomposition as an established diagnostic tool, not a novelty
claim. Attention-to-recurrence derivations, broad architecture comparisons,
F2 capability/long-context benchmarks, repeated development failures, and
antithetic-training details belong in supporting material. Keep a concise
main-text disclosure of any failure that materially limits the central claim.

## Current execution and resource constraints

All three frozen fresh-seed parents completed. Six independent/complementary
branches were already submitted; these test a supporting posterior-learning
question, not the long-context thesis above. Let them finish and report them
honestly. No main-text positive claim is licensed by those pending results.
The user authorized32 H100s on the original project plus16 on
project-632c8db8-4530-413a-ada5-df91774a7e09. Fastest completion takes precedence
over delaying the frozen experiment to enforce80% utilization. New experiments
must share that total cap; queued jobs count. The discarded RWKV and attention GPU qualifications passed; the fixed RWKV
600-update development run is training. No learned endpoint result from the
new task is available. Qualification, training, exact
evaluation, and validation are prepared; their completion must be checked from
actual receipts. The focused draft is Long_RWKV_ICLR2027.tex; the F2 manuscript
remains the supporting study.
