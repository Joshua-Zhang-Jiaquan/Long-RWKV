# Manuscript revision map (results pending)

The reviewed19-page paper remains the historical baseline. The new manuscript
will distinguish established accounting from an empirical prediction test.
No transfer result is currently claimed.

## Main contribution to test

A recurrent denoiser makes full-context scans with cost linear in context length.
Diffusion provides repeated opportunities to condition on newly revealed answer
coordinates. The question is when an additional scan buys useful joint fidelity,
and how to select its reveal schedule when the conditionals are imperfect.

The earlier intervention establishes an exposure-dependent positional effect
within one selected lineage. The new experiment asks a stronger question:
can conditional-error calibration on source dependency codes choose between
zero-factorization-loss, equal-call schedules on unseen code classes?

## Minimal theory section

Retain the fixed-partition identity KL(p||q_pi)=D_pi+E_pi with appropriate prior
attribution. Present the proposed schedule selector as an empirical approximation
to E_pi, not a new KL decomposition. For invertible B in GF(2)^(4x4), the two bases
U and P=BU xor s are uniform and independent within each basis, so both two-round
policies have D=0. Their quality difference therefore isolates learned error.

A standard deterministic stability observation clarifies the role of prediction:
if |E_pi-Ehat_pi|<=epsilon_pi for both candidates, selection by Ehat has regret
at most epsilon_selected+epsilon_optimal. This is not a guarantee that the
source-fitted predictor transfers. Disjoint source/test context families have
infinite full-context density ratio; the paper must not invoke its earlier
finite-coverage bound as cross-family justification.

The two directions require different parity computations. Their public features
are parity fan-in and number of displayed equations combined. Difficulty is a
hypothesis to measure, not a monotonicity theorem.

## Main evidence layout

1. Architecture/sampler schematic: fixed context; eight labeled answer slots;
   tied bidirectional recurrent scan; public A/B bases; two reveal rounds.
2. Prospective test:15 source code classes,10 held-out classes, independent fresh
   task-training lineages, sealed predictions. Source calibration has180 cells
   per model; held-out evaluation has240. Report full endpoint KL and valid mass.
3. Main plot: predicted vs realized policy-error difference; selected-policy
   regret and paired improvement over dependence-only hash tie. Show every
   lineage, including failures; uncertainty resamples lineages/classes/prompts.
4. Old balanced-position result as mechanistic motivation/support. Preserve
   existing failed training, changed-interface and distractor-order outcomes.
5. Restricted quality-cost certificate in supporting material. Do not describe
   it as a practical architecture frontier without matched competent causal and
   optimized attention comparisons.

The abstract will be rewritten only after the prospective gates are known.
If the study fails, state the failure and retain the restricted existing claim;
additional compute does not turn a failed predictive test into a positive result.

## Reproducibility and editorial work

Provide the public-base training path for all six new lineages. For the earlier
selected checkpoint, document both parent1500 and independent1000 additional
updates; adaptations alone are not a reconstruction recipe. Prepare a mapping
from available original checkpoint hashes to any inference-only exports.

Move budget-project identifiers, scheduler status narrative, reservation limits,
and utilization targets into operations records. Preserve total H100-hours,
precision, token counts, failure accounting and measured runtime in the paper.
