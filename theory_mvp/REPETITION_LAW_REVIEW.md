# Repetition-law proof audit

Reviewed the entropy-profile refinement and repetition-law proposition in the
active manuscript. The finite formulas, fixed-N expansion, and N=T² example
are consistent with their stated assumptions. Added the proportional-regime
limits below, with a dominated-convergence proof. This is a specialized
calculation within the cited entropy-profile framework, without a novelty claim.

## Required assumptions

The target is uniform on the two constant binary strings, N≥2. Reveal times
are coordinate-independent and uniform on T stages. A batch samples from the
product of exact conditional marginals. Revealed values are never revised.
Off-support conditional distributions may be any normalized extension.

Only the first nonempty batch has conditional dependence under the data law.
If its size is R, its cost is (R−1)ln2 and its independent draws agree with
probability 2^(1−R). An inconsistent revealed prefix cannot reach either valid
endpoint under irreversibility. On a consistent prefix, all later exact
conditional draws are deterministic. Both valid endpoints consequently have
mass v/2, where v=E[2^(1−R)], so endpoint KL is exactly −ln v. This explains why
the arbitrary off-support extension does not alter the endpoint calculation.

## Proportional regime

For N,T→∞ with N/T→λ in (0,∞), the path dependence cost tends to

    (λ/(1−exp(−λ)) − 1) ln2,

and endpoint KL tends to

    ln((1+exp(λ/2))/2).

For the path sum, (1−j/T)^(N−1) converges to exp(−λj), dominated by
exp(−λj/2) for all sufficiently large T after extension by zero. Reversing the
validity-sum index gives a difference converging to
exp(−λ(k+1/2))−exp(−λ(k+1)), also with geometric domination. Summing yields
v→2/(1+exp(λ/2)). These arguments hold for fixed positive λ. They do not assert
a uniform remainder for arbitrary joint scalings.

This shows nonvanishing error along fixed positive target-to-stage ratios for
this particular independently scheduled law, even with exact marginals. It
does not show that all samplers require this many calls: one chosen coordinate
followed by all remaining coordinates is exact in two calls on the same law.
Nor does it bound the trained RWKV's estimation error or replace its N8 task.

## Verification and limits

The script `theory_mvp/repetition_mesh_sharpness.py` passed 20 exhaustive reveal
schedule enumerations and retained the fixed-N and large-error checks. Added
15 finite formula evaluations at λ=0.125,0.5,1,2,8 and T=128,1024,8192. At
T=8192, all path-limit differences are below 1e-4 and endpoint-limit differences
below 2e-4. Numerical convergence examples are checks, not proofs of a limit.
Results are in `results/theory_mvp/repetition_mesh_sharpness.json`.

The manuscript's three-pass LaTeX build passed after the addition. This review
covers the named propositions only; it is not a completion certificate for the
entire theory, empirical campaign, or paper.
