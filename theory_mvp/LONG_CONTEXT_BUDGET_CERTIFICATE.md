# A testable cost–accuracy certificate for recurrent diffusion

This is a proposed main-paper argument, conditional on measurements that are
not yet available. It uses the established chain-rule KL decomposition; neither
that identity nor recurrent diffusion is claimed as new.

## Variables and access contract

Fix a public context c of H tokens and an N-bit answer law P_c. All denoisers
retain and reread the full context. A deterministic ordered partition pi of
answer positions specifies which positions are sampled independently at each
call, conditioned on previously sampled positions. No confidence-based or
outcome-dependent choice of the partition is included in this statement.

For a model M, let C_M(pi,c) be measured full-request cost and let
Q_M,pi,c be its normalized endpoint distribution. Cost may be latency or
energy, but the experiment must choose and report its unit. Cached audit
computations are not request costs. Distinguish H, the uncorrupted context,
from N, the generated answer length.

## Exact criterion

For each partition, the KL chain rule gives

    KL(P_c || Q_M,pi,c) = D_pi(c) + E_M,pi(c),

where D is the sum of conditional total correlations inside reveal groups and
E is the sum of conditional marginal KL errors, with histories averaged under
P_c. Both are nonnegative. Therefore, for the same checkpoint,

    KL_multi < KL_one  iff  E_multi - E_one < D_one - D_multi.

This is an exact diagnostic of the tested conditional laws, not a guarantee
that optimization will learn them or that more calls always improve accuracy.

## Budget certificate (restricted policy class)

Consider a recurrent denoiser R and a softmax denoiser A. Suppose a declared
request budget B permits R's two-call partition and A's one-call partition,
but not any of A's other policies in the predeclared comparison set:

    C_R(two,c) <= B, C_A(one,c) <= B,
    B < min_{pi in tested A policies other than one} C_A(pi,c).

If D_two(c) + E_R,two(c) < D_one(c), then R has strictly smaller forward KL
than A's feasible one-call product distribution, irrespective of A's marginal
estimation error. Proof: R's KL equals the left side; A's KL is at least D_one
because E_A,one is nonnegative. This also compares R with every one-call product
head on this target law. It does not compare R with correlated output heads,
unmeasured optimized attention policies, or causal samplers.

For real latency, timing variation must be reported. A budget feasibility claim
should use a predeclared timing statistic and uncertainty; three development
timings are only a screen. Do not choose B after seeing final test gains.

The asymptotic motivation is that, at fixed widths and layer count, a recurrent
full scan has O(H+N) mixing arithmetic and dense softmax has O((H+N)^2).
Repeated scans multiply these costs by the executed call count. Projection and
head work, bidirectional scan constants, fused kernels, precision, and batching
can dominate the observed range. Thus asymptotics motivate the certificate;
measured costs decide whether its budget conditions hold.

## Concrete eight-bit prediction

The dependent lookup task gives four public bits v and draws a uniformly random
four-bit U. The answer is Y=(U,U XOR v). The evidence location changes with H,
but the target law conditioned on v is unchanged. For one simultaneous reveal,

    D_one = 4 log(2).

For the fixed partition (first four bits, last four bits), D_two=0: the first
four are independent fair bits and the last four are deterministic given them
and v. If the eight relevant conditional marginal KL errors average at most
epsilon(H), then

    KL_R,two <= 8 epsilon(H),
    valid_mass_R,two >= exp(-8 epsilon(H)).

Hence epsilon(H) < log(2)/2 certifies a strict KL advantage over every one-call
product head. The stronger development gate KL<0.1 implies valid mass>exp(-0.1)
but the actual valid mass is computed separately. These are population endpoint
statements on the finite known posterior, not downstream language benchmarks.

For proof of the validity bound, let S be the valid support and V=Q(S). On S,
Q(y)=V Q(y|S), so KL(P||Q)=-log V+KL(P||Q(.|S)) >= -log V.
High validity alone is insufficient: a sampler may collapse to one valid answer.
The within-valid KL distinguishes collapse from learning the uniform law.

The retrieval control has one deterministic answer and D_pi=0 for all pi.
Its failures therefore identify estimation/retrieval problems, and improvement
there cannot by itself verify reduction of simultaneous-reveal dependence.

## What makes this a long-context test

Padding the prompt does not change D_one or D_two. The quantity at risk is
epsilon(H,location), reflecting extraction and conditional use of evidence.
Measure it on the same base problems across evidence-only, 1K, 4K, and 16K,
with near/middle/far placements. A positive result needs the cost conditions
and conditional-error certificate to hold at long lengths, together with
observed exact valid mass and diversity. Report where either condition fails.

This theory does not assert that diffusion improves memory retention. Nor does
it assert an accuracy or speed advantage over cached causal RWKV: sequential
causal factorization also has D=0 and its prefix can be cached. That comparison
must be learned, timed, and reported as a practical baseline with a different
training/access pattern.

## Required evidence and falsifiers

1. Same-checkpoint one/two/random-partition/sequential endpoint audits test the
   D/E criterion without confounding it with different pretrained models.
2. Retrieval controls and paired evidence positions show whether distant facts
   are used; a failed evidence-only gate blocks a retention interpretation.
3. A trained softmax denoiser tests practical cost/accuracy. Native pretrained
   comparators have different pretraining and tokenization; document those
   differences and do not label their performance gap architecture-causal.
4. Cached causal RWKV tests whether the proposed operating point is useful
   against the natural recurrent alternative.
5. If E_two stays above D_one, or measured costs provide no useful budget band,
   the proposed mechanism fails for this implementation. More theory cannot
   substitute for either missing condition.
