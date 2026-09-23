# From supervised histories to the error in a reveal policy

This is a standard change-of-measure argument specialized to the experiment,
not a claim of a new generalization theorem. It sharpens the operational
requirement: useful extra calls need accurate predictions on the histories
that those calls actually visit under the target law.

Fix a context distribution rho, or one context. A prediction query z includes
the public context, the visible coordinates and their values, the target
coordinate, and any stage/timestep input supplied to the model. Let e(z) be
the Bernoulli KL between the exact conditional target and the model marginal.
Normalize the training supervision weights into a probability law nu(z).
For a fixed reveal policy pi on N targets, choose one of its N coordinate
predictions uniformly and use target-distributed histories; this defines
mu_pi(z). Then

    E_pi = N E_{mu_pi}[e(z)].

If mu_pi is dominated by nu and d mu_pi / d nu <= kappa, nonnegativity gives

    E_pi <= N kappa E_nu[e(z)].

The rightmost expectation is excess conditional cross-entropy above the exact
target-entropy floor, not raw cross-entropy. The proof is substitution of the
density ratio followed by its upper bound. The context law must be the same
on both sides: an in-sample training loss does not certify unseen keys or long
contexts. Held-out risk and its uncertainty are needed for that interpretation.

In the new task's balanced two-round history objective, the loss averages one
of two four-token groups, with scale 1/4 per selected token. Therefore its
population excess risk is exactly E_info/8. Mixing it with ordinary denoising
with probability alpha gives

    E_info <= (8/alpha) * excess_mixture_risk.

For alpha=1/2 the multiplier is16. This bound is over the stated context law;
a dependent-family guarantee requires conditioning that law on the dependent
family, or including the family's sampling probability in the multiplier.
It also assumes the same stage input as evaluation. It does not turn the
current logged minibatch losses into test-set certificates.

There is no finite uniform bound without coverage. For the four-pair target,
set every prediction equal to the oracle on the empty and first-four-visible
queries used by the information-set policy. Its excess risk is zero. On the
single-bit-visible query of the sequential policy, assign probability epsilon
to the next bit being one, although that bit is fair under the target. Leave
all other queries at their oracle values. The information-set KL is zero but
the sequential KL contains

    KL(Bernoulli(1/2) || Bernoulli(epsilon))
      = -log(2) - 1/2 log(epsilon(1-epsilon)),

which diverges as epsilon tends to zero. Both policies have D=0. Thus even
perfect two-round policy supervision need not support eight-call generation.
An ordinary-denoising mixture can supply coverage, but rare histories can
still make kappa too large for a useful bound. Earlier parent training also
does not guarantee preservation under later policy-specific fine-tuning.

This argument motivates the same-call partition control and mandatory reporting
of the sequential policy. The seed71 observation (low two-call KL, high
sequential KL) is compatible with this mechanism; it does not prove that a
specific internal feature or optimizer effect caused the failure.

## Exact coverage constants for the implemented mask law

For the generalized N-stage absorbing objective, t is uniform in {1,...,N},
each coordinate is masked with probability p=t/N, and each masked coordinate
has loss weight 1/t. Its normalized query weight for a particular mask M and
coordinate i in M is p^|M| (1-p)^(N-|M|)/(N t). The target history distribution
cancels from the density ratio because both objectives use target-distributed
visible values at the same context. Evaluation sets stage t=|M|. A policy
query has coordinate weight 1/N, so its ratio is

    t / [(t/N)^t (1-t/N)^(N-t)].

For a balanced two-round policy with even N>=2, the sharp maximum is
(N/2) 2^N, attained in the second round; the first-round ratio is N. Thus
ordinary denoising covers the required histories, but the worst-case risk
conversion constant can grow exponentially in the answer length. This is a
coverage bound, not an exponential lower bound on actual training sample
complexity or a claim about increasing context length H.

For the N=8 objective with the canonical first-four information set, the exact constants are:

- information-set or sequential policy under ordinary masking: 1024;
- information-set policy under the equal ordinary/history mixture:
  2048/1025 = 1.99804878 (less than the coarse bound 2);
- canonical-order sequential policy under that mixture: 1988.410785;
- sequential policy after permuting the information set away from the first four coordinates: up to2048, attained by the last-four information set;
- pair-preserving two-call policy under that mixture: 2048.

These are density ratios kappa; multiply by N to bound total endpoint E from
normalized excess risk. They quantify why supervising one policy is different
from merely increasing call count. They do not establish that the observed
errors attain the worst case. `mask_coverage.py` enumerates all masks and
coordinate weights; tests check normalization, sharpness and the uncovered
history counterexample. Constants apply to the stated objective and stage
encoding, not automatically to the earlier frozen training campaign.
