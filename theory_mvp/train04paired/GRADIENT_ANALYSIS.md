# What complementary histories change

This is an elementary antithetic-sampling calculation, not a new general variance theorem. Fix the public condition and reveal round. Let U be uniform over the free-bit assignments, let J(U) complement all free bits, and let g(U) be the per-history gradient at fixed parameters. J is a measure-preserving involution. Define

- g_even(U) = [g(U) + g(J(U))]/2;
- g_odd(U) = [g(U) - g(J(U))]/2.

Then E g_odd = 0, E g_even = E g, and Cov(g_even,g_odd) = 0. The last statement follows by replacing U with J(U): the centered even component is unchanged and the odd component changes sign. Consequently

Cov(g) = Cov(g_even) + Cov(g_odd).

For two independent histories with this same public condition, the averaged-gradient covariance is Cov(g)/2. For the complementary pair, the averaged gradient is exactly g_even(U), with covariance Cov(g_even). Their difference is

Cov(complementary average) - Cov(independent average)
= [Cov(g_even) - Cov(g_odd)]/2.

There is no universal positive-semidefinite ordering. Pairing helps in directions dominated by odd variation, and can hurt in directions dominated by even variation. The shared public-condition variation is unchanged between arms. The calculation assumes per-history gradients add: serial document forwards and no cross-document statistics; gradient clipping and AdamW updates are nonlinear, so equal mean raw gradients do not imply equal mean optimizer updates.

## Specific motivation for parity

In the second information-set round of the paired-parity task, complementing all revealed free bits complements all dependent target labels while preserving the public condition. For a history-independent scalar binary logit z, the two cross-entropy derivatives with respect to z are sigma(z)-y and sigma(z)-(1-y). Their mean is sigma(z)-1/2; at z=0 it is exactly zero. Pairing therefore eliminates random target imbalance in this shared-logit direction, while a representation that responds to the history can still receive a nonzero learning signal. This does not prove that the RWKV hidden states will learn the necessary interaction or that all parameter-gradient noise decreases. It motivates the controlled experiment.

## Counterexample to universal improvement

On two independent uniform sign bits s1,s2, let g=(s1*s2,s1). Complementation negates both signs. The first component is even and the second is odd. Single-history covariance is diag(1,1); independent-pair covariance is diag(1/2,1/2); complementary-pair covariance is diag(1,0). The difference has both positive and negative eigenvalues.

`tests/test_train04paired_core.py` checks the finite covariance identity and this counterexample, in addition to unbiasedness, feasibility, and matched training-batch fields. These are objective/sampler checks, not evidence of neural learning.
