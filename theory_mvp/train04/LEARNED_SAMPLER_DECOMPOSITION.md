# Connecting the exact posterior theory to a trained denoiser

This is an application of KL chain rules and conditional total correlation, **not a new-theorem claim**. Closely related exact-oracle adaptive cost identities and model diagnostics already appear in [Wen, Liang and Lu, *Conditional Total Correlation and the Serial Depth of Adaptive Parallel Sampling*](https://arxiv.org/abs/2608.25505). Dependence-adaptive diffusion schedules are also studied in [Zhao and Cai, *Adaptation to Intrinsic Dependence in Diffusion Language Models*](https://arxiv.org/abs/2602.20126). Our contribution here is a falsifiable controlled application with an exactly known target posterior and a trained tied RWKV.

## Fixed reveal policies: exact dependence-plus-estimation accounting

Fix a public condition c=(A,C). Let P be the uniform conditional target distribution and let J1,...,JK be a fixed ordered partition of output coordinates. At round k the neural denoiser returns binary distributions q_i(. | y_S,c), where S is the union of earlier groups. The implemented sampler independently draws the new coordinates, so its endpoint distribution is

    Q(y|c) = product_k product_{i in Jk} q_i(y_i | y_S,c).

Assume q_i is positive wherever P assigns positive probability. By the chain rule,

    KL(P || Q)
      = sum_k E_{Y_S~P} KL(P(Y_Jk | Y_S,c) || product_i q_i(.|Y_S,c))
      = sum_k E_P TC_P(Y_Jk | Y_S,c)
        + sum_k sum_{i in Jk} E_P KL(P(Y_i | Y_S,c) || q_i(.|Y_S,c)).

Proof of the second equality: add and subtract the logarithm of the product of the true conditional marginals inside each block KL. The first part is conditional total correlation; marginalizing the second part over the other coordinates gives the sum of marginal KL terms. Every expectation uses target histories distributed according to P, not self-generated histories distributed according to Q.

Call the two nonnegative terms D_pi (dependence discarded by the policy) and E_pi (denoiser estimation error under that policy). The equality is **KL(P||Q_pi)=D_pi+E_pi**. Therefore a policy with lower D need not perform better after training: it may visit histories where its denoiser error is larger.

For an information-set two-round policy, D_info=0. Its advantage over a matched random-halves policy is characterized exactly by

    KL(P||Q_random) - KL(P||Q_info)
      = D_random + E_random - E_info.

It improves forward KL if and only if E_info-E_random < D_random. This is the meaningful learned-model test. Merely showing that information-set oracle sampling is exact does not establish this inequality for the trained RWKV.

## Exact accounting in the eight-bit experiment

For systematic constraints all remaining random coordinates are mutually independent, so D_pi=0 for every fixed policy in the experiment. Any nonzero endpoint KL is estimation error.

For four independent parity pairs, a pair incurs ln(2) nats exactly when both of its coordinates are first revealed in the same group. If the coordinates are split across groups, the later coordinate is determined by the earlier one and the public parity, and the pair incurs no dependence cost. Consequently:

- One-round sampling: D=4ln(2).
- Information-set two rounds: D=0.
- Sequential sampling: D=0.
- Fixed random halves: D is ln(2) times the number of pairs contained entirely within one half.

The evaluator enumerates all16valid endpoints for each fixed-policy audit condition and computes neural Q using stable log-softmax values. Thus endpoint KL is evaluated directly; subtracting the independently computed D gives E for exactly those audited conditions. Averaging the decomposition must use the same condition set on both sides. Tiny negative residuals from floating-point summation may be tolerated only within a declared numerical tolerance, never silently clipped to hide substantive inconsistency.

## Randomized reveal paths

For a Bernoulli reveal schedule, introduce its random reveal path Z. Compare the target joint distribution P(Y)pi(Z|Y) to the sampler joint Q(Y,Z), using the same policy transition law in both processes. The KL chain rule decomposes path KL into target-averaged dependence and estimation terms. Marginalizing Z gives

    KL(P_Y || Q_Y) <= KL(P_{Y,Z} || Q_{Y,Z}) = D_path + E_path.

The difference is E_{Y~P} KL(P(Z|Y)||Q(Z|Y)). A sampled path probability is therefore not generally the endpoint probability. The present neural evaluator deliberately does not report endpoint joint KL for Bernoulli schedules; it reports validity, diversity, calibration and actual network calls. Exact oracle Bernoulli endpoint values remain available analytically for the independent-pair family.

## What calibration does and does not certify

The separate forward-corruption calibration panel estimates denoiser error under the training corruption distribution. It measures whether the model learned deterministic conditional bits and fair-bit uncertainty. It is not automatically E_pi for a chosen decoder, because the decoder can weight masks/histories differently. The fixed-policy endpoint enumeration supplies the schedule-specific error accounting on the small audit panel without assuming these distributions coincide.

For the information-set sampler, Pinsker yields TV(P,Q_info) <= sqrt(E_info/2), with natural logarithms. Since P is supported on valid codewords, invalid mass is at most that TV bound. Conversely, high validity alone does not upper-bound KL or certify diversity: a generator concentrated on one valid vector can have perfect validity and arbitrarily bad forward KL.

The trained experiment can corroborate this mechanism on held-out labeled structures. It does not yet supply a learned dependence detector, a storage-limited representation theorem, general natural-language performance, or a new complexity separation.

A useful falsification control is a denoiser that always predicts a fair bit. Its endpoint law is uniform on all256vectors for every fixed schedule, so every schedule has KL=4ln(2) against the16-element target coset and validity1/16. Information-set scheduling removes dependence cost but cannot compensate for this denoiser's missing conditional competence. Conversely, predicting a deterministic bit with probability only0.5001 can have perfect thresholded accuracy while remaining a poor sampler; proper log-loss/KL is necessary alongside accuracy.

## A sharper validity certificate and the remaining diversity error

Let S be the valid affine coset and v=Q(S)>0. Because P is supported on S, the ordinary KL chain rule also gives

    KL(P||Q) = -log(v) + KL(P || Q(.|S)).

The first term is invalid-output mass expressed as log loss; the second is mismatch among valid outputs. Hence v >= exp(-KL(P||Q)), and an information-set policy satisfies v >= exp(-E_info). This validity certificate is sharper than the generic Pinsker statement for this support event. For example, a genuinely established E_info<=0.1nats certifies at least exp(-0.1), or90.48%, valid mass for the same audited condition/distribution. This is an analytic implication, not a claim that the trained checkpoint meets that bound. Averaging over the same condition distribution preserves the lower bound exp(-meanKL) by Jensen's inequality.

Perfect validity sets only the first term to zero and leaves conditional diversity error unconstrained. Conversely, empirical valid fractions from32draws should not be inserted into this identity as if they were exact Q(S). The present exact-joint audit stores KL, while its separate sampling panel estimates validity and coverage; it does not recover an exact validity/diversity decomposition for the learned endpoint distribution.

Numerical qualification update: a post-primary audit additionally enumerated endpoint valid mass on the full condition panel. Cross-execution KL differences up to0.00970nats were detected, so tiny policy contrasts remain provisional pending repeated-canvas/precision tests. Exact finite support is not itself proof of reproducible neural probabilities.

## Training on the histories used by the policy

This is a proposed development extension, not a completed experiment or a novelty claim. For a fixed policy pi, draw Y from the target conditional P and draw a round k with positive probability rho_k. Construct exactly the same canvas, noise-stage input, and visible history as decoder round k. With exact conditional marginals p_i, define the soft-label objective

    L_pi(theta) = E_{Y,k} [ (1/(N rho_k)) sum_{i in J_k} CE(p_i(.|Y_S,c), q_theta,i(.|Y_S,c)) ].

Its excess over the oracle conditional entropy is exactly E_pi/N. This follows by CE=H+KL and cancellation of rho_k when averaging k. Consequently, for the information-set policy, forward endpoint KL equals N times this population excess loss, and its valid mass is at least exp(-N times excess). These are population identities for the same condition distribution, not finite-sample certificates. For randomized policies the corresponding claim concerns path KL and bounds endpoint KL from above.

The model inputs must match inference, including stage codes; matching the visible subset while using a different time code does not suffice. Histories for this identity come from P. Self-generated Q histories optimize a different objective. The exact synthetic oracle is privileged supervision; access to it is part of the experimental scope.

Why the original corruption loss can be a weak proxy: at N=8, the second information-set round masks one specified set of four coordinates and uses stage4. Under the current training law, stage4 occurs with probability1/8, and that exact mask occurs with probability1/256. The normalized absorbing weight is8/(4*8)=1/4. Hence this particular second-round marginal-error sum enters the population training excess with coefficient1/8192. The first round at stage8 contributes at least its selected-coordinate error sum with coefficient1/64. Nonnegativity gives E_info <=8192 times the population corruption excess. This is a conservative bound, and generally too weak for useful certification. It illustrates why low aggregate corruption loss alone is insufficient to validate a scheduler. It does not prove the observed pilot failure was caused by mask coverage.

A next ablation, if the current curriculum establishes conditional competence, is a mixture of ordinary corruption and these policy-history examples. Compare the same mixture across hard and conditional soft targets; preserve standard-corruption evaluation and both decoding policies. Freeze this development change before any new confirmatory evaluation. Do not retrospectively change the currently submitted matched curriculum arms.

## A mixture with an explicit coverage guarantee

Let alpha in(0,1] and define L_mix=(1-alpha)L_corruption+alpha L_pi. Define the oracle baseline by the identical mixture of the two conditional-entropy objectives. Nonnegative marginal KL gives

    L_mix - H_mix >= alpha * E_pi/N.

For the information-set policy, therefore,

    KL(P||Q_info) <= (N/alpha)(L_mix-H_mix),
    Q_info(valid) >= exp[-(N/alpha)(L_mix-H_mix)].

At N8 and alpha1/2 the coefficient is16 rather than the conservative8192coverage coefficient for the existing unmodified corruption law. This improves the population guarantee supplied by the objective; it does not assert that optimizing the mixture attains a particular loss, that finite validation data certifies it, or that different objectives' raw loss values can be compared without subtracting their respective entropy baselines. For a fixed condition the validity statement holds directly. With a condition average, Jensen yields the corresponding lower bound on mean validity.

An unbiased implementation draws the branch with probabilities1-alpha and alpha. In the policy branch draw k with rho_k and weight the selected group's marginal cross-entropies by1/(N*rho_k). The actual decoder canvas masks every unrevealed coordinate, while loss is applied only to the group revealed in this round. In particular, training all masked coordinates at the first policy round would be a different objective. Use public A to choose an information set and target Y only to construct a teacher-forced history. At test time do not reveal target values or replace neural outputs with solved bits.

A matched ablation can compare ordinary corruption against this50/50mixture, holding curriculum, update count, examples, precision and initialization fixed and distinguishing hard from soft supervision. This candidate was activated on2026-09-22 through theory_mvp/train04mix/PROTOCOL.json and immutable stage25ce57244ae9a384. Both label arms restore their own completed300-update parent model and optimizer, then train onN4through update700. The standard controls remain unchanged. Results are pending; this is development, not a confirmatory test.
