# What balanced policy supervision predicts

This note derives the normalization of the already frozen training objective; it changes no experiment, selector, threshold, or analysis. The comparison concerns a common public-context law. It is not a generalization guarantee from the training structure catalog to the held-out catalog.

For a fixed public context c and a two-round policy pi, the implemented loss samples the round uniformly, samples its visible history from the target posterior, and uses (2/8) times the sum of conditional cross-entropies on that round's four revealed coordinates. Consequently its population excess risk above oracle entropy is E_pi(c)/8. Drawing one history for each of two policies and averaging their losses has population excess risk

R_mix(c) = [E_info(c) + E_pair(c)] / 16.

The histories can be independently sampled; independence changes estimator variance but not this expectation. The stage input and visible-coordinate set remain part of the conditioning query. On paired parity, the balanced-policy oracle entropy floor is (3/4)ln2. It is (1/2)ln2 on the systematic family. Equal family weighting gives the overall population floor (5/8)ln2. A minibatch with randomly selected rounds need not have this exact floor, so the worker logs each batch's actual oracle entropy separately.

For paired parity, KL_pair - KL_info = 4ln2 + E_pair - E_info. Nonnegativity therefore yields the operational bound

abs[(KL_pair - KL_info) - 4ln2] <= 16 R_mix.

This gives a quantitative meaning to controlling the estimation-error confound in a same-call comparison. Balanced supervision alone does not remove that confound: the relevant held-out conditional risk must actually be small. Also E_info <= 16 R_mix, so R_mix < ln2/4 is sufficient for the information-set model to beat every one-call product predictor on that same paired-task context law. This is a sufficient threshold, not a necessary one.

If risk is instead averaged over both equally weighted task families and three equally weighted positions, nonnegativity gives E_info for the paired-family position-average <=32 R_all, and E_info at a specified paired-family position <=96 R_all. These constants apply only when the risk and endpoint use the same underlying public-example distribution. Training at near positions alone provides no uniform change-of-measure bound on far-position queries: their public contexts are absent from that training measure. Neither version converts observed training minibatch loss into a held-out or longer-length guarantee.

The exact evaluation already reports E for both policies, so it can quantify the residual confound directly. The equal-call dependence reduction remains fixed at4ln2; the independently manipulated training distance tests whether conditional error, especially in the second round, decreases. The novelty sought is the measured predictive mechanism and successful intervention, not this elementary risk identity.
