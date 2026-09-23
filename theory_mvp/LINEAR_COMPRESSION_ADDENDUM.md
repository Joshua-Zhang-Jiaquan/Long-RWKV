# Addendum: linear compression and parallel posterior sampling

Audit date: 2026-09-21. Verdict: **the proposed identities are valid with the qualifications below**. This is a sharper exact example connecting information loss and parallel generation than the restricted pointer walker. It does not yet establish a novel standalone theorem: entropy/rank identities, information-set encoding, and code-based dependence constructions are established tools. The useful statement is that equal retained Shannon information does not determine the error of independently sampled output coordinates.

## Setup and exact identities

Let X be uniform on GF(2)^N, Y=X, and C=AX, where A is a fixed known binary matrix of rank r. Use natural logarithms. Conditional on a feasible syndrome c, Y is uniform on an affine coset of ker A, of dimension k=N-r. Thus H(Y|C)=k ln 2. The full-history posterior is a point mass, so the minimum expected full-history-to-compressed-model KL is exactly k ln 2 for every rank-r encoder.

Let d(A) be the number of standard coordinate vectors e_j in row(A). Coordinate j is deterministic given c iff e_j belongs to row(A); otherwise it is a fair bit. The product of exact coordinate marginals q_c therefore has entropy (N-d) ln 2 and is uniform on a coordinate face containing the true coset. Consequently

- KL(p(Y|c) || q_c) = TC(Y|c) = (r-d) ln 2.
- TV(p(Y|c),q_c) = 1 - 2^{-(r-d)}.
- q_c assigns probability 2^{-(r-d)} to satisfying all syndrome constraints.
- Expected full-history loss against q_C is (N-d) ln 2 = (N-r) ln 2 + (r-d) ln 2.

The TV formula follows because the true coset has 2^{N-r} points and the product law's face has 2^{N-d} points. On the true support p>=q. Reverse KL(q||p) is infinite when r>d, because q puts positive mass outside the true coset. Keep the KL direction explicit.

This cleanly distinguishes **irrecoverable information loss** k ln 2 from **avoidable product-sampling mismatch** (r-d) ln 2. Sampling the correct compressed posterior still does not reconstruct the original history: it has residual conditional entropy k ln 2. Reporting zero compressed-posterior KL as zero reconstruction error would be false.

For example, with N=4,r=2, retaining coordinates X_1,X_2 gives d=2 and zero product mismatch. Retaining parities X_1+X_2 and X_3+X_4 gives d=0, mismatch 2 ln 2, TV 3/4, and only 1/4 probability that a one-round independent draw obeys both constraints. Both encoders retain two bits and leave two bits of irreducible uncertainty.

## Partial visibility and a proposed batch

Let U be the unknown coordinates, with feasible visible values fixed. The remaining constraint is A_U Y_U=b. For J subset U, set r_U=rank(A_U), r_R=rank(A_{U\J}), and

`d_{U,J} = #{j in J : e_j in row(A_U)}`,

where e_j is the unit vector in the coordinate space indexed by U. Then

`H(Y_J | C, visible) = (|J| - r_U + r_R) ln 2`,

because projecting ker(A_U) onto J has dimension (|U|-r_U)-(|U\J|-r_R). Its coordinate entropies sum to (|J|-d_{U,J}) ln 2, giving

`TC(Y_J | C, visible) = (r_U-r_R-d_{U,J}) ln 2`.

The integer on the right is nonnegative. It is the exact batch posterior-to-product KL. It does not depend on the values of a feasible syndrome or visible assignment, only on A,U,J. Batch TV is likewise 1-2^{-(r_U-r_R-d_{U,J})}. J empty gives zero. For J=U the formula reduces to rank(A_U)-d(A_U).

State feasibility explicitly. If a previous incorrect product batch creates an impossible partial assignment, a true conditional posterior at that assignment is undefined. The formula applies on the true data path or to a sampler whose prefixes remain feasible; it cannot silently repair an already infeasible trajectory. Selection of J based on realized token samples can also create a different joint law; the intended construction chooses J from A and the visibility pattern before sampling values.

## One versus two coordinate-reveal rounds

Define the sampler class precisely: an output coordinate is committed once, in its original coordinate system; each round samples its selected coordinates independently from their exact current conditional marginals using independent fresh randomness; later rounds can condition on committed coordinates. The matrix and syndrome are available. Arbitrary linear algebra and oracle construction are free, and the round count measures parallel reveal depth, not runtime or neural NFE.

Choose an information set I of size k for ker A: projection onto I is a bijection from every syndrome coset to GF(2)^k. Such I exists because a full-rank generator of ker A has k independent coordinate columns. Therefore Y_I given C is exactly k independent fair bits. Sample them in one batch. Then every remaining coordinate is a deterministic function of C,Y_I; reveal these in a second batch. The resulting joint distribution is the exact posterior.

A one-round sampler in this class must generate all coordinates by the product law, so it is exact iff r=d. Hence two rounds are necessary and sufficient when r>d. When r=d one round suffices. If r=N the posterior is wholly deterministic: conventionally it takes one output-commit round but zero random-draw rounds. If N=0 it needs no output round. The empty first batch when k=0 should not be counted as a genuine oracle evaluation.

**This is not a general sampling complexity lower bound.** An unrestricted algorithm can draw k independent latent bits and apply a linear map to produce the whole affine coset sample in one invocation. Shared randomness between original-coordinate outputs, arbitrary output transformations, remasking/correction, or a latent scratch channel can escape the stated one-round restriction. Gaussian elimination has a real cost, and a neural marginal oracle need not discover the information set. A fixed absorbing Bernoulli schedule also need not choose this information set: the two-round construction is a specially selected reveal schedule, not a result for every diffusion sampler.

Exact affine posteriors have zero probabilities outside their support. If invoking a theorem stated with strictly positive probabilities, either extend that theorem to the appropriate absolute-continuity/support condition or take a stated positive-probability approximation limit. The standalone finite entropy/rank calculations above do not need strict positivity everywhere.

## Prior art and novelty boundary

Entropy functions of uniform linear codes and projections are standard rank functions. Relevant primary sources include [Abbe, *Mutual information, matroids and extremal dependencies*](https://arxiv.org/abs/1012.4755) and [*Matroidal Entropy Functions: A Quartet of Theories of Information, Matroid, Design, and Coding*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7999956/). Information-set sampling is ordinary systematic encoding of a code coset.

The relationship between code structure and total correlation is also explicit prior art: [*Maximizing Multivariate Information with Error-Correcting Codes*](https://arxiv.org/abs/1811.10839) studies multivariate dependence measures through coding and matroids. Therefore an affine-code total-correlation identity should not be advertised as a newly discovered coding or information theorem.

Conditional independence as the obstacle to parallel masked sampling is already central to [Azangulov et al., *Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing*](https://arxiv.org/abs/2510.21961), published at ICLR 2026. This review establishes topical overlap, not that its method is equivalent to the proposed rank-aware schedule or that it fails on these examples; that would require inspecting and applying its exact testing/selection procedure.

**What is useful here:** an exactly measurable decomposition at fixed compression rank; a complete finite enumeration across encoders; a coordinate-aware, oracle-exact two-round construction; and an explicit demonstration that retained bits and independent-generation error are different quantities. Present these as propositions and controlled illustrations derived from classical linear-code facts. A broader novelty claim needs a more specific contribution than these facts alone.

## A meaningful extension to test, without overstating novelty

Pairwise independence does not imply a zero batch dependence penalty. If row(A) has minimum nonzero Hamming weight at least 3, no one- or two-coordinate subset supports a nonzero constraint. Conditional on a syndrome, every coordinate pair is independent and fair, yet the whole output has TC=r ln 2 and product-sampler TV=1-2^{-r}. More generally, minimum nonzero row-space weight t+1 yields t-wise independent coordinates while leaving global constraints. This is the classical linear-code construction of limited independence.

This gives an exact stress test for any proposed **pairwise-only** parallel-selection certificate: it can see no pairwise dependence while accepting a strongly dependent batch. Do not attribute that failure to a named modern sampler without verifying whether it tests higher-order or sequentially conditioned dependence. The practical research opportunity is a certified or efficiently approximated batch-selection criterion that controls the actual higher-order dependence term under computational and oracle-query budgets. The rank formula supplies a complete ground-truth laboratory, but the general algorithmic improvement and its novelty remain to be established.

Recommended CPU checks: enumerate all 35 rank-two row spaces for N=4; verify H(Y|C), d, KL and TV by direct distributions; enumerate all feasible visibility masks and J subsets; verify rank batch identities; verify information-set two-round joint probabilities; include parity examples with zero pairwise mutual information and nonzero TC. Count distinct row spaces rather than matrices, since changing a row basis does not change the encoder's information. Do not present these finite checks as proofs of a learned-model benefit.
