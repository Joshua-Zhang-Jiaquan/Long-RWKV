# Memory, access and parallel refinement: exact MVP findings

This is an exploratory research note, not a claim that the current F2 checkpoint is a superior long-context model. The proof and novelty audits are in `THEORY_REVIEW.md` and `LINEAR_COMPRESSION_ADDENDUM.md`.

## What the first candidate establishes

A restricted online chain walker that cannot store off-path records needs

\[
K_{\to}=1+\operatorname{des}(p_1,\ldots,p_h)
\]

forward history scans. The positions p describe the order of the chain's unique edges in the read-only history. Under uniform random order, the distribution is Eulerian and the mean is (h+1)/2. If scans alternate directions starting forward, the count is one plus the initial direction mismatch plus subsequent direction changes. Its mean is (4h+1)/6 for h>=2. This is the classical longest-alternating-subsequence statistic under the appropriate convention, not a new combinatorial result.

Exact independent simulation checked all **46,233 permutations** for h=1 through 8 with zero formula or Eulerian-histogram mismatches. At h=8, mean forward cost is 4.5 directional scans; alternating scans cost 5.5. Reverse-ordered records instead take h forward scans but only two alternating scans. One off-path cached record invalidates the no-cache lower bound; the separately plotted cache control demonstrates why this assumption matters.

![Scan cost](../results/theory_mvp/scan_cost.png)

Repeated computation without new history access is a different interface. For M independent uniform V-ary records and a B-bit encoding formed before a uniform query, standard information theory gives

\[
H(Y\mid C,Q)\ge \max\{0,\log_2 V-B/M\}.
\]

The Fano converse constrains answer error regardless of decoder iterations. Exhaustive binary codebook search confirms that retaining whole records need not be optimal: for three source bits and one memory bit, the optimal random-query accuracy is 75%, whereas storing one source bit gives 66.7%. Both the converse and codebook formulation are standard rate-distortion facts.

## A stronger link between compression and parallel generation

Let Y=H be uniform on binary vectors of length N, and retain C=AH, with all operations over GF(2) and rank(A)=r. Every such encoder retains r bits and leaves conditional entropy N-r bits. Let d(A) count coordinate unit vectors in the rowspace of A; these are the individually determined output coordinates.

Given C, the exact posterior is uniform on an affine code. The product of its exact one-coordinate marginals instead has

\[
D_{\rm KL}(p(Y\mid C)\Vert \prod_i p(Y_i\mid C))=(r-d(A))\ln2,
\qquad
\operatorname{TV}=1-2^{-(r-d(A))}.
\]

**Proof.** Each coordinate is fixed if its unit vector is in the rowspace, and otherwise is uniform. The product law is uniform on 2^{N-d} vectors. The true law is uniform on its subset of size 2^{N-r}. Their likelihood ratio on the true support is 2^{r-d}; their common mass is 2^{-(r-d)}. This proves both identities. The full-history expected log loss is (N-d)ln2: it is the sum of irreducible compression loss (N-r)ln2 and the product-law error, not the posterior KL alone.

After observing some coordinates, let U be unknown coordinates, J a proposed reveal batch, and d_{U,J} the number of j in J whose coordinate unit vectors lie in rowspace(A_U). Then

\[
\operatorname{TC}(Y_J\mid C,Y_{U^c})
=\left[\operatorname{rank}(A_U)-\operatorname{rank}(A_{U\setminus J})-d_{U,J}\right]\ln2.
\]

The projected affine posterior has dimension |J|-rank(A_U)+rank(A_{U\setminus J}); subtracting its entropy from the sum of marginal entropies gives the expression.

Choose an information set of N-r coordinates of the nullspace code. Their conditional joint distribution is a product of fair bits. Sample them in the first round, then solve the remaining coordinates, which are deterministic, in the second. The output law is exact. One round is exact iff r=d(A). The two-round necessity is restricted to irreversible original-coordinate reveals with independent within-round draws. An unrestricted sampler can draw shared latent bits and apply a linear map in one computation; the result is not a general computation lower bound.

## Exact finite results

All **35 rank-two rowspaces on four coordinates** have the same two-bit memory and two-bit conditional entropy. Across these encoders:

| One-round forward KL | Encoders | Stored-constraint satisfaction |
|---|---:|---:|
| 0 bits | 6 | 100% |
| 1 bit | 16 | 50% |
| 2 bits | 13 | 25% |

The information-set two-round construction is exact for all 35. Independent enumeration of all compression values and feasible partial observations verifies **24,592 batch-dependence identities**, with zero residual.

![Representation and parallelism](../results/theory_mvp/representation_parallelism.png)

These results establish a useful distinction: retained information alone does not specify how difficult it is to sample the retained conditional distribution using independent parallel output heads. Representation structure matters. They do not establish a new general diffusion theorem; linear-code information identities and conditional-independence sampling have substantial prior art.

## Exact absorbing-schedule experiment

To connect the example to the paper's sampler, an exact dynamic program evaluates independent absorbing reveals with uniform survival increments for 1, 2, 4, 8, 16, 32, 64 and 128 nominal stages, for all 35 encoders. It tracks both the probability that committed coordinates still admit a valid completion and the expected path dependence cost. Irreversible commitments make any infeasible partial assignment impossible to repair. Affine symmetry makes endpoint mass uniform on valid codewords, giving forward KL = -log2(valid mass).

For the two independent parity constraints A=(0011,1100), the two coordinates in each pair collide in the same reveal stage with probability 1/T. Thus valid endpoint mass is exactly (1-1/(2T))^2, while expected path dependence is 2/T bits.

| Stages | Constraint satisfaction | Endpoint KL, bits | Path bound, bits |
|---|---:|---:|---:|
| 1 | 25.0000% | 2.000000 | 2.000000 |
| 8 | 87.8906% | 0.186219 | 0.250000 |
| 64 | 98.4436% | 0.022631 | 0.031250 |

The information-set construction uses two rounds and has zero error. A systematic encoder retaining two coordinates also has zero product-law error even in one round. These are exact predictions about oracle posteriors and named schedules. They do not establish wall-time advantages, learned-model accuracy, or a novel sampler.

![Schedule and representation](../results/theory_mvp/reveal_schedule.png)

## Higher-order dependence control

For a six-bit source with encoder rows 001111 and 110011, every conditional two-coordinate marginal is independent, yet the full posterior has two bits of total correlation. Independent simultaneous reveal satisfies the stored constraints only 25% of the time (TV 0.75). Exact enumeration verifies all 15 pairs. This is a classical limited-independence construction showing that pairwise screening alone cannot certify safe parallel reveals; it does not establish a failure of a named existing sampler without checking its actual tests.

## Novelty and implementation limits

The scan identities, random-access bound, and linear-code construction are correct explanatory results with classical ingredients. The promising research target is a **certified method or sharp tradeoff for approximate recurrent representations under explicit storage and computation limits**. The current oracle encoder and exact conditional marginals do not supply such a method, and cannot be identified with F2's learned states.

For the original deterministic full-history lookup tasks, the true answer posterior is a point mass and its conditional total correlation is zero. Those tasks cannot validate a positive dependence penalty. Compression in this controlled example deliberately creates a nontrivial posterior, which makes the dependence term testable.

The frozen-model natural-language pilot tests whether a viable empirical interface exists for further work. All 480 records are now audited. F2 scores 22/120 one-hop and 1/120 two-hop; causal RWKV scores 63/120 and 29/120. All cells fail the fixed 96/120 gate, so no long-context expansion is authorized by the protocol. See `MVP_REPORT.md` for diagnostics, resource use, and the research decision.

## Reproduction

```sh
python -m lrwkv_evidence.theory_mvp.exact
python -m lrwkv_evidence.theory_mvp.linear_compression
python -m lrwkv_evidence.theory_mvp.reveal_schedule
python -m pytest -q tests/test_theory_mvp_exact.py tests/test_theory_mvp_linear.py
```

Results and standalone PDF/PNG figures live in `results/theory_mvp/`. The full classical-prior discussion and citations are in the two accompanying audit notes.
