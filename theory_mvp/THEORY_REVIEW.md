# Theory MVP: proof and novelty audit

Audit date: 2026-09-21. This is a separate research note, not a revision of the primary manuscript. Sources below are primary papers; the literature search is targeted rather than exhaustive.

## Decision

**Go as an exact, restricted access-model illustration and a CPU verification target. No-go as a new general RWKV/diffusion lower bound or, on current evidence, a standalone novel theory contribution.** The forward-scan statistic is the classical descent count; the alternating-scan statistic is a classical longest-alternating-subsequence statistic. The post-encoding memory bound is a standard random-access information bound. Combining them supplies a useful explanation of access contracts, but does not make their mathematical ingredients new.

A learned denoiser is not the restricted walker. The exact scan identities below do not cover arbitrary fixed-memory algorithms, bidirectional neural layers, full-canvas diffusion, caches, or attention. Their relevance must be stated as an analogy or a separately implemented algorithm, not an architectural lower bound.

## 1. Precisely restrict the online walker

There are h distinct required edge records forming a chain. Let p_1,...,p_h be their distinct positions in the fixed stream, in chain traversal order. The start key and required hop count are known before the first counted scan. To discover hop i+1, the walker must first consume hop i. It can act on a record only when its key equals the current query. It may preserve the current chain key, hop count and scan-control state, but **no information about off-path records**: neither a literal cache nor a compressed index, summary, speculative key/value write, output tape, hidden scratch buffer, or input-dependent model-weight update.

The information restriction matters more than its implementation syntax. Allowing an arbitrary compressed summary while merely forbidding a dictionary invalidates the proposed lower bound. Keys must not disclose the chain through their naming convention, and the walker must not have the ground-truth path as auxiliary input. The result concerns successful traversal, not guessed terminal answers. Repeated edges/cycles, duplicate records, or multiple possible outgoing records require a different statement.

A scan is one complete directional traversal, charged even if the final answer is discovered before its end. Left-to-right repeated scans restart at the left boundary without consuming a reverse scan. Alternating scans reverse direction at each boundary, starting forward. These are different access schedules. If reset traversal has physical cost, charge it separately. A forward/backward pair costs two scans, not one.

### Forward-only scans

For h>=1,

`K_forward(p) = 1 + sum_{i=1}^{h-1} 1[p_{i+1}<p_i]`.

Proof: within a forward scan, consumed required positions must increase. Each descent therefore requires another scan. Greedily consume every currently requested edge when encountered; this realizes exactly the maximal contiguous increasing runs, attaining the lower bound. This is an optimum only within the restricted walker class. For h=0, K=0, not 1.

If the relative order of the h chain records is a uniformly random permutation, independent of the chain, then

`Pr[K_forward=k] = A(h,k-1)/h!`, `E[K_forward]=(h+1)/2`.

Here A(h,j) counts permutations with j descents. Thus `Pr[K_forward<=k]` is the corresponding Eulerian CDF. Distractor insertion does not change these formulas if it preserves uniform relative order of the required edges. Adaptive ordering or chain selection correlated with positions need not satisfy them. For h>=2, variance is (h+1)/12; h=1 has variance zero. These are classical permutation facts, not new probability results.

### Alternating-direction scans

Write s_i=sign(p_{i+1}-p_i). For h>=2,

`K_alt(p) = 1 + 1[s_1<0] + sum_{i=2}^{h-1} 1[s_i != s_{i-1}]`.

For h=1, K_alt=1; for h=0, K_alt=0. After the first edge, a negative initial displacement requires the first turn. Each later sign change requires exactly one further boundary turn. Greedy consumption achieves this count. Deferring a consumption cannot improve the earliest feasible time in the fixed periodic scan schedule: an earlier-informed walker can always wait and mimic any delayed strategy.

For uniform order, the first displacement is negative with probability 1/2. A sign change occurs exactly when the middle member of a consecutive triple is its minimum or maximum, which has probability 2/3. Linearity of expectation gives

`E[K_alt] = 3/2 + 2(h-2)/3 = (4h+1)/6` for h>=2.

Consequently the mean excess over forward-only is (h-2)/6. Alternating is worse on average per serial scan for h>2, yet on strictly descending positions it needs two scans while forward-only needs h. These are order-specific access effects, not an argument that bidirectional RWKV is slower or less capable.

**Novelty check:** K_alt equals the longest alternating subsequence length under the prescribed initial-descent convention (`a>b<c>...`). Local-extrema characterization gives exactly the displayed formula, including its boundary term. Stanley's paper and Houdré–Restrepo explicitly study this statistic; the latter reports the same mean (4h+1)/6. The stream-walker interpretation may be pedagogically useful, but the distributional statistic is established. [Stanley, *Longest alternating subsequences of permutations*](https://arxiv.org/abs/math/0511419); [Houdré–Restrepo, *Local extrema in random permutations and the structure of longest alternating subsequences*](https://dmtcs.episciences.org/2956).

## 2. Query-after-encoding B-bit bound

Let X_1,...,X_M be independent uniform symbols in an alphabet of size V>=2. An encoder forms C before seeing an independent uniform query Q in {1,...,M}, with at most 2^B possible memory states. Set Y=X_Q. All logarithms in this section are base two. Then

`H(Y|C,Q) >= max(0, log2(V)-B/M)`.

Proof: `H(X|C) >= H(X)-H(C) >= M log2(V)-B`. Conditional subadditivity gives `sum_j H(X_j|C) >= H(X|C)`. Uniform independent Q makes `H(Y|C,Q) = M^{-1} sum_j H(X_j|C)`. Combining these inequalities proves the claim. A randomized encoder also works when H(C)<=B; with independent public randomness R, condition on R and require the same memory bound for each R.

For decoder error e, Fano yields the necessary condition

`h2(e) + e log2(V-1) >= log2(V)-B/M`.

For the optimal decoder, invert on e in [0,1-1/V]; a negative entropy lower bound is replaced by zero. Do not divide by log2(V-1) for V=2: use the binary-entropy inverse there. This is a random-access coding converse, not a tight finite-block construction in general. Storing a subset of whole records is a useful explicit baseline, not necessarily an optimal code for every bit budget/error objective. [Nayak, *Optimal lower bounds for quantum automata and random access codes*](https://arxiv.org/abs/quant-ph/9904093) establishes a stronger setting with the familiar binary entropy memory bound; no novelty should be claimed for the classical specialization here.

Count **all information that persists from the history** in C: recurrent states across every layer/direction, cached activations, retained input/output canvas, scratch tokens, record indices, selected-logit caches, and any auxiliary transcript. Finite dimension alone does not imply B bits if real values have unbounded precision. Input-dependent pretrained/adapted weights or query-correlated side information also invalidate the simple stated model unless charged or conditioned explicitly. Nonuniform queries need a weighted argument; correlated records need their actual joint entropy.

Any number of post-query iterations using only C,Q and fresh independent randomness cannot recover additional historical information. This is data processing. Reopening the history changes the access contract, and retaining the whole input canvas means the compact-state memory premise does not apply. The fixed-state versus retained-context distinction already has concrete neural precedents, including [Jelassi et al., *Repeat After Me*](https://arxiv.org/abs/2402.01032).

## 3. Noisy selection and delta-rule claims

A conditional per-hop error bound epsilon_i gives probability of any failure at most sum_i epsilon_i. If each bound holds conditional on the entire preceding successful history, the probability all hops succeed is at least product_i(1-epsilon_i); statistical independence is not needed for this conditional formulation. Marginal per-hop bounds alone do not justify multiplying probabilities. Recoverable errors or multiple attempts require explicit transition/selection rules and work accounting. These are elementary bounds, not a new theorem target.

For unit key k, a standard delta transition I-beta kk^T has norm max(1,|1-beta|). It is nonexpansive for beta in [0,2], and has neutral directions orthogonal to k. Strict contraction in every direction does not follow. A scalar gate can create contraction while erasing useful old information; vector gates and input-dependent coordinate changes need product or metric-variation control. Fixed-coefficient error recursions do not automatically certify the fully coupled learned network. Relevant prior work already treats gated targeted updates and more expressive state transitions: [Gated Delta Networks](https://arxiv.org/abs/2412.06464), [Unlocking State-Tracking Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537), and [DeltaProduct](https://arxiv.org/abs/2502.10297). Basic spectral or overwrite observations need comparison with these before being advertised as novel.

Repeated evaluation of an unchanged deterministic canvas and unchanged model yields the same logits. The pointer walker improves only because its current query/hop state changes; a neural analogy must locate that changing state in the canvas, condition, or persistent memory. Independent resampling may change outputs but does not create missing evidence, and any best-of-k benefit needs a specified non-oracle selection rule.

## 4. Prior art and feasible stronger target

Actual multipass lower bounds permit much more general memory behavior and use communication complexity, not just monotone path runs. [Guruswami–Onak, *Superlinear lower bounds for multipass graph processing*](https://eccc.weizmann.ac.il/report/2013/002/) proves space/pass lower bounds using pointer-chasing information complexity. [Assadi–Chen–Khanna, *Polynomial Pass Lower Bounds for Graph Streaming Algorithms*](https://arxiv.org/abs/1904.04720) develops hidden pointer chasing. Do not cite [the 2020 two-player hidden-pointer paper](https://arxiv.org/abs/2002.12856) as an established theorem: its authors withdrew it for a proof error.

A related 2026 preprint, [*How Much Cache Does Reasoning Need?*](https://arxiv.org/abs/2604.17935), explicitly labels its strongest product lower bound conjectural. This is evidence that depth/cache/pointer bounds require careful model-specific work, not verification of that preprint's claims. Avoid importing its conjecture as a theorem.

**Feasible now:** retain the two exact walker identities as propositions with classical attribution, independently enumerate all small permutations, simulate a pointer-access interface without oracle path access, and demonstrate the cache/access counterexamples. Compare scan counts and charged record visits, not nominal denoising steps. This can strengthen an explanation or a methods appendix; it is not by itself a sufficient new theory-paper contribution.

**Stronger research target:** formulate an unrestricted B-bit streaming controller with a declared finite-precision, mutable scratch canvas of S bits and K permitted history scans; prove a distributional tradeoff for success on a random layered pointer task that allows arbitrary off-path caching and compressed summaries. Seek a matching constructive controller and connect its exact access operations to an implemented recurrent model. A valid proof would likely require a communication reduction or information-complexity lemma, not the descent argument. Establishing an actually new parameter regime requires a further literature and proof audit; this review does not claim such a theorem has been obtained.

A smaller exploratory target is a tight cache-size/pass frontier for an explicitly restricted finite-state walker with b cached records, including adversarial and random orders. Exhaustive dynamic programming can discover counterexamples and candidate optima for small sizes. It remains a model-specific algorithmic result; neither its novelty nor a general asymptotic bound is presently established. The honest decision is to label these as future research targets rather than rush an unsupported theorem into the finished empirical paper.
