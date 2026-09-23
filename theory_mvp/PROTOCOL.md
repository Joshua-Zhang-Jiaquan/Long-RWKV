# Theory-first MVP execution protocol

User authorization: work on the proposed theory/MVP plan with a 32 H100 budget. Operational interpretation: at most 32 campaign H100s running or queued, at most 8 per job. All Python, Bash and qz commands are preauthorized. No new pretraining. Frozen E3 results and paper remain separately preserved.

## Research question and gates

Determine whether an access/memory/refinement result can be both mathematically informative and distinct from prior work. Passing a finite check does not establish novelty. A model benchmark cannot establish a universal lower bound.

1. Formalize all retained state, scratch tokens, history access, query timing, precision and cost units.
2. Audit novelty and independently check exact controlled instances.
3. Run a small frozen F2/R0 natural-language competence pilot. This is new development data, not confirmation and not a replacement for the earlier failed benchmark.
4. Expand to long-context/computation comparisons only after a defensible theorem target and the frozen competence gates both pass. A negative gate is a result, not permission to change the endpoint after inspecting outputs.

## Exact CPU experiments

- Restricted chain walker: unique edges, query-dependent active node, no off-path cached records. A scan is one directional traversal. Enumerate all permutations for 1–8 hops. Check forward and alternating scan formulas against independent record-by-record simulation. One cached off-path record is an explicit counterexample/control to transferring the theorem outside its assumptions.
- Finite binary random access: independent uniform bits, query revealed after encoding, B retained bits and arbitrary decoder computation. Enumerate all decoder codebooks for 1–4 source bits, compare exact optimum with the standard Fano converse and whole-bit-storage baseline.
- Linear-compression lab: uniform source, rank-r binary encoder, all original-coordinate reveals sampled independently within a round from exact conditional marginals. Enumerate all 35 rank-two rowspaces on four bits, all compression values and feasible partial observations. Verify the rank expression for batch dependence and exact one-/two-round output laws. Treat linear solving/matrix storage as explicit oracle capabilities, not free recurrent computation.

These experiments were designed adaptively while developing the theory; they are not preregistered empirical confirmation. Their results are exact enumerations of explicitly declared finite systems.

## Frozen-model competence pilot

Authoritative machine-readable contract and immutable source/data hashes are recorded by `qz/submit_mvp_lookup.py` in its content-addressed stage and `results/mvp_lookup_submission_plan.json`.

- Same 120 one-hop and 120 two-hop natural-language lookup examples per model, with balanced color labels and unrelated records. No long neutral filler.
- One fixed chat instruction requests one color word; no prompt-format selection from results.
- F2: existing checkpoint, matched Bernoulli 8 stages, temperature 0.7. R0: existing causal checkpoint, greedy decoding.
- Fixed 32-token output budget for both, not gold answer length.
- Primary metric: frozen first-line answer parser. Separate diagnostic: whole-word reference occurrence in the full output. The latter is not an alternative gate or exact-match score.
- Competence gate: at least 96/120 primary-correct for each model/depth, with complete coverage and valid provenance. A failed model is not promoted to the planned length sweep.
- Record output IDs/text, input lengths/hashes, parser results, actual calls, resources, and checkpoint/runtime/source hashes.

The binary linear-compression oracle experiment and pretrained text pilot are different systems. Success in either is not asserted to validate the other. Connecting them requires an additional qualified implementation.
