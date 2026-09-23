# Independent final evaluation: design before development outcomes

This is a design, not a frozen experiment receipt or completed result. No test
examples have been evaluated by this analysis work. The source/checkpoint
manifest, sample count, model replicates, and measured cost budgets still need
to be frozen after the development gates and runtime measurements.

The primary mechanism contrast is the same-checkpoint information-set policy
versus one call on the dependent task at 16K, averaging the three evidence
positions equally within each base problem. Forward KL is the primary endpoint;
exact valid mass and within-valid KL describe its practical meaning. The
information-set versus pair-preserving contrast controls the number of calls
and reveal-group sizes. Report it even if it disagrees with the one-call result.

Report all evidence-only/1K/4K/16K cells and all near/middle/far positions. For
each long cell, also report the paired difference between its policy gain and
the evidence-only policy gain. A positive absolute gain at 16K does not by itself
mean diffusion improves retention with increasing context. Retrieval controls
and the long-context numerical qualification remain necessary.

Use disjoint test keys with one predeclared data seed and an explicit index
list. A practical candidate is 64 base problems per family, paired across all
lengths and positions. Development variability and measured audit time may
justify 32 or 128 instead; freeze the choice and explain its precision/runtime
tradeoff before generating any model outputs on test. The sample size is not a
guarantee of significance. Do not drop failed positions or silently use only
complete cases.

For each fixed checkpoint, resample base indices, retaining every policy,
length and position together. Exact endpoint enumeration removes sampling
noise from the decoder law; it does not remove variability across held-out
problems. `paired.py` implements this analysis, including pointwise 95% paired
bootstrap intervals and the length interaction. It refuses incomplete panels.
It is an analysis helper, not yet a final-data collector.

Training replicates use the same frozen recipe and terminal update, with
separate data/corruption seeds from the same released initialization. Retain
every predeclared replicate and report each separately. Do not pool their
condition rows as independent samples or interpret problem-bootstrap intervals
as uncertainty over training. Additional replicates are justified only after
the development run learns the short task; repeated failure is not evidence
for the intended mechanism.

Cost claims require complete requests with prefill or all denoising calls,
warmup and synchronization. Causal RWKV uses a cached prefix; audit caches are
not timing optimizations. Final timing repeats and any budget threshold must be
fixed from development measurements, not chosen from test outcomes. The
attention comparison uses native tokenization and different pretraining, so it
is a practical operating-point comparison, not an architecture-causal result.
Report native token lengths, distractor counts, peak allocated memory and the
actual GPU/precision/software configuration. A nonempty budget interval and a
positive KL certificate must hold together for the restricted policy claim.

If the main short gate fails, finish and retain the development audit, diagnose
the learned conditionals using the archived step-200 checkpoint, and record any
new recipe as a new development experiment. Do not spend the final test panel
to tune the failed recipe. If long-context gains or the cost advantage fail,
retain the full result and limit the paper's claim accordingly.
