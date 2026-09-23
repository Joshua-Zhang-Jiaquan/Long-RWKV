# Controlled distance study: final claim audit

Status: COMPLETE.

The review uses all seven validated evaluation selectors, all six fixed adaptation checkpoints, the raw-output collectors, the frozen paired report, and all six matched cost certificates. No seed, position, checkpoint, or endpoint was removed.

## Supported claims

- **Matched distance intervention:** near-only minus balanced far-error reductions are 8.3036, 7.0757, and 2.7723 nats. Every conditional problem-bootstrap interval excludes zero. The seed-averaged reduction is 6.0506 [5.3916, 6.7353]. The far-minus-near interaction is 6.0506 [5.3917, 6.7354]; the averaged near-position difference is negligible and its interval includes zero.
- **Conditional prediction:** original far first/second errors are 0.00043/21.84973 nats. The paired second-error reduction, 6.0505, accounts for essentially the full primary effect. The inference partition is unchanged and D=0, so the contrast changes measured E.
- **Selective evidence use:** every balanced model responds correctly to the declared syndrome flip, with far fixed-history CE below 3e-7 and unaffected-target probability change below 2.7e-6. Near-only seeds 11/12 show weak contrasts and substantial centering penalties; seed 13 is nearly uninformative at far/middle. This supports appropriate constraint use, not identification of a unique internal circuit.
- **Reveal-policy prediction:** balanced mean intact-pair minus information-set KL differs from 4log2 by at most 0.000123686 nats. Both learned errors are measured. Near-only residual estimation differences remain large; equal call count alone does not isolate D.
- **Restricted empirical quality–cost point:** all three balanced checkpoints satisfy the frozen 1.5-second mean-budget certificate on the same eight-problem, three-position quality/timing law. Attention means are 1.003/2.002/2.003 seconds; recurrent two-call means are 1.301–1.307 seconds across all six models.
- **Retained exception:** near-only seed 13 also passes the aggregate certificate (matched KL 1.8483), despite far KL 2.7725 and near-chance distant conditionals. The paper explicitly distinguishes aggregate certification from uniformly competent distant evidence use.
- **Secondary evidence:** all six secondary cells for every selector are in the appendix, including exploratory eight-problem 32K far/near panels. Complete JSON retains all policies and endpoints.
- **Budget:** all 18 jobs are terminal. Reservation intervals peak at 48 total, primary 32, extra 16, max 8/job. Follow-up runtime is 106.34 H100-hours. Six runs process 1,891,699,200 input tokens; qualification adds 3,152,832. Earlier selected-checkpoint training is additional history.

## Scope and exclusions

Three adaptation seeds share one selected lineage. Earlier initial seeds 53/89 failed; this is not reliable learning across independent initial lineages. Fresh prompts use a previously observed held-out structure catalog. Whole public blocks move, including instructions; this is not isolated factual-memory manipulation. Counterfactual CE uses one fixed history; endpoint E integrates all histories. Intervals resample problems conditional on observed seeds and lineage, not a population of training seeds.

The generalized-delta recurrent denoiser has linear sequence arithmetic and is not exact softmax attention. Latency concerns qualified FP32/IEEE implementations and a restricted policy set. Attention quality is not measured; the product-law lower bound supplies the certificate. Cached causal decoding and optimized alternatives are not excluded. No memory advantage is demonstrated. FLA numerical/isolation checks pass, but official-kernel parity remains unverified.

The identities, change-of-measure bound, and convexity diagnostic are elementary/established tools. No new general contraction or sample-complexity theorem, broad natural-language superiority, first recurrent diffusion model, or conference-level novelty is claimed.

## Retained history and presentation

The preceding 640-condition failed extrapolation study and figures/tables remain in the appendix. Final working-draft placeholders are removed. Publication tables copy every selector from final reports. Figure reflow preserves every plotted coordinate and uncertainty-band vertex; publication_figure_receipt.json records this.

Three pre-model-load failed launches remain in accounting. The execution-manifest repair preserved bytes and scientific settings. Admission waits only on typed pre-submission capacity refusals; no GPU job was automatically retried. The attention job started during a routing recheck; no cancellation or duplicate occurred. Approximately 26% memory occupancy does not meet the earlier 80% target, consistent with the later fastest-completion priority.

Independent numeric, live-resource, hash and PDF-render checks are recorded in results/distance_intervention/delivery_audit.json and distance_intervention_visual_review.json.
