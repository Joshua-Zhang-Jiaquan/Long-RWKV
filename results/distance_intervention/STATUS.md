# Controlled 16K study complete

All six fixed600-update adaptations and all seven evaluation selectors are complete and validated. The study retains1008 exact conditions,3024 endpoint laws, and2520 RWKV timed requests, plus360 matched attention requests. No selectors or conditions were omitted.

The primary near-only minus balanced far-error reduction is6.0506 nats [5.3916,6.7353], with positive effects in all three adaptation seeds. All three balanced models satisfy the matched empirical quality–cost certificate. Near-only seed13 also passes the aggregate certificate despite near-chance distant predictions; this exception is retained in the paper.

The final paper includes all-seed tables, the mechanism figure, all secondary cells, earlier negative findings, theory and scope limits. The clean14-page PDF has been visually reviewed. Independent numeric, frozen-source, raw-output, manuscript and live-resource completion checks all passed.

All18 jobs are terminal, with zero campaign GPUs reserved. Peak queued+running allocation was48 GPUs, split32 primary+16 extra. Follow-up scheduler runtime was106.34 H100-hours, including qualification and three failed launches before model loading. The attention job waited for shared quota and then ran normally; it was not cancelled or duplicated.

Final records:
- FINAL_REPORT.md and final_report.json
- ../distance_intervention_cost/BUDGET_REPORT.md and budget_certificates.json
- delivery_audit.json
- resource_audit.json and capacity_timeline_audit.json
- ../../PAPER_DELIVERY.md
- ../paper_audit/distance_intervention_delivery_manifest.json

Scientific scope remains conditional on one selected starting lineage, three adaptation seeds, a previously observed held-out structure catalog, synthetic binary tasks and the declared inference implementations/policies. Problem-bootstrap intervals do not estimate population training-seed uncertainty.
