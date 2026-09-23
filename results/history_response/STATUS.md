# Exhaustive evidence-response follow-up

Status: COMPLETE.

All seven fixed checkpoints, 672 conditions and 43,008 paired interventions passed the frozen validation. All 16 histories and four constraints are retained for each problem/position. The 16-page paper includes the exact response identity, supplementary sufficient bound, complete results, and rare collateral failures.

- Primary far near-minus-balanced paired CE: 1.477852 nats, conditional 95% CI [1.327451, 1.637084]; positive in all three adaptation seeds.
- Three correct-to-incorrect collateral events among 55,296 unaffected-target comparisons; no uniform-selectivity claim.
- Independent audit: delivery_audit.json. Full results: report.json and REPORT.md. Scientific scope: CLAIM_AUDIT.md.
- All seven jobs succeeded; 11.89 additional H100-hours; peak 48 (32+16), max 8/job; zero GPUs remain reserved.
- PDF: build/Long_RWKV_ICLR2027.pdf; main text and references through page8, appendices through page16.

This is a post-study diagnostic using existing problems and one selected starting lineage, not independent replication. The prior completed delivery is preserved in results/paper_audit/completed_distance_study_20260922.tar.gz. Frozen prior protocols/data remain intact. The new manuscript corrects the native-prefix wording: RWKV 16K uses 16,383 tokens and attention cost 16,384 under the same requested budget.
