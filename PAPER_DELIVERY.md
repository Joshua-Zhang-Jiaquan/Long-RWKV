# Updated paper and independent task-position test

**When Do Linear-Cost Diffusion Passes Help with Long Contexts?**

PDF: `build/Long_RWKV_ICLR2027.pdf` (19 pages; main text through page8, references through page9). Source: `Long_RWKV_ICLR2027.tex`. Rebuild: `bash build_focused.sh`.

The latest frozen follow-up separates the generic instruction header from the task-bearing block at inference. It evaluates 32 fresh prompts, four independent position combinations and two legacy controls on all six existing adaptations: **1,152 conditions and 2,304 exact endpoint laws**. It adds no training.

With instructions nearby and task information distant, balanced training reduces conditional error by **7.626609 nats [6.908102, 8.301452]**. All three seed effects are positive: 12.471797, 7.635032 and 2.772998 nats. Every balanced model's mean error is below 0.000153 nats in this cell. All predeclared stronger-interpretation criteria pass. Both training arms remain accurate when the task block is nearby, including with distant instructions.

This supports the training-arm advantage under the declared task-distance intervention with nearby generic instructions. It does not establish layout invariance: regrouping irrelevant records changes one near-trained model's mean far error by **4.089890 nats**. Header position also has an effect. The task block includes names, equations and output-order information, so this is not a pure factual-memory intervention. The inference test does not disentangle the earlier training treatment. All intervals remain conditional on six checkpoints and one selected starting lineage; fresh prompts use a previously observed structure catalog.

The theory remains a cost–error criterion: inexpensive recurrent scans make refinement affordable, but the learned conditional error must be sufficiently low to realize the dependence reduction. The new layout contrasts hold the target dependence term fixed and therefore measure changes in that error. They do not add a new latency comparison or general memory theorem.

| Artifact | Location |
|---|---|
| Frozen protocol and input/checkpoint hashes | `theory_mvp/header_distance/DESIGN.md`, `FROZEN.json`, `INPUTS.json` |
| Complete results and paired inference | `results/header_distance/exact_*.json`, `report.json`, `REPORT.md` |
| Claim review | `results/header_distance/CLAIM_AUDIT.md` |
| Standalone figure | `results/header_distance/header_task_distance.pdf` |
| Independent numerical and provenance audit | `results/header_distance/delivery_audit.json` |
| PDF visual review | `results/paper_audit/header_distance_visual_review.json` |
| Resources and operational replacement | `results/header_distance/resource_audit.json`, `replacement_balanced_seed20271013.json` |
| Current delivery manifest | `results/paper_audit/header_distance_delivery_manifest.json` |

35 CPU tests passed before submission. All frozen source hashes, layouts, numerical screens, raw endpoint laws and paired estimates are checked. Independent recomputation reproduces all 314 reported summaries and raw endpoint arithmetic within 7.11e-15. The final PDF builds without undefined references or overfull boxes and all 19 pages have been visually reviewed.

The six completed evaluations and one stopped startup attempt use **6.597778 scheduler-reported H100-hours**. Peak queued-plus-running reservation is 48 GPUs, split primary32/extra16, eight/job. All jobs are terminal and no campaign GPUs remain reserved. Completed jobs average **55.1–80.7% GPU busy** and **6.6–8.1% memory occupancy**; the earlier 80% targets are not met. These are whole-job measurements, including startup, screens and slower-rank waits; GPU busy is not FLOP utilization. The later fastest-completion preference governed execution.

One job started as a queue-migration stop landed. It produced startup logs but no prediction files; the scheduler reports zero runtime. Its unchanged scientific workload was manually replaced on the primary project with a distinct output directory. Both attempts and the race receipt are retained. No automatic retry or result-based model selection occurred.

The earlier matched study remains intact: 6.051-nat far-error reduction, three balanced quality–cost certificates and approximately 1.30s recurrent versus 2.00s declared attention two-call latency. The exhaustive history audit also remains: 1.477852-nat far counterfactual CE reduction and **three collateral correct-to-incorrect changes among 55,296 unaffected-target comparisons**. The paper retains the failed seeds, failed extrapolation, rare collateral errors, no demonstrated memory advantage and unverified official-kernel parity.

The prior completed delivery and its 141 manifest-pinned files are preserved in `results/paper_audit/completed_history_response_20260923.tar.gz`, SHA256 `5dba1a9938a5a9eb075c747ae9d2180c5be7275517ca652a619665c805c99b17`. Earlier delivery manifests describe historical snapshots; the header-distance manifest describes the current paper.
