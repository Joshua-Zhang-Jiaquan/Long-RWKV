# Independent header/task distance study

Status: all six evaluations complete and validated; all predeclared interpretation criteria pass. The paper is updated and all 19 pages visually reviewed.

All 35 CPU tests passed. The panel contains 32 fresh prompts, four independent fixed-slot header/task placements and two legacy controls. Exact token spans, whole-record fillers and token multisets are checked. No prompt duplicates the preceding 32-problem panel.

Protocol: theory_mvp/header_distance/DESIGN.md. Frozen manifest SHA256: e467d6a45a7266bc2c60a0cbf9619694bd1daf837c954f4c141c63db24c26c5c.

All six matched adaptation checkpoints are retained. Completed output: 1,152 conditions and 2,304 exact endpoint laws. No retraining or new latency comparison. Allocation is capped at 48 queued+running H100s (primary32 + extra16), eight/job. No automatic retries.

The interpretation concerns independent inference positions of the generic header and task-bearing block. It does not isolate syndrome tokens from task metadata or identify separate components of the training treatment. Fresh prompts still use a previously observed structure catalog and one selected starting lineage.

Operational note: the extra-project quota delayed two jobs. One started normally; the other started as a queue-only stop landed. It was stopped before any prediction files, with zero scheduler-reported runtime but startup warning/GPU logs. Its original plan and files are retained; a manually reviewed replacement uses the identical stage and checkpoint with a unique output directory on the primary project. See replacement_balanced_seed20271013.json and queued_attempts/.

Primary reduction: 7.626609 nats [6.908102, 8.301452]. All jobs are terminal; current campaign reservation is zero. Resource cost: 6.597778 scheduler-reported H100-hours including the stopped-attempt receipt (zero reported runtime). Whole-job GPU busy 55.1–80.7%, memory occupancy 6.6–8.1%; the 80% targets were not met.

Independent delivery audit passed: 314 summaries, maximum endpoint arithmetic discrepancy7.11e-15. Final PDF SHA256: cd8482454a871ec57e455488c226e5669aae952f5acde72e1df097cb3c282634.
