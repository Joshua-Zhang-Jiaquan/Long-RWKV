# Separating instruction-header distance from task-block distance

This prospective follow-up is motivated by the completed distance-training study and exhaustive response audit. Their results were observed before this design. Freeze this protocol, six fixed checkpoints, input layouts and analysis before any new GPU predictions. No retraining, checkpoint selection, outcome-based sample extension or omitted cells.

## Question and design

Does the balanced model retain its advantage on distant task information when the instruction header stays near the answer? The preceding training intervention moved header and task block together. This experiment separately controls their inference positions. It does not retrospectively separate the two components of the training treatment.

Use all six terminal matched adaptations, seeds20271011/12/13 and near/balanced arms. Draw32 fresh paired-parity problems using test seed2026092301, indices0..31. Verify public prompts do not duplicate the preceding32-problem panel, with no outcome-based replacement. The labeled-structure catalog remains previously observed; fresh prompts are not untouched structures or independent training lineages.

Split the native prompt at the exact boundary after the generic instruction header. The task block contains coordinate names, all equations and output-order metadata. Keep the answer footer and output scaffold fixed. Thus this isolates the generic header from the task-bearing block, not syndrome distance from coordinate/query metadata.

Create four fixed-slot conditions: header far/near x task block far/near. Reserve separate far and near slots of exactly the header and task-block token lengths. Fill unoccupied slots with deterministic groups of complete irrelevant archive records of the same native lengths; keep remaining records in original order. Swap each meaningful block with its matching filler. Both blocks' exact near/far token spans stay fixed as the other factor changes. All layouts and legacy controls have the identical token multiset, complete records, prefix length and output scaffold. No truncation, arbitrary padding or extra task information is introduced. Record filler choices and input hashes.

Include legacy_far and legacy_near controls using the original unsplit serializer on the same fresh problems. These quantify any effect of regrouping distractor records for the fixed-slot layout. All six cells are retained. Co-located fixed-slot cells are not claimed to be token-identical to legacy layouts: their irrelevant records may be reordered.

Total: six checkpoints x32 problems x six layouts =1,152 conditions. Each condition evaluates all-masked input and all16 target-distributed information-set histories, retaining full normalized binary probabilities. This is19,584 forward calls excluding numerical screens. Compute exact one-call and information-set endpoint KL, valid mass, first/second estimation errors, and one-minus-two-call gain. No new timing or memory advantage is tested.

## Predictions and inference

Primary endpoint: information-set KL at header_near_task_far. Primary contrast: near-trained minus balanced-trained, separately for all three matched seeds and seed-averaged within problem. Use10,000 paired problem bootstrap draws, seed2026092302, with pairing preserved across every model/layout. Intervals condition on these six checkpoints and selected lineage. Histories are exact inner expectations, not additional independent examples.

The theory keeps D_info=0, so every placement contrast in exact KL is a change in E. The stronger interpretation requires a positive primary conditional interval and matching direction in all three seeds, plus balanced mean E below4ln2 in both header_near_task_far and header_near_task_near for every seed. The4ln2 threshold is theoretical; it is not a post hoc empirical cutoff. Report direct gains over the same model's one-call law as well.

Report the task-distance penalty at each fixed header position, header-distance penalty at each fixed task position, their factorial interaction, the matched training-arm difference in each penalty, and co-located versus legacy discrepancies. Include all cells even if controls fail. A positive arm contrast with poor balanced conditional accuracy does not establish a useful two-pass operating point. Failure on separated layouts narrows transfer claims; it does not erase the completed whole-block result. No internal circuit or training-component mediation claim is identified by this inference intervention.

## Execution and checks

One eight-H100 job per checkpoint; at most48 queued+running GPUs, split primary32 and extra16, max8/job. Preserve FP32/IEEE execution, TF32off and serial predictions. Reuse the qualified model source and compiled cache. Each job repeats the old numerical screen and additionally checks all four split layouts under all-masked and visible-history inputs, with within-repeat tolerance1e-5 and cross-rank tolerance1e-4. Freeze CPU checks for oracle endpoint accounting, exact slot invariance, token multiset/whole-record preservation and fresh-prompt separation. Scheduler limit90minutes/job; no automatic retries. Expected useful inference is about4.5minutes/job plus initialization and queueing, based on previous latency, not a guarantee.

Preserve prior PDFs, data and manifests. Integrate all results, including failures and control discrepancies, only after full validation and independent arithmetic review.
