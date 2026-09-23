# Replication implementation status

The candidate paired-history recipe now has a seed-parameterized training data
and loss implementation in `lrwkv_evidence/train04confirm/recipe.py`. The recipe is now frozen in FROZEN_RECIPE.json after both terminal development
audits were reviewed. Three fresh-seed parent training jobs are submitted.
This is not a report of completed confirmation results.

If this comparison is selected, each fresh seed reproduces the entire lineage:

1. Updates1--300: two-bit ordinary corruption, oracle soft labels, contiguous
   target suffix, from the released checkpoint with initialization seed0.
2. Updates301--700: four-bit50/50corruption/policy-history mixture with public
   output labels; retain optimizer state.
3. Updates701--1500: the same mixture and public-label format on the original
   eight-bit task; retain optimizer state.
4. Fork that seed's step1500model and optimizer into independent and
   complementary histories. Both branches run1000additional eight-bit updates.

The seeds53,71,89 change public training conditions, corruptions, and sampled
histories; they do not change the pretrained initialization or held-out panel.
The two branches share all training before the intervention and share public
conditions, rounds, first histories, and token budgets afterward. Do not reuse
the seed17parent as a substitute for full fresh-seed replication.

CPU regression checks passed at steps1,300,301,700,701,1500,1501,2500: seed17
canvases equal the existing development implementations, and loss values and
logit gradients agree exactly on fixed nonconstant logits. Separate checks
verify changed data across fresh seeds and matching between treatment/control.
The combined recipe and paired-history suite passed13tests. These tests do not
qualify GPU optimizer execution or establish task competence.

The full-model runner and submission path are now implemented in
`lrwkv_evidence/train04confirm/worker.py` and `qz/submit_train04confirm.py`.
They validate source hashes, require a frozen research recipe, restrict fresh
seeds, preserve the model and optimizer across curriculum transitions, and
require a same-seed step1500parent for each branch. A separate
`qualification_only` manifest permits only a discarded20-update seed17run.
The contract and recipe suites passed16CPU tests. The eight-GPU qualification
was submitted as job-32ce21a4-48de-4f78-9276-86b6221ee6a1, with immutable stage
4548a3c431165fd9; its GPU result passed: all20updates were finite, both serialization checks passed atN2/N4/N8, and the saved checkpoint hash was verified. The qualification checkpoint is discarded. Later-phase optimizer execution remains to be exercised by the research runs.

The manifest is frozen and the limited GPU qualification passed.
Still required: finish training three fresh seeds; evaluate fixed terminal models
using the frozen panel/partitions and qualified IEEE inference; report all
seed/family/structure outcomes without selecting on confirmation results.

The terminal confirmation evaluator, collector, and submission path are now
implemented in `lrwkv_evidence/train04confirm/evaluate.py`, `collect.py`, and
`qz/submit_train04confirm_eval.py`. They require frozen source/input hashes and
step2500fresh-seed checkpoints, use explicit IEEE inference, verify all216
condition/policy endpoint rows, and preserve per-condition and per-structure
contrasts with separate family summaries. The21contract, recipe, and collector
tests passed using development conditions only. No confirmation checkpoint has
been evaluated; the recipe was subsequently frozen before submitting the three fresh-seed parent jobs.

Final aggregation is prepared in `theory_mvp/train04confirm/report.py`. Run it
as a module after all six terminal audits exist. It revalidates raw shards and
checkpoint hashes, requires the full three-seed/two-arm design, and reports all
48seed/family/policy rows plus paired training and scheduling contrasts, with
per-seed values and descriptive ranges. Seven focused tests passed, including
retention of an adverse seed and rejection of missing, duplicate, mixed-precision,
and nonfinite evidence. No final confirmation report has been produced yet.
