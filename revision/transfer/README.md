# Review-driven revision in progress

The additional experiment is running. No new transfer outcome is claimed.

- [Working manuscript PDF](build/Long_RWKV_revision.pdf): the main text centers on prospective prediction; earlier interventions and the restricted cost certificate remain supporting evidence. Results-pending language is explicit.
- [Frozen protocol](FROZEN.json): six fresh lineages,15 source code classes,10 held-out classes; both policies have zero discarded dependence and two calls.
- [Design rationale](REVISION_PLAN.md) and [working response to the review](REVIEW_RESPONSE.md).
- [Original-study checkpoint downloads](../../models_release/README.md): seven verified lossless inference exports and the evaluation wrapper.
- [New-study full reproduction](NEW_STUDY_REPRODUCTION.md): exact training/evaluation capsules, public-base commands, prediction sealing and independent analysis.
- [Reconstruction guide](RECONSTRUCTION.md): exact public base inputs, initial parent/branch training, six adaptations, and new evaluations with new checkpoint identities.
- [Operations removed from the manuscript](OPERATIONS_FROM_PAPER.txt): historical scheduler/resource narrative is retained outside the scientific exposition.

The controller runs source calibration as fixed terminal models complete. All six
prediction artifacts must be sealed before any held-out evaluation. The study
retains every lineage, fixed fallback and failure; source-only qualification and
training loss do not establish success on held-out classes.

Build this working draft from the repository root:

```bash
bash revision/transfer/build_draft.sh
python tools/verify_transfer_protocol.py
```

The original reviewed paper, original evidence seal and all frozen historical
scientific sources remain unchanged. The working manuscript becomes the final
revision only after complete prospective results have been incorporated and
reviewed.

The optional software integration check
`python revision/transfer/check_synthetic_pipeline.py` builds synthetic probability
panels in a temporary workspace, exercises all six lineages and 1,440 target cells,
retains two chance-level fixtures, and independently rechecks 96 compressed rank
files after deleting their originals. Its receipt is explicitly synthetic and
must not be cited as model performance.
