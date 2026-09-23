# Reconstructing the earlier selected lineage from public base weights

`tools/reconstruct_lineage.py` supplies the previously missing path from the
public pretrained RWKV weights to the selected initial checkpoint, all six
adaptations, and new evaluations. This is a reconstruction workflow, not a
claim that independent retraining has already reproduced the archived numbers.
No pretrained/adapted weights are bundled in Git.

Use eight local H100s and the numerical environment recorded in
`REPRODUCIBILITY.md`. The base directory must contain the native tokenizer,
config and weights whose `model.safetensors` SHA256 is
`e162387e439dfa3387a0ca7da61638749d00c9862b8cc0192ae5d366c8c1a524`.
All commands validate this hash and the archived source closure. They print
commands and prepare isolated stages by default; add `--execute` to train/evaluate.
Use a new work directory for execution after inspecting a dry-run directory.

```bash
python tools/reconstruct_lineage.py --phase initial \
  --base /your/models/rwkv7-0.4B --work /your/reconstruction --execute

python tools/reconstruct_lineage.py --phase adaptations \
  --base /your/models/rwkv7-0.4B --work /your/reconstruction --execute

python tools/reconstruct_lineage.py --phase evaluation \
  --base /your/models/rwkv7-0.4B --work /your/reconstruction \
  --study distance_intervention --role balanced_seed20271011 --execute
```

The initial phase starts from the public base, runs seed71 through parent
update1500, restores its model **and optimizer**, and runs the independent-history
branch through update2500. It uses the original short-training precision
(Torch TF32 disabled; Triton default), not the later IEEE-only adaptation path.
The original worker performs its document-isolation checks in each phase.

The adaptation phase performs a discarded six-update qualification followed
by all six600-update near/balanced adaptations. Each starts from the newly
reconstructed seed71 checkpoint with a fresh optimizer; the three fixed
adaptation seeds and training data remain unchanged. It runs sequentially on
eight GPUs and performs no automatic retries or checkpoint selection.

The evaluation phase supports `distance_intervention`, `history_response`, and
`header_distance`. Run each required role separately. The first two also accept
`original`. It keeps task generators, inference code, cells and numerical
screens unchanged, but binds a separate copy of the manifests to the newly
reconstructed checkpoint hashes. This avoids claiming that new `resume.pt`
serialization bytes match historical checkpoint hashes. Recomputed measurements
must be reported as a new replication, including any training failure or
difference from the paper. The archived results and frozen stages are untouched.

A dry-run verifies launch construction and source/hash checks only. It does not
verify learning, numerical equivalence of a replacement software environment,
or equality of newly trained models. The new six-lineage transfer experiment
uses a different task family and does not replace replication of this older study.
