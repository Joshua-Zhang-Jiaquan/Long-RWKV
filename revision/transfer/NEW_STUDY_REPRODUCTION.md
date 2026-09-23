# Reproduce the complete prospective study locally

The six-lineage study starts from public weights and requires no externally
supplied adapted checkpoint. The exact training and evaluation source closures
are in `reproduction/stages/predictive_transfer_training/` (68 files) and
`reproduction/stages/predictive_transfer_evaluation/` (71 files). Their capsule
manifests and `FROZEN.json` bind the files and scientific protocol. Verify them:

```bash
python tools/verify_transfer_protocol.py
```

Fetch the pinned public base with `tools/fetch_public_base.py`, as described in
`RECONSTRUCTION.md`. Use eight local H100 GPUs and the PyTorch/CUDA/FLA environment
recorded in `REPRODUCIBILITY.md`. The wrapper below submits no scheduler jobs.
It validates public assets and source hashes, runs unchanged workers, and puts
new outputs in a separate workspace. It refuses existing run directories and
performs no automatic retries or checkpoint selection.

From the repository root, set your paths:

```bash
export TRANSFER_BASE=/your/models/rwkv7-0.4B
export TRANSFER_REPLICA=/your/new-transfer-replication
```

First run the discarded source-only qualification. Omit `--execute` to inspect
its launch command without starting a GPU process.

```bash
python tools/run_transfer_replication.py --phase qualification \
  --base "$TRANSFER_BASE" --work "$TRANSFER_REPLICA" --execute
```

Run all six fixed seeds. This loop runs sequentially on eight GPUs; parallel
copies require separate GPU allocations. The worker includes700 small-task,
1,800 source-code and600 long-context updates, for3,100 total.

```bash
for seed in 202709231 202709232 202709233 202709234 202709235 202709236; do
  python tools/run_transfer_replication.py --phase train --seed "$seed" \
    --base "$TRANSFER_BASE" --work "$TRANSFER_REPLICA" --execute || exit 1
done
```

Calibrate all terminal checkpoints on the fixed source panel, then seal all
six prediction artifacts. Sealing invokes the original frozen analysis functions
with only their filesystem roots relocated into this replication workspace.
No score, threshold, panel, seed or algorithm is changed.

```bash
for seed in 202709231 202709232 202709233 202709234 202709235 202709236; do
  python tools/run_transfer_replication.py --phase calibration --seed "$seed" \
    --base "$TRANSFER_BASE" --work "$TRANSFER_REPLICA" --execute || exit 1
done
python tools/run_transfer_replication.py --phase seal \
  --work "$TRANSFER_REPLICA" --execute
```

Only then run held-out evaluation. The wrapper and evaluator require the complete
prediction seal, matching checkpoint identities and unchanged prediction hashes.

```bash
for seed in 202709231 202709232 202709233 202709234 202709235 202709236; do
  python tools/run_transfer_replication.py --phase heldout --seed "$seed" \
    --base "$TRANSFER_BASE" --work "$TRANSFER_REPLICA" --execute || exit 1
done
python tools/run_transfer_replication.py --phase report \
  --work "$TRANSFER_REPLICA" --execute
```

The last command writes the frozen report and independently checks public
constraints, all256 endpoint probabilities, source calibration, sealed choices,
all reported strata and the hierarchical interval. The original archived
results are never copied into the replication. New checkpoint files and
measurements must be identified as a new replication, including every failure.

Dry-run/source validation and synthetic checker tests have passed locally.
The complete portable workflow has not been rerun on a second GPU environment;
do not treat these checks as independent neural replication. The active cluster
study uses the same frozen scientific source bytes under its recorded launcher.
