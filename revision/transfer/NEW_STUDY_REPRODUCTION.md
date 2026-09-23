# Reproduce the prospective training study locally

The new six-lineage study does not require any externally supplied adapted
checkpoint. `reproduction/stages/predictive_transfer_training/` is an exact copy
of the stage used for the discarded qualification and all six main jobs.
`TRAINING_CAPSULE.json` binds all68 files; `FROZEN.json` fixes the scientific
protocol. Some unused predecessor modules remain in the inherited closure;
the new worker initializes from public base weights and loads no predecessor
adaptation.

Fetch the pinned public base with `tools/fetch_public_base.py` as described in
`RECONSTRUCTION.md`. Use the recorded PyTorch/CUDA/FLA numerical environment and
eight H100 GPUs. From the repository root, set your local paths:

```bash
export TRANSFER_STAGE="$PWD/reproduction/stages/predictive_transfer_training"
export TRANSFER_PROTOCOL="$PWD/revision/transfer/FROZEN.json"
export TRANSFER_BASE=/your/models/rwkv7-0.4B
export TRANSFER_RUNS=/your/new-transfer-replication
export TRITON_F32_DEFAULT=ieee CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$TRANSFER_STAGE"
```

First run the discarded qualification (eight updates spanning all curriculum
shapes, plus document-isolation checks). It is not a learning-success screen:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m lrwkv_evidence.predictive_transfer.worker \
  --base "$TRANSFER_BASE" --model-root "$TRANSFER_STAGE/model_source" \
  --out "$TRANSFER_RUNS/qualification" --seed 202709230 --qualification
```

Then run each of the six fixed training seeds. This loop uses eight GPUs at a
time; running copies concurrently requires separate GPU allocations. It does
not perform checkpoint selection or retry a failed lineage.

```bash
for seed in 202709231 202709232 202709233 202709234 202709235 202709236; do
  torchrun --standalone --nproc_per_node=8 \
    -m lrwkv_evidence.predictive_transfer.worker \
    --base "$TRANSFER_BASE" --model-root "$TRANSFER_STAGE/model_source" \
    --out "$TRANSFER_RUNS/seed$seed" --seed "$seed" \
    --qualification-out "$TRANSFER_RUNS/qualification" || exit 1
done
```

The workers validate the frozen source digest and protocol, train to update3100,
and record checkpoint hashes. Fresh task-training seeds share the public
pretrained backbone. New checkpoint files and new measurements must be identified
as a new replication; historical results are not copied into it.

The independent CPU checker is `tools/verify_transfer_evidence.py`. It parses
public equations and verifies all256 generated endpoints, not only the16 valid
targets. The release's calibration/held-out raw-output capsule and portable
evaluation instructions will be added after those jobs have completed. Do not
claim an independent replication merely from the training commands or synthetic
checker tests.
