# Archived-study inference checkpoints

Seven lossless inference exports are prepared locally: the selected initial seed71 model
and all six near/balanced adaptations. Each contains every original model tensor
and the original checkpoint contract. Optimizer state is omitted, reducing each
file from about 5.1 GiB to about 1.7 GiB. `EXPORTS.json` maps original checkpoint
hashes to new export hashes and verified tensor-content hashes.

Checkpoint publication is deferred to a later, separate Hugging Face job, as instructed by the user. The GitHub checkpoint release and its uploaded assets have been removed. No current public checkpoint download is claimed. Local exports and original checkpoints remain intact.

These are the models underlying the earlier completed positional, exhaustive
history-response and header/task-layout studies. They are not the newly training
six-lineage transfer models. Source weights derive from
[fla-hub/rwkv7-0.4B-world](https://huggingface.co/fla-hub/rwkv7-0.4B-world), whose
model card identifies Apache-2.0 licensing. Upstream component notices remain
applicable; this release does not add a blanket license grant over unrelated code.

For an existing local export, verify its recorded SHA256:

```bash
python tools/fetch_inference_checkpoint.py --role balanced_seed20271011 \
  --out /your/exports --verify-only
```

The downloader refuses remote fetching while no publication URL is registered.
The complete [reconstruction workflow](../revision/transfer/RECONSTRUCTION.md)
starts from public base weights and is available without adapted-weight downloads.

After obtaining an export, validate it and print an eight-GPU evaluation command:

```bash
python tools/run_exported_checkpoint.py --study header_distance \
  --role balanced_seed20271011 --base /your/models/rwkv7-0.4B \
  --checkpoint /your/exports/balanced_seed20271011.pt \
  --work /your/new-evaluation
```

Add `--execute` to perform inference in a new work directory after inspecting the
dry-run. The wrapper verifies file and tensor hashes, keeps scientific code and
task panels unchanged, and binds a separate manifest to the export's new container
hash. It does not relabel the export as byte-identical to the original `resume.pt`.
For continued training with optimizer state, use the reconstruction workflow.

Tensor equality has been checked for all seven exports. Export preparation does
not count as new neural inference or independent experimental replication.

The earlier publication-verification receipt is historical and is superseded by
`results/predictive_transfer/checkpoint_release_withdrawal.json`. It does not
establish current download availability.
