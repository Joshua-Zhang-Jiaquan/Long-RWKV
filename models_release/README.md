# Archived-study inference checkpoints

Seven lossless inference exports are prepared: the selected initial seed71 model
and all six near/balanced adaptations. Each contains every original model tensor
and the original checkpoint contract. Optimizer state is omitted, reducing each
file from about5.1GiB to about1.7GiB. `EXPORTS.json` maps original checkpoint
hashes to new export hashes and verified tensor-content hashes.

Publication is pending; download availability is not yet claimed.

These are the models underlying the earlier completed positional, exhaustive
history-response and header/task-layout studies. They are not the newly training
six-lineage transfer models. Source weights derive from
[fla-hub/rwkv7-0.4B-world](https://huggingface.co/fla-hub/rwkv7-0.4B-world), whose
model card identifies Apache-2.0 licensing. Upstream component notices remain
applicable; this release does not add a blanket license grant over unrelated code.

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
