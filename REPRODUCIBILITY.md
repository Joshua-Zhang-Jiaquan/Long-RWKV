# Reproduction and provenance

## Offline measurement verification

Follow the CPU commands in the README. The release includes every raw rank output for the matched intervention, exhaustive history response, independent header/task layout panel, and matched attention timing panel. `evidence/raw/INDEX.json` maps each historical absolute path to a repository-relative gzip file. Decompressed bytes retain their original collector hash. No historical checkpoint selector, scientific source, raw prediction, interval rule or result is rewritten to make packaging pass.

`tools/verify_evidence.py` uses NumPy and the pure-Python task generator. It independently enumerates binary target support, evaluates endpoint/response arithmetic, checks complete panels, computes the three primary intervals, reproduces the collateral-error count and re-evaluates the six budget certificates. The archived numerical screen records are checked too. It is an offline reanalysis, not independent experimental replication. The original cluster audit additionally checked native tokenizer layouts and live checkpoint files; those checks remain recorded in `results/*/delivery_audit.json` and are not represented as fresh CPU checks here.

Historical absolute paths and scheduler/project IDs in receipts are provenance, not runtime prerequisites for the portable checks. The historical 211-file delivery manifest and prior completed archives are preserved. `RELEASE_MANIFEST.json` seals this standalone export, including the relocation index and portable tools. The original local `delivery_audit.py` scripts retain their original cluster dependencies and should not be confused with the portable entry point.

## Rerun a frozen model evaluation

The following inputs must be supplied separately:

1. The released RWKV base directory, including native tokenizer implementation, vocabulary, config and weights.
2. A selected adapted `resume.pt` with the exact SHA256 in the corresponding frozen manifest. The six checkpoints are not included in Git; their public download availability is not claimed.
3. Eight H100 GPUs and the recorded numerical environment. Recorded packages are PyTorch `2.8.0a0+5228986c39.nv25.6`, Triton `3.3.0`, Transformers `5.3.0`, flash-linear-attention/fla-core `0.5.0`, CUDA runtime `12.9`. Arbitrary replacement wheels have not been qualified as equivalent.

For example, validate the stage and supplied checkpoint and print a local launch command:

```bash
python tools/run_frozen.py --study header_distance \
  --role balanced_seed20271011 \
  --base /your/models/rwkv7-0.4B \
  --checkpoint /your/checkpoints/balanced_seed20271011/resume.pt \
  --out /your/new-run-directory
```

Add `--execute` to run the printed command on eight local GPUs. The study choices are `distance_intervention`, `history_response`, and `header_distance`; the last uses the six adaptations, while the earlier two also permit `original`. The wrapper supplies local paths around unchanged frozen code, validates source/checkpoint hashes, preserves IEEE FP32 and TF32-off settings, and refuses an existing output directory. It does not submit a cluster job or silently retry a failed run. It is a relocation wrapper, not a claim that inference was newly rerun for this release.

## Training code and lineage

The original training worker and its full local dependency closure are in `reproduction/stages/training/`. The frozen design is `theory_mvp/distance_intervention/FROZEN_DESIGN.json`; the 600-update execution selection is `results/distance_intervention/FROZEN_EXECUTION.json`. The six fixed runs use three adaptation seeds, two position arms, eight GPUs/run, global batch32, and the same selected seed71 starting checkpoint. Training traces and terminal receipts are in `evidence/run_records/`.

To reproduce training, supply the initial seed71 checkpoint identified by SHA256 in the frozen design, set `DISTANCE_EXECUTION` to the local execution manifest, and use the original worker's explicit `--base`, `--model-root`, `--out`, `--initial-checkpoint`, `--initial-sha256`, `--arm`, `--seed`, `--steps 600`, and `--qualification-out` arguments. First run the worker's six-update discarded `--qualification` with the prescribed balanced/20271011 selector; use that new qualification output for the main run. Set `TRITON_F32_DEFAULT=ieee`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and launch eight ranks from that immutable stage. The original selected initial checkpoint is an external dependency; this release does not promise reliable learning from every independent initial seed.

## Scope of resource accounting

The authorized cap was 32 primary-project plus 16 extra-project H100s, at most eight/job, counting queued and running reservations. All completed campaign jobs were terminal at delivery. The matched study used 106.34 scheduler-reported H100-hours, the exhaustive response study 11.89, and the header/task study 6.60; earlier initial-checkpoint training is excluded. No GPU experiment is launched by the release checks.

The header/task study retains one stopped startup-race attempt and its manually reviewed replacement. The stopped job had no predictions; scheduler runtime is zero despite startup logs. Whole-job GPU-busy and memory-use measurements are reported honestly, including failure to meet the earlier 80% utilization targets. The later user priority was fastest completion without changing the frozen experiment.

## Third-party components

The supplied conference style retains its source notices. The local `longrwkv` implementation uses external PyTorch and FLA dependencies; base model and tokenizer assets remain external. This export does not add a blanket license grant for upstream code, style files or weights. Consult each component's own terms before redistribution.
