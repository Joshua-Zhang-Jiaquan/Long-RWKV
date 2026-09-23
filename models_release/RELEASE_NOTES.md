# Draft notes for the later Hugging Face job

Publication is deferred by user instruction. No upload is authorized by this document.

Lossless inference exports for the completed Long-RWKV positional and evidence-use studies.

Includes the selected initial seed71 checkpoint and all six near/balanced adaptations. Every model tensor and checkpoint contract matches its archived original; optimizer states are omitted. `EXPORTS.json` records original file hashes, new export file hashes and verified tensor-content hashes. Each export is approximately1.82GB.

These local exports are prepared for later publication through a separate Hugging Face job. There is no current public adapted-checkpoint release. Once an export is available, it can support the earlier frozen evaluations without reconstructing task adaptation from scratch. Use `tools/run_exported_checkpoint.py` and the pinned public base/tokenizer assets described in `models_release/README.md`. Numerical reproduction still requires the recorded eight-H100 environment. The export audit establishes tensor identity; it is not a new neural evaluation or independent replication.

These are **not** the six fresh models in the ongoing prospective-transfer revision. No new transfer result is claimed by this release.

Base model: fla-hub/rwkv7-0.4B-world, revision793168635c21ffa8882ed20f908bbb65a964e48b. The base model card identifies Apache-2.0 licensing; the license text and source attribution accompany the exports. Changes comprise bidirectional denoiser adaptation, synthetic posterior-task training, and inference-only packaging. Upstream model authors are credited in the model card. Scientific sources and historical evidence remain in the repository.
