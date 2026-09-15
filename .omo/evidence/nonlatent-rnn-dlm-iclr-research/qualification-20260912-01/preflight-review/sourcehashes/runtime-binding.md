# Runtime binding status

- The SHA-256 values in `sha256.txt` identify the files inspected during this CPU review.
- The current launcher does **not** compare those digests inside the allocated job before importing the staged model or loading weights.
- `STAGED_ROOT`, `CHECKPOINT_DIR`, `MODEL_DIR`, and the repository payload remain mutable path references. A successful future rank result therefore would not prove it ran the reviewed bytes.
- The image is referenced by mutable tag `relay2:v2`, not an observed immutable digest. The locally available source inspected here reports `flash-linear-attention=0.5.0`, `fla-core=0.5.0`, `transformers=5.3.0`, `torch=2.8.0a0+5228986c39.nv25.6`, `safetensors=0.5.3`, and `pytorch-triton=3.3.0+git96316ce52.nvinternal`; this is not yet bound to the eventual job runtime.
- Before the sole submission, the launcher must fail closed on reviewed source/config/checkpoint/HF-shard digests and emit the observed package versions plus image/job identity into every rank record.
