# Qualification payload contract

- Scope: CPU-reviewed payload for the one authorized 8×H100, 30-minute qualification attempt. No job was submitted and no CUDA/model execution occurred while producing this evidence.
- Trust root: `runtime_manifest.json`, externally pinned as `364010efe2d9313c35e017a44620d90df16bdc0767a8912f3d518ad4dde1fdee`, binds 32 payload, staged, package/API, HF, and checkpoint files.
- Image gate: the mutable request tag is never evidence of image identity. `QUALIFICATION_IMAGE_DIGEST` must contain an externally resolved and independently reviewed digest before launch.
- Job gate: runtime success requires a scheduler-issued `QZ_JOB_ID` or `JOB_ID`; the operator run id is not substituted.
- Collision policy: a pre-reserved run id names a newly created output directory; rank, preflight, aggregate, and launcher records use no-clobber atomic publication.
- Runtime policy: one `torchrun` attempt, eight ranks, zero restarts, 1,500-second TERM timeout, 30-second KILL grace, and a 90% per-rank memory ceiling.
- Success policy: exactly eight fresh, manifest-bound, cross-rank-consistent passing records produce `PARTIAL_QUALIFICATION` with `corevalid/fullmodelcacheunqualified`.
- Failure policy: missing, extra, stale, malformed, failed, or inconsistent rank evidence produces `FAILED`.
- Supported cache scope: standalone forward `RWKV7Attention` prefix/suffix equivalence only.
- Unsupported full-model cache reasons: `NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED` and `NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES`.
- Submission remains blocked on independent payload review, current private-image digest resolution, live resource re-resolution, and persisted unique submission reservation.
