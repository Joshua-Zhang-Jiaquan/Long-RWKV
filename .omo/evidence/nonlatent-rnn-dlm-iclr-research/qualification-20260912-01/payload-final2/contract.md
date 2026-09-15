# Qualification payload-final2 contract

- Scope: CPU-repaired payload for the single authorized 8×H100-SXM-80GB, 30-minute qualification attempt. No qz submission, CUDA call, model construction, or checkpoint load occurred while producing this bundle.
- Trust root: `scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json`, externally pinned as `92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c`, binds 35 payload, staged-source, package/API, HF, and checkpoint files.
- Controller binding: every rank must load the same strict receipt produced from the successful controller-side `CreateJob` response. The receipt binds the real returned job ID, exact submitted-request SHA-256, authorization/resource tuple, run ID, nonce, manifest, image catalog identity, and authorized limits.
- Receipt path: `/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/controller-receipts/${QUALIFICATION_RUN_ID}.json`, supplied through `QUALIFICATION_JOB_RECEIPT` or `--controller-receipt`.
- Receipt timing: ranks wait at most 60 seconds before importing CUDA libraries. Missing, malformed, oversized, symlinked, stale, or inconsistent receipts fail closed.
- Job identity: nonempty `QZ_JOB_ID` and `JOB_ID` values must equal the receipt job ID. If neither exists, evidence records origin `controller_qz_createjob_receipt` rather than inventing an environment identity.
- Image identity: requested tag `docker.sii.shaipower.online/inspire-studio/relay2:v2` and catalog ID `image-7330d118-df9d-4e2e-82b1-c2543e831eb4` are recorded. `observed_image_digest` is strictly null because the reviewed qz interfaces expose no digest or digest submission field.
- Hardware identity: only `NVIDIA H100 80GB HBM3` and `NVIDIA H100-SXM5-80GB` with 79,000–83,000 MiB and eight unique UUIDs are accepted. NVL, PCIe, other names, and out-of-range capacities are rejected.
- Checkpoint boundary: `torch.load(..., weights_only=True, map_location="cpu", mmap=True)` accepts a flat `Mapping[str, torch.Tensor]`, including `OrderedDict`, followed by `load_state_dict(..., strict=True)` and exact tensor-count checks. No unsafe fallback or safetensors conversion was introduced.
- Workflow budget: 1,650-second outer timeout plus 30-second kill grace; preflight 300 seconds, runtime 1,200 seconds, aggregation 90 seconds, receipt wait 60 seconds, and NCCL timeout 120 seconds.
- Process policy: one `torchrun` attempt, eight ranks, one node, zero restarts, no automatic retry, bounded process-group cleanup, and terminal launcher receipts on timeout paths.
- Success policy: exactly eight fresh nonce-, manifest-, and controller-binding-consistent passing records produce `PARTIAL_QUALIFICATION` with `corevalid/fullmodelcacheunqualified`.
- Failure policy: missing, extra, stale, malformed, failed, controller-inconsistent, resource-inconsistent, or image-inconsistent rank evidence produces `FAILED`.
- Supported cache scope: standalone forward `RWKV7Attention` prefix/suffix equivalence only.
- Unsupported full-model cache reasons remain `NOT_QUALIFIED_FFN_TOKEN_SHIFT_STATE_UNWIRED` and `NOT_QUALIFIED_TIED_LAYER_STATE_ALIASES_PASSES`.
- Submission remains blocked on an independent payload-final2 review, live resource re-resolution, persisted unique request bytes/reservation, and controller-side receipt publication after the sole successful `CreateJob` response.
