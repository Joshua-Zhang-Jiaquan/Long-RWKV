# Independent payload-final2 handoff

## Review target

- Manifest: `scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json`
- Pinned SHA-256: `92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c`
- Manifested files: 35
- Authorization: `.omo/authorizations/nonlatent-h100-qualification-20260912-01.json` remains unused (`jobs_submitted: 0`).

## Prior NO-GO closure map

1. Exact-dict checkpoint rejection: `model_checks.py` now accepts only flat `Mapping[str, Tensor]` values under `weights_only=True`; the CPU-only `OrderedDict` regression is in `test_qualification_runtime_contracts.py`.
2. Broad H100 admission: `runtime_identity.py` now accepts two reviewed SXM80 aliases with a bounded 79,000–83,000 MiB range; NVL and PCIe negatives are covered.
3. Incomplete deadline: `run_qualification.sh` now has a 1,650-second outer deadline, 300/1,200/90-second phase deadlines, terminal timeout receipts, process-group cleanup, and 30-second kill grace. NCCL is capped at 120 seconds.
4. Unobservable digest requirement: `QUALIFICATION_IMAGE_DIGEST` was removed. Receipt/evidence records requested tag, catalog ID, and strict null `observed_image_digest`.
5. Environment-only job identity: `controller_receipt.py` requires the real returned qz job ID and exact request/auth/run/nonce/manifest/resource/image bindings. Native job variables are comparison-only.

## Independent CPU commands

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m pytest -q -p no:cacheprovider scale/tests/nonlatent_iclr
/usr/bin/bash -n scale/experiments/nonlatent_iclr/qualification/run_qualification.sh
scale/experiments/nonlatent_iclr/qualification/run_qualification.sh --help
sha256sum -c .omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260912-01/payload-final2/payload-sha256.txt
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -m scale.experiments.nonlatent_iclr.qualification.cli validate --output-dir /tmp/nonlatent-final2-review --nonce 00000000000000000000000000000000 --run-id qualification-20260912-01-review000 --controller-receipt /inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/controller-receipts/qualification-20260912-01-review000.json --controller-receipt-timeout-seconds 60 --expected-world-size 8 --timeout-seconds 1200 --nccl-timeout-seconds 120 --memory-fraction 0.90 --manifest scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json --manifest-sha256 92fbf9afe7d0c7b47829337639dae54debddc8e9409c0903f30358aa75aa656c --preflight-receipt /tmp/nonlatent-final2-review/preflight.json
```

Do not submit a qz job or execute CUDA as part of this recheck. Acceptance should remain CPU-payload-only, with image digest explicitly unknown and all GPU/model/cache claims pending the sole authorized run.
